import numpy as np
import cupy as cp  # 引入 CuPy 进行 GPU 加速
import matplotlib.pyplot as plt
from tqdm import tqdm
import time

# ==========================================
# 1. 物理常数与全局配置 (Configuration)
# ==========================================
class LaserConfig:
    def __init__(self):
        # --- 光源参数 ---
        self.lam = 1030e-9       # 波长 (m) (Yb:CAYLO 典型波长)
        self.P_peak = 0.5e6      # 锁模峰值功率 (W) (估算值)
        self.P_cw = 1.0          # 连续光计算功率 (W)

        # --- 晶体参数 1: 增益介质 (Yb:CAYLO) ---
        self.n0_gain = 1.82      # 线性折射率 (估算, YAG/CALGO类约为1.8)
        self.L_gain_crys = 3e-3  # 增益晶体长度 (m)

        # --- 晶体参数 2: 克尔介质 (CaF2) ---
        self.n0_kerr = 1.43      # CaF2 线性折射率
        self.n2_kerr = 1.9e-20   # CaF2 非线性系数 (m^2/W) (典型值)
        self.L_kerr_crys = 3e-3  # 克尔介质长度 (m)

        # --- 腔体几何 (Double-Confocal) ---
        # Gain Section: M1, M2 (R=300mm)
        self.R_gain = 300e-3
        # Kerr Section: M3, M4 (R=100mm)
        self.R_kerr = 100e-3
        
        # 腔臂长度 (总长 ~1.85m)
        # Sequence: HR -> M4 -> KM -> M3 -> Long_Arm -> M2 -> Gain -> M1 -> OC
        
        # 1. Kerr Arm (HR to M4)
        self.L_arm_kerr = 0.25   # (m) 估算
        
        # 2. Middle Long Arm (M3 to M2) - 包含 GTI, DM 等
        self.L_arm_mid = 1.00    # (m) 主要是这一段贡献长度
        
        # 3. Gain Arm (M1 to OC)
        self.L_arm_gain = 0.40   # (m) 估算
        
        # 折叠角 (用于像散计算)
        self.theta_gain = 10.0   # M1/M2 折叠角 (deg)
        self.theta_kerr = 10.0   # M3/M4 折叠角 (deg)

        # --- 仿真网格参数 (BPM) ---
        self.N = 256             # 网格点数
        self.window = 4e-3       # 物理窗口大小 (m) (稍大一点以容纳长臂衍射)
        self.bpm_steps = 10      # 晶体分层切片数量

        # --- 优化参数 ---
        self.max_round_trips = 100 # 最大允许圈数
        self.tolerance = 1e-5      # 早停收敛阈值


# ==========================================
# 2. ABCD 矩阵计算引擎 (CPU - NumPy)
# ==========================================
def get_abcd_lens_reflect(R, theta_deg):
    """ 计算凹面镜反射的 ABCD 矩阵 (即薄透镜近似) """
    theta = np.radians(theta_deg)
    # Tangential (切向): f_t = (R/2) * cos(theta)
    ft = (R / 2) * np.cos(theta)
    # Sagittal (弧矢): f_s = (R/2) / cos(theta)
    fs = (R / 2) / np.cos(theta)
    return ft, fs

def _is_stable_round_trip(M_rt):
    A = M_rt[0, 0]
    D = M_rt[1, 1]
    val = (A + D) / 2.0
    return abs(val) <= 1.0

def calculate_stability_abcd(cfg, z_kerr, x_kerr, z_gain=None):
    """
    计算双共焦腔往返矩阵。
    
    参数:
    - z_kerr: Kerr Section 镜间距 (M3 - M4 直线距离, 含晶体)
    - x_kerr: Kerr 介质中心偏离 M3 的距离 (相对于 z_kerr 中点的位移? 或者直接定义相对 M4 的距离?)
              这里定义: x_kerr = 0 表示介质在 M3-M4 几何中心。
              d_M4_KM = z_kerr/2 - x_kerr - L_kerr/2
              d_KM_M3 = z_kerr/2 + x_kerr - L_kerr/2
    - z_gain: M1-M2 间距。如果为 None，则默认为 R_gain + small_offset (共焦)
    """
    if z_gain is None:
        z_gain = cfg.R_gain + 0.005 # 默认略大于 R (Confocal is R)
        # 注意: R=300mm, f=150mm. Confocal spacing is 2*f = R? 
        # Double focus cavity usually means lens separation is ~ 2*f. 
        # For curved mirrors R, f=R/2. Distance M1-M2 approx R+delta.
        
    # --- 1. 元件参数 ---
    ft_g, fs_g = get_abcd_lens_reflect(cfg.R_gain, cfg.theta_gain)
    ft_k, fs_k = get_abcd_lens_reflect(cfg.R_kerr, cfg.theta_kerr)
    
    # 晶体有效长度 (Effective Length)
    Leff_g_t = cfg.L_gain_crys / (cfg.n0_gain**3)
    Leff_g_s = cfg.L_gain_crys / cfg.n0_gain
    Leff_k_t = cfg.L_kerr_crys / (cfg.n0_kerr**3)
    Leff_k_s = cfg.L_kerr_crys / cfg.n0_kerr
    
    # --- 2. 距离计算 ---
    # Kerr Section (M4 -> KM -> M3)
    # Total air gap = z_kerr - L_kerr_crys
    # d4 = M4 to KM_face
    d4 = z_kerr / 2 - x_kerr - cfg.L_kerr_crys / 2
    # d3 = KM_back to M3
    d3 = z_kerr / 2 + x_kerr - cfg.L_kerr_crys / 2
    
    # Gain Section (M2 -> Gain -> M1)
    # 假设 Gain 晶体在 M1-M2 中心
    d2 = z_gain / 2 - cfg.L_gain_crys / 2
    d1 = z_gain / 2 - cfg.L_gain_crys / 2
    
    # 传输矩阵
    def space(d): return np.array([[1, d], [0, 1]])
    def lens(f):  return np.array([[1, 0], [-1/f, 1]])
    def crys(leff): return np.array([[1, leff], [0, 1]])
    
    def get_half_rt_matrix(f_g, f_k, L_g, L_k):
        # 顺序: HR -> ... -> OC (One Way)
        # 但是我们要算 Round Trip，可以从一点切开。
        # 选 HR 面作为切面: M_rt = M_return @ M_go
        
        # M_go: HR -> M4 -> KM -> M3 -> LongArm -> M2 -> Gain -> M1 -> OC
        
        # Segment 1: HR -> M4
        M_sect1 = space(cfg.L_arm_kerr)
        
        # Atom: M4 (Lens)
        M_M4 = lens(f_k)
        
        # Segment 2: M4 -> KM -> M3
        M_sect2 = M_M4 @ space(d4) @ crys(L_k) @ space(d3)
        
        # Atom: M3 (Lens)
        M_M3 = lens(f_k)
        
        # Segment 3: M3 -> LongArm -> M2
        M_sect3 = M_M3 @ space(cfg.L_arm_mid)
        
        # Atom: M2 (Lens)
        M_M2 = lens(f_g)
        
        # Segment 4: M2 -> Gain -> M1
        M_sect4 = M_M2 @ space(d2) @ crys(L_g) @ space(d1)
        
        # Atom: M1 (Lens)
        M_M1 = lens(f_g)
        
        # Segment 5: M1 -> OC
        M_sect5 = M_M1 @ space(cfg.L_arm_gain)
        
        # Total One Way (Matrix multiply is reverse order: Last applied @ ... @ First applied)
        # Light travels: HR --(Sect1)--> M4 --(Sect2)--> M3 --(Sect3)--> M2 --(Sect4)--> M1 --(Sect5)--> OC
        # So M_way = M_sect5 @ M_sect4 @ M_sect3 @ M_sect2 @ M_sect1
        
        # 注意：以上写法 M_sect2 包含了 M4 和 M3 的 lens 作用吗？
        # M_sect2 是 M4_Lens ... 
        # 重写一下链条以免混淆
        
        # M_go 序列 (Right to Left in math):
        # OC <(L_arm_gain)- M1(L) <(d1)- G -(d2)- M2(L) <(L_mid)- M3(L) <(d3)- K -(d4)- M4(L) <(L_arm_kerr)- HR
        
        M_go = (space(cfg.L_arm_gain) @ M_M1 @ 
                space(d1) @ crys(L_g) @ space(d2) @ M_M2 @ 
                space(cfg.L_arm_mid) @ M_M3 @ 
                space(d3) @ crys(L_k) @ space(d4) @ M_M4 @ 
                space(cfg.L_arm_kerr))
        
        # M_back (Reverse):
        # HR <...- OC
        # 即使是反向，透镜矩阵是一样的(薄透镜)。空间矩阵一样。
        # 顺序反过来。
        M_back = (space(cfg.L_arm_kerr) @ M_M4 @ 
                  space(d4) @ crys(L_k) @ space(d3) @ M_M3 @ 
                  space(cfg.L_arm_mid) @ M_M2 @ 
                  space(d2) @ crys(L_g) @ space(d1) @ M_M1 @ 
                  space(cfg.L_arm_gain))
                  
        return M_back @ M_go

    M_rt_t = get_half_rt_matrix(ft_g, ft_k, Leff_g_t, Leff_k_t)
    M_rt_s = get_half_rt_matrix(fs_g, fs_k, Leff_g_s, Leff_k_s)

    return _is_stable_round_trip(M_rt_t), _is_stable_round_trip(M_rt_s)


# ==========================================
# 3. BPM 引擎 (GPU)
# ==========================================
class BPM_Engine_GPU:
    def __init__(self, cfg):
        self.cfg = cfg
        n = cfg.N

        # 空间网格
        x_cpu = np.linspace(-cfg.window / 2, cfg.window / 2, n)
        self.dx = x_cpu[1] - x_cpu[0]
        self.x_gpu = cp.asarray(x_cpu)
        self.y_gpu = cp.asarray(x_cpu)
        self.X_gpu, self.Y_gpu = cp.meshgrid(self.x_gpu, self.y_gpu)

        # 频率网格
        k_cpu = np.fft.fftfreq(n, d=self.dx) * 2 * np.pi
        self.KX_gpu, self.KY_gpu = cp.meshgrid(cp.asarray(k_cpu), cp.asarray(k_cpu))
        self.k0 = 2 * np.pi / cfg.lam

        # 软边光阑 (放在 OC 处?) -> 随便放哪，反正每圈作用一次
        # 放在 HR 端比较方便计算
        R_mask = cfg.window * 0.45
        self.Absorber_gpu = cp.exp(-((self.X_gpu ** 2 + self.Y_gpu ** 2) / R_mask ** 2) ** 20)

    def propagate(self, E_gpu, dist, n_ref=1.0):
        if dist == 0: return E_gpu
        if n_ref == 1.0:
            eff_dist_t = dist
            eff_dist_s = dist
        else:
            eff_dist_t = dist / (n_ref ** 3)
            eff_dist_s = dist / n_ref

        phase_gpu = -1j * (self.KX_gpu ** 2 * eff_dist_t + self.KY_gpu ** 2 * eff_dist_s) / (2 * self.k0)
        return cp.fft.ifft2(cp.fft.fft2(E_gpu) * cp.exp(phase_gpu))

    def apply_lens(self, E_gpu, f_t, f_s):
        phase_gpu = (self.k0 / 2) * ((self.X_gpu ** 2 / f_t) + (self.Y_gpu ** 2 / f_s))
        return E_gpu * cp.exp(-1j * phase_gpu)

    def apply_kerr(self, E_gpu, dz, n2):
        if n2 == 0: return E_gpu
        I_gpu = cp.abs(E_gpu) ** 2
        phi_gpu = self.k0 * n2 * I_gpu * dz
        return E_gpu * cp.exp(-1j * phi_gpu)
    
    def apply_gain(self, E_gpu, dz, power_norm):
        # 简单线性增益，保持能量守恒 (由外部归一化控制)
        # 这里只做 pump? 不，我们在外部做总能量归一化。
        # 这里仅占位，如果要做 Soft Aperture，需在这里加 gain profile
        return E_gpu

    def get_beam_width_gpu(self, E_gpu):
        I_gpu = cp.abs(E_gpu) ** 2
        total_2d = cp.sum(I_gpu)
        if total_2d.item() == 0.0: return 0.0, 0.0
        Ix = cp.sum(I_gpu, axis=0) 
        Iy = cp.sum(I_gpu, axis=1)

        def calc_width_1d(profile_1d, axis_val_1d):
            norm = cp.sum(profile_1d)
            if norm.item() == 0.0: return 0.0
            mean = cp.sum(profile_1d * axis_val_1d) / norm
            var = cp.sum(profile_1d * (axis_val_1d - mean) ** 2) / norm
            return (2 * cp.sqrt(var)).item()

        return calc_width_1d(Ix, self.x_gpu), calc_width_1d(Iy, self.y_gpu)

    def run_simulation(self, z_kerr, x_kerr, power, z_gain=None, trace_mode=False):
        if z_gain is None:
            z_gain = self.cfg.R_gain + 0.005 # Default
            
        # 准备透镜参数
        ft_k, fs_k = get_abcd_lens_reflect(self.cfg.R_kerr, self.cfg.theta_kerr)
        ft_g, fs_g = get_abcd_lens_reflect(self.cfg.R_gain, self.cfg.theta_gain)
        
        # 准备距离
        d4 = z_kerr / 2 - x_kerr - self.cfg.L_kerr_crys / 2
        d3 = z_kerr / 2 + x_kerr - self.cfg.L_kerr_crys / 2
        d2 = z_gain / 2 - self.cfg.L_gain_crys / 2
        d1 = z_gain / 2 - self.cfg.L_gain_crys / 2
        
        step_k = self.cfg.L_kerr_crys / self.cfg.bpm_steps
        step_g = self.cfg.L_gain_crys / self.cfg.bpm_steps

        # 初始光场 (Start at HR)
        w0 = 50e-6
        E_gpu = cp.exp(-(self.X_gpu ** 2 + self.Y_gpu ** 2) / w0 ** 2) + 0j
        E_gpu *= cp.sqrt(max(power, 1e-3) / (cp.sum(cp.abs(E_gpu) ** 2) * self.dx ** 2))

        def single_round_trip(E_in, record=False):
            d_hist, wx_hist, wy_hist = [], [], []
            curr_d = 0.0
            
            def rec(E, d):
                if record:
                    wx, wy = self.get_beam_width_gpu(E)
                    d_hist.append(d)
                    wx_hist.append(wx)
                    wy_hist.append(wy)
            
            def prop(E, dist, d_acc, n=1.0):
                # 如果不记录或距离很短，直接一步传播
                if not record or dist < 0.02:
                    return self.propagate(E, dist, n), d_acc + dist
                
                # 如果需要记录轨迹 (Trace Mode)，将长距离切分为小段
                step_size = 0.02 # 2cm 分辨率
                rem_dist = dist
                curr_dist = 0.0
                
                curr_E = E
                
                while rem_dist > step_size:
                    curr_E = self.propagate(curr_E, step_size, n)
                    d_acc += step_size
                    rem_dist -= step_size
                    rec(curr_E, d_acc)
                
                # 剩余部分
                if rem_dist > 0:
                    curr_E = self.propagate(curr_E, rem_dist, n)
                    d_acc += rem_dist
                
                return curr_E, d_acc

            rec(E_in, curr_d)
            
            # === Forward Path: HR -> OC ===
            # 1. HR -> M4
            E_in, curr_d = prop(E_in, self.cfg.L_arm_kerr, curr_d)
            rec(E_in, curr_d)
            
            # 2. M4 Lens
            E_in = self.apply_lens(E_in, ft_k, fs_k)
            
            # 3. M4 -> Kerr Crystal
            E_in, curr_d = prop(E_in, d4, curr_d)
            
            # 4. Inside Kerr Crystal
            for _ in range(self.cfg.bpm_steps):
                E_in, curr_d = prop(E_in, step_k, curr_d, self.cfg.n0_kerr)
                if power > 1.0: E_in = self.apply_kerr(E_in, step_k, self.cfg.n2_kerr)
                if record: rec(E_in, curr_d)
            
            # 5. Kerr Crystal -> M3
            E_in, curr_d = prop(E_in, d3, curr_d)
            rec(E_in, curr_d)
            
            # 6. M3 Lens
            E_in = self.apply_lens(E_in, ft_k, fs_k)
            
            # 7. M3 -> M2 (Long Arm)
            E_in, curr_d = prop(E_in, self.cfg.L_arm_mid, curr_d)
            rec(E_in, curr_d)
            
            # 8. M2 Lens
            E_in = self.apply_lens(E_in, ft_g, fs_g)
            
            # 9. M2 -> Gain Crystal
            E_in, curr_d = prop(E_in, d2, curr_d)
            
            # 10. Inside Gain Crystal
            for _ in range(self.cfg.bpm_steps):
                E_in, curr_d = prop(E_in, step_g, curr_d, self.cfg.n0_gain)
                # Linear Gain logic if needed...
                if record: rec(E_in, curr_d)
                
            # 11. Gain Crystal -> M1
            E_in, curr_d = prop(E_in, d1, curr_d)
            rec(E_in, curr_d)
            
            # 12. M1 Lens
            E_in = self.apply_lens(E_in, ft_g, fs_g)
            
            # 13. M1 -> OC
            E_in, curr_d = prop(E_in, self.cfg.L_arm_gain, curr_d)
            rec(E_in, curr_d)
            
            # === Backward Path: OC -> HR ===
            # REVERSE operations
            
            # 13r. OC -> M1
            E_in, curr_d = prop(E_in, self.cfg.L_arm_gain, curr_d)
            E_in = self.apply_lens(E_in, ft_g, fs_g) # M1 Lens
            
            # 11r. M1 -> Gain -> M2
            E_in, curr_d = prop(E_in, d1, curr_d)
            for _ in range(self.cfg.bpm_steps):
                E_in, curr_d = prop(E_in, step_g, curr_d, self.cfg.n0_gain)
            E_in, curr_d = prop(E_in, d2, curr_d)
            
            E_in = self.apply_lens(E_in, ft_g, fs_g) # M2 Lens
            
            # 7r. M2 -> M3 (Long)
            E_in, curr_d = prop(E_in, self.cfg.L_arm_mid, curr_d)
            rec(E_in, curr_d)
            
            E_in = self.apply_lens(E_in, ft_k, fs_k) # M3 Lens
            
            # 5r. M3 -> Kerr -> M4
            E_in, curr_d = prop(E_in, d3, curr_d)
            for _ in range(self.cfg.bpm_steps):
                E_in, curr_d = prop(E_in, step_k, curr_d, self.cfg.n0_kerr)
                if power > 1.0: E_in = self.apply_kerr(E_in, step_k, self.cfg.n2_kerr)
            E_in, curr_d = prop(E_in, d4, curr_d)
            
            E_in = self.apply_lens(E_in, ft_k, fs_k) # M4 Lens
            
            # 1r. M4 -> HR
            E_in, curr_d = prop(E_in, self.cfg.L_arm_kerr, curr_d)
            rec(E_in, curr_d)
            
            return E_in, d_hist, wx_hist, wy_hist

        # Main Loop
        last_w = 0.0
        w_curr = 0.0
        
        for i in range(self.cfg.max_round_trips):
            E_gpu, _, _, _ = single_round_trip(E_gpu, record=False)
            
            # Aperture & Norm
            E_gpu *= self.Absorber_gpu
            curr_en = cp.sum(cp.abs(E_gpu) ** 2) * self.dx ** 2
            E_gpu *= cp.sqrt(max(power, 1e-3) / curr_en)
            
            # Early Stop
            if i > 20 and (i % 5 == 0):
                wx, wy = self.get_beam_width_gpu(E_gpu)
                w_curr = float(np.sqrt(wx * wy))
                if abs(w_curr - last_w) / (w_curr + 1e-12) < self.cfg.tolerance:
                    break
                last_w = w_curr

        if trace_mode:
            E_out, dists, wxs, wys = single_round_trip(E_gpu, record=True)
            # Close loop logic
            E_final = E_out * self.Absorber_gpu
            loss = 1.0 - (cp.sum(cp.abs(E_final)**2) / cp.sum(cp.abs(E_out)**2)).item()
            print(f"   [Trace] Aperture Loss: {loss*100:.2f}%")
            wx, wy = self.get_beam_width_gpu(E_final)
            dists = np.append(dists, dists[-1])
            wxs = np.append(wxs, wx)
            wys = np.append(wys, wy)
            return np.array(dists), np.array(wxs), np.array(wys)
            
        if w_curr == 0.0:
            wx, wy = self.get_beam_width_gpu(E_gpu)
            w_curr = float(np.sqrt(wx * wy))
        return w_curr

# ==========================================
# 4. 主程序 (Main Workflow)
# ==========================================
if __name__ == "__main__":
    try:
        dev_count = cp.cuda.runtime.getDeviceCount()
        device_name = cp.cuda.runtime.getDeviceProperties(0)["name"].decode("utf-8")
        print(f"检测到 GPU: {device_name}")
    except Exception:
        print("未检测到 CUDA 环境。")
        exit()

    cfg = LaserConfig()
    bpm = BPM_Engine_GPU(cfg)
    
    print("\n>>> 开始仿真 Double-Confocal Cavity ...")
    print(f"Gain Mirrors R={cfg.R_gain*1e3}mm, Kerr Mirrors R={cfg.R_kerr*1e3}mm")

    # --- 重新回归 2D 稳定性扫描 (Z_gain vs Z_kerr) ---
    # 原因：固定 Z_gain=302mm 时并未找到稳定点，说明甚至连一个稳区都切不到。
    # 需要先做宽范围的 2D 扫描来定位 "双稳区岛屿" 的位置。
    
    # 1. Gain Section Scan (covering R=300mm to 325mm)
    z_gain_scan = np.linspace(295e-3, 335e-3, 60)
    
    # 2. Kerr Section Scan (covering R=100mm to 160mm)
    z_kerr_scan = np.linspace(70e-3, 170e-3, 80)
    
    # 3. Fix Crystal Position for Stability Map
    fixed_x_kerr = 0.0

    k_map = np.zeros((len(z_gain_scan), len(z_kerr_scan)))
    w_min_map = np.zeros((len(z_gain_scan), len(z_kerr_scan)))

    print(f"正在执行全局稳定性搜索...")
    print(f"扫描 Z_gain: {z_gain_scan[0]*1e3:.1f}-{z_gain_scan[-1]*1e3:.1f} mm")
    print(f"扫描 Z_kerr: {z_kerr_scan[0]*1e3:.1f}-{z_kerr_scan[-1]*1e3:.1f} mm")
    print(f"扫描点数: {len(z_gain_scan)}x{len(z_kerr_scan)}")
    
    start_time = time.time()

    for i, zg in enumerate(tqdm(z_gain_scan, desc="Z_gain")):
        for j, zk in enumerate(z_kerr_scan):
            # ABCD check
            st_t, st_s = calculate_stability_abcd(cfg, zk, fixed_x_kerr, zg)
            if not (st_t and st_s):
                k_map[i, j] = np.nan
                w_min_map[i, j] = np.nan
                continue
            
            try:
                # CW only for stability map
                w_cw = bpm.run_simulation(zk, fixed_x_kerr, power=0, z_gain=zg, trace_mode=False)
                if w_cw <= 0 or np.isnan(w_cw) or w_cw > 500e-6:
                    k_map[i, j] = np.nan
                    w_min_map[i, j] = w_cw
                    continue
                
                # ML check
                w_ml = bpm.run_simulation(zk, fixed_x_kerr, power=cfg.P_peak, z_gain=zg, trace_mode=False)
                k_val = (w_cw - w_ml) / w_cw
                k_map[i, j] = k_val
                w_min_map[i, j] = w_cw
                
            except Exception:
                k_map[i, j] = np.nan
                w_min_map[i, j] = np.nan

    print(f"计算耗时: {time.time() - start_time:.2f} 秒")

    # --- 绘图 ---
    plt.figure(figsize=(16, 7))
    
    # Plot 1: KLM Strength Map
    plt.subplot(1, 2, 1)

    if np.all(np.isnan(k_map)):
        print("错误：未找到任何有效稳定工作点 (w < 500um).")
        # 尝试画一下 w_min_map 看看发生了什么
        plt.imshow(w_min_map, aspect="auto", origin="lower", extent=[z_kerr_scan[0]*1e3, z_kerr_scan[-1]*1e3, x_kerr_scan[0]*1e3, x_kerr_scan[-1]*1e3])
        plt.title("Beam Size Map (No valid KLM point found)")
        plt.colorbar(label="Beam Waist (m)")
        plt.xlabel("Kerr Separation z_kerr (mm)")
        plt.ylabel("Crystal Pos x_kerr (mm)")
        plt.tight_layout()
        plt.show()
        raise SystemExit
        
    extent = [z_kerr_scan[0]*1e3, z_kerr_scan[-1]*1e3, z_gain_scan[0]*1e3, z_gain_scan[-1]*1e3]
    
    # 自动切除边缘坏点 (可选)
    k_plot = k_map.copy()
    
    im = plt.imshow(k_plot, extent=extent, origin="lower", cmap="jet", aspect="auto")
    plt.colorbar(im, label="KLM Strength k")
    plt.title(f"Global Stability Map (R_kerr={cfg.R_kerr*1e3}, R_gain={cfg.R_gain*1e3})")
    plt.xlabel("Kerr Separation z_kerr (mm)")
    plt.ylabel("Gain Separation z_gain (mm)")

    # 找最佳点
    flat_idx = np.nanargmax(k_plot)
    best_idx = np.unravel_index(flat_idx, k_map.shape)
    best_zg = z_gain_scan[best_idx[0]]
    best_zk = z_kerr_scan[best_idx[1]]
    max_k = k_map[best_idx]
    
    print(f"\nBest Operating Point found in Global Map:")
    print(f"  Z_gain = {best_zg*1e3:.4f} mm")
    print(f"  Z_kerr = {best_zk*1e3:.4f} mm")
    print(f"  KLM Strength k = {max_k:.2e}")
    print(f"  CW Spot Size = {w_min_map[best_idx]*1e6:.2f} um")
    
    # Trace Plot
    dist, cw_wx, cw_wy = bpm.run_simulation(best_zk, fixed_x_kerr, power=0, z_gain=best_zg, trace_mode=True)
    _, ml_wx, ml_wy = bpm.run_simulation(best_zk, fixed_x_kerr, power=cfg.P_peak, z_gain=best_zg, trace_mode=True)
    
    ax2 = plt.subplot(1, 2, 2)
    dist_mm = dist * 1e3
    ax2.plot(dist_mm, cw_wx * 1e6, "b--", alpha=0.5, label="CW X")
    ax2.plot(dist_mm, cw_wy * 1e6, "r--", alpha=0.5, label="CW Y")
    ax2.plot(dist_mm, ml_wx * 1e6, "b-", linewidth=2, label="ML X")
    ax2.plot(dist_mm, ml_wy * 1e6, "r-", linewidth=2, label="ML Y")
    
    ax2.set_xlabel("Dist (mm)")
    ax2.set_ylabel("Radius (um)")
    ax2.set_title(f"Intracavity Beam @ Zg={best_zg*1e3:.2f}, Zk={best_zk*1e3:.2f}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
