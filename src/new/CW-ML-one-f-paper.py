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
        self.lam = 800e-9        # 波长 (m)
        self.P_peak = 0.6e6      # 锁模峰值功率 (W)
        self.P_cw = 1.0          # 连续光计算功率 (W)
        
        # --- 晶体参数 (Ti:Sapphire) ---
        self.n0 = 1.76           # 线性折射率 
        self.n2 = 3e-20          # 非线性系数 (m^2/W)
        self.L_crys = 5e-3       # 晶体物理长度 (m)
        
        # --- 腔体几何 (Z-Cavity) ---
        self.R = 100e-3          # 凹面镜曲率半径 (m)
        self.L1 = 0.60           # 短臂长 (m)
        self.L2 = 0.80           # 长臂长 (m)
        
        # --- 仿真网格参数 (BPM) ---
        self.N = 256             # 网格点数
        self.window = 3e-3       # 物理窗口大小 (m)
        self.bpm_steps = 10      # 晶体分层切片数量
        
        # --- 优化参数 ---
        self.max_round_trips = 50 # 最大允许圈数 (设有早停，数值大点无所谓)
        self.tolerance = 1e-5      # 早停收敛阈值

# ==========================================
# 2. ABCD 矩阵计算引擎 (CPU - NumPy)
# ==========================================
def get_abcd_elements(cfg, theta_deg):
    theta = np.radians(theta_deg)
    ft = (cfg.R / 2) * np.cos(theta)
    fs = (cfg.R / 2) / np.cos(theta)
    return ft, fs

def calculate_stability_abcd(cfg, z, x, theta_deg):
    ft, fs = get_abcd_elements(cfg, theta_deg)
    
    d_a = z/2 + x - cfg.L_crys/2
    d_b = z/2 - x - cfg.L_crys/2
    
    L_eff_t = cfg.L_crys / (cfg.n0**3)
    L_eff_s = cfg.L_crys / cfg.n0
    
    M_cry_t = np.array([[1, L_eff_t], [0, 1]])
    M_cry_s = np.array([[1, L_eff_s], [0, 1]])
    
    M_da = np.array([[1, d_a], [0, 1]])
    M_db = np.array([[1, d_b], [0, 1]])
    M_L1 = np.array([[1, cfg.L1], [0, 1]])
    M_L2 = np.array([[1, cfg.L2], [0, 1]])
    
    def get_one_way_matrix(f_val, M_cry_spec):
        M_f = np.array([[1, 0], [-1/f_val, 1]])
        return M_L2 @ M_f @ M_db @ M_cry_spec @ M_da @ M_f @ M_L1

    M_t = get_one_way_matrix(ft, M_cry_t)
    M_s = get_one_way_matrix(fs, M_cry_s)
    
    stable_t = (0 <= M_t[0, 0] * M_t[1, 1] <= 1)
    stable_s = (0 <= M_s[0, 0] * M_s[1, 1] <= 1)
    return stable_t, stable_s

def find_optimal_theta(cfg):
    print(">>> [Step 1] (CPU) 正在优化折叠角 Theta ...")
    RHS = 2 * cfg.L_crys * (cfg.n0**2 - 1) / (cfg.n0**3)
    th_test = np.linspace(0.1, 45, 1000)
    lhs_vals = cfg.R * np.sin(np.radians(th_test)) * np.tan(np.radians(th_test))
    theta_th = th_test[np.argmin(np.abs(lhs_vals - RHS))]
    
    thetas = np.linspace(theta_th - 2.0, theta_th + 2.0, 40)
    z_scan = np.linspace(cfg.R - 0.005, cfg.R + 0.015, 200) 
    
    best_theta = theta_th 
    min_overlap_diff = 1e9
    
    for th in thetas:
        st_t = [calculate_stability_abcd(cfg, z, 0, th)[0] for z in z_scan]
        st_s = [calculate_stability_abcd(cfg, z, 0, th)[1] for z in z_scan]
        
        edges_t = np.where(np.diff(np.array(st_t, int)) != 0)[0]
        edges_s = np.where(np.diff(np.array(st_s, int)) != 0)[0]
        
        if len(edges_t) > 0 and len(edges_s) > 0:
            diff_matrix = np.abs(z_scan[edges_t][:, None] - z_scan[edges_s])
            current_diff = np.min(diff_matrix)
            if current_diff < min_overlap_diff:
                min_overlap_diff = current_diff
                best_theta = th

    print(f"   ✅ 优化后最佳折叠角: {best_theta:.2f} 度")
    return best_theta

# ==========================================
# 3. BPM 引擎 (GPU - 性能优化版)
# ==========================================
class BPM_Engine_GPU:
    def __init__(self, cfg):
        self.cfg = cfg
        n = cfg.N
        
        # 预计算坐标网格 (减少循环内的内存分配)
        x_cpu = np.linspace(-cfg.window/2, cfg.window/2, n)
        self.dx = x_cpu[1] - x_cpu[0]
        self.x_gpu = cp.array(x_cpu)
        self.y_gpu = cp.array(x_cpu)
        self.X_gpu, self.Y_gpu = cp.meshgrid(self.x_gpu, self.y_gpu)
        
        # 预计算频率域网格
        k_cpu = np.fft.fftfreq(n, d=self.dx) * 2 * np.pi
        self.KX_gpu, self.KY_gpu = cp.meshgrid(cp.array(k_cpu), cp.array(k_cpu))
        self.k0 = 2 * np.pi / cfg.lam
        
        # 软边光阑
        R_mask = cfg.window * 0.45
        self.Absorber_gpu = cp.exp( -((self.X_gpu**2 + self.Y_gpu**2)/R_mask**2)**20 )

    def propagate(self, E_gpu, dist, n_ref=1.0):
        if n_ref == 1.0:
            eff_dist_t = dist
            eff_dist_s = dist
        else:
            eff_dist_t = dist / (n_ref**3)
            eff_dist_s = dist / n_ref

        # 使用预计算的 Grid
        phase_gpu = -1j * (self.KX_gpu**2 * eff_dist_t + self.KY_gpu**2 * eff_dist_s) / (2 * self.k0)
        return cp.fft.ifft2(cp.fft.fft2(E_gpu) * cp.exp(phase_gpu))

    def apply_lens(self, E_gpu, f_t, f_s):
        phase_gpu = (self.k0 / 2) * ( (self.X_gpu**2 / f_t) + (self.Y_gpu**2 / f_s) )
        return E_gpu * cp.exp(-1j * phase_gpu)

    def apply_kerr(self, E_gpu, dz):
        I_gpu = cp.abs(E_gpu)**2
        phi_gpu = self.k0 * self.cfg.n2 * I_gpu * dz
        return E_gpu * cp.exp(-1j * phi_gpu)

    def get_beam_width_gpu(self, E_gpu):
        # 快速计算 D4Sigma
        I_gpu = cp.abs(E_gpu)**2
        total = cp.sum(I_gpu)
        if total == 0: return 0.0, 0.0
        
        # 利用 GPU 归约加速
        Ix = cp.sum(I_gpu, axis=0)
        Iy = cp.sum(I_gpu, axis=1)
        
        def calc_width(profile, axis_val):
            mean = cp.sum(profile * axis_val) / total
            var = cp.sum(profile * (axis_val - mean)**2) / total
            return 2 * cp.sqrt(var)
            
        return calc_width(Ix, self.x_gpu).item(), calc_width(Iy, self.y_gpu).item()

    def run_simulation(self, z, x, theta, power, trace_mode=False):
        ft, fs = get_abcd_elements(self.cfg, theta)
        d_a = z/2 + x - self.cfg.L_crys/2
        d_b = z/2 - x - self.cfg.L_crys/2
        
        # 初始光场
        w0 = 30e-6
        E_gpu = cp.exp(-(self.X_gpu**2 + self.Y_gpu**2)/w0**2) + 0j
        E_gpu *= cp.sqrt(max(power, 1e-3) / (cp.sum(cp.abs(E_gpu)**2) * self.dx**2))
        
        # 预计算常用传播算子
        step = (self.cfg.L_crys / 2) / self.cfg.bpm_steps
        step_full = self.cfg.L_crys / self.cfg.bpm_steps

        def single_round_trip(E_in_gpu, record=False):
            d_hist, wx_hist, wy_hist = [], [], []
            curr_d = 0.0
            
            def rec(E, d):
                if record:
                    wx, wy = self.get_beam_width_gpu(E)
                    d_hist.append(d); wx_hist.append(wx); wy_hist.append(wy)
            
            # 空气段传播函数：普通模式一步到位，绘图模式分步走
            def prop_air(E, dist, d_acc):
                if not record:
                    return self.propagate(E, dist), d_acc + dist
                else:
                    rem, s_sz = dist, 10e-3 # 10mm 步长用于绘图
                    while rem > 1e-6:
                        th_s = min(rem, s_sz)
                        E = self.propagate(E, th_s)
                        rem -= th_s; d_acc += th_s
                        rec(E, d_acc)
                    return E, d_acc

            if record: rec(E_in_gpu, curr_d)

            # 1. Crystal Back
            for _ in range(self.cfg.bpm_steps):
                E_in_gpu = self.propagate(E_in_gpu, step, self.cfg.n0)
                if power > 1.0: E_in_gpu = self.apply_kerr(E_in_gpu, step)
                if record: curr_d += step; rec(E_in_gpu, curr_d)
            
            # 2. Arm 2 (Long Arm)
            E_in_gpu, curr_d = prop_air(E_in_gpu, d_b, curr_d)
            E_in_gpu = self.apply_lens(E_in_gpu, ft, fs)
            E_in_gpu, curr_d = prop_air(E_in_gpu, self.cfg.L2, curr_d) # To End
            E_in_gpu, curr_d = prop_air(E_in_gpu, self.cfg.L2, curr_d) # Return
            E_in_gpu = self.apply_lens(E_in_gpu, ft, fs)
            E_in_gpu, curr_d = prop_air(E_in_gpu, d_b, curr_d)

            # 3. Crystal Full
            for _ in range(self.cfg.bpm_steps):
                E_in_gpu = self.propagate(E_in_gpu, step_full, self.cfg.n0)
                if power > 1.0: E_in_gpu = self.apply_kerr(E_in_gpu, step_full)
                if record: curr_d += step_full; rec(E_in_gpu, curr_d)

            # 4. Arm 1 (Short Arm)
            E_in_gpu, curr_d = prop_air(E_in_gpu, d_a, curr_d)
            E_in_gpu = self.apply_lens(E_in_gpu, ft, fs)
            E_in_gpu, curr_d = prop_air(E_in_gpu, self.cfg.L1, curr_d) # To OC
            E_in_gpu, curr_d = prop_air(E_in_gpu, self.cfg.L1, curr_d) # Return
            E_in_gpu = self.apply_lens(E_in_gpu, ft, fs)
            E_in_gpu, curr_d = prop_air(E_in_gpu, d_a, curr_d)

            # 5. Crystal Front
            for _ in range(self.cfg.bpm_steps):
                E_in_gpu = self.propagate(E_in_gpu, step, self.cfg.n0)
                if power > 1.0: E_in_gpu = self.apply_kerr(E_in_gpu, step)
                if record: curr_d += step; rec(E_in_gpu, curr_d)
            
            return E_in_gpu, d_hist, wx_hist, wy_hist

        # --- 主迭代循环 (引入早停机制) ---
        last_w = 0
        w_curr = 0
        
        for i in range(self.cfg.max_round_trips):
            # 只在最后一圈开启 Trace 记录
            is_last = (i == self.cfg.max_round_trips - 1)
            E_gpu, _, _, _ = single_round_trip(E_gpu, record=(trace_mode and is_last))
            
            # 能量归一化
            E_gpu *= self.Absorber_gpu
            curr_en = cp.sum(cp.abs(E_gpu)**2) * self.dx**2
            E_gpu *= cp.sqrt(max(power, 1e-3) / curr_en)
            
            # [优化] 早停检测：每 5 圈检查一次
            if not trace_mode and i > 20 and i % 5 == 0:
                wx, wy = self.get_beam_width_gpu(E_gpu)
                w_curr = np.sqrt(wx*wy)
                if abs(w_curr - last_w) / (w_curr + 1e-12) < self.cfg.tolerance:
                    break # 收敛，提前结束
                last_w = w_curr

        if trace_mode:
            # 强制运行一圈 trace 模式来获取绘图数据
            _, dists, wxs, wys = single_round_trip(E_gpu, record=True)
            return np.array(dists), np.array(wxs), np.array(wys)
        else:
            if w_curr == 0: # 如果循环还没触发计算
                wx, wy = self.get_beam_width_gpu(E_gpu)
                w_curr = np.sqrt(wx*wy)
            return w_curr

# ==========================================
# 4. 主程序 (Main Workflow)
# ==========================================
if __name__ == "__main__":
    try:
        dev_count = cp.cuda.runtime.getDeviceCount()
        device_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode('utf-8')
        print(f"✅ 检测到 GPU: {device_name}")
    except:
        print("❌ 未检测到 CUDA 环境。"); exit()

    cfg = LaserConfig()
    bpm = BPM_Engine_GPU(cfg)
    best_theta = find_optimal_theta(cfg)
    
    # --- 优化后的扫描参数 ---
    # Z轴: 100点 (高分辨率，保证精度)
    z_scan = np.linspace(cfg.R + 0.002, cfg.R + 0.012, 100)
    # X轴: 5点 (低分辨率，足够确定中心)
    x_scan = np.linspace(-2e-3, 2e-3, 20) 
    
    k_map = np.zeros((len(x_scan), len(z_scan)))
    print(f"\n>>> [Step 2] 计算稳定性图谱 (100x5 Grid, Early Stopping Enabled)...")
    
    start_time = time.time()
    
    for i, x in enumerate(tqdm(x_scan, desc="Scanning")):
        for j, z in enumerate(z_scan):
            st_t, st_s = calculate_stability_abcd(cfg, z, x, best_theta)
            if not (st_t and st_s):
                k_map[i, j] = np.nan
                continue
            try:
                w_cw = bpm.run_simulation(z, x, best_theta, power=0)
                w_ml = bpm.run_simulation(z, x, best_theta, power=cfg.P_peak)
                if w_cw > 0:
                    k_map[i, j] = (w_cw - w_ml) / w_cw
                else:
                    k_map[i, j] = np.nan
            except:
                k_map[i, j] = np.nan

    print(f"✅ 计算耗时: {time.time() - start_time:.2f} 秒")

    # --- 绘图部分 (完整还原) ---
    plt.figure(figsize=(16, 7))
    plt.subplot(1, 2, 1)
    
    if np.all(np.isnan(k_map)):
        print("错误：未找到任何稳定工作点。")
    else:
        extent = [z_scan[0]*1e3, z_scan[-1]*1e3, x_scan[0]*1e3, x_scan[-1]*1e3]
        im = plt.imshow(k_map, extent=extent, origin='lower', cmap='jet', aspect='auto', interpolation='bilinear')
        plt.colorbar(im, label='KLM Strength k')
        plt.title(f'KLM Stability Map\n(Z_res=0.1mm, Early Stop)')
        plt.xlabel('Z (mm)'); plt.ylabel('X (mm)')
        
        # --- 边缘切除逻辑 ---
        k_search = k_map.copy()
        edge_cut = 10 # 配合 100 点精度，切除 1mm
        
        for i in range(k_map.shape[0]):
            valid = np.where(~np.isnan(k_map[i, :]))[0]
            if len(valid) > 2 * edge_cut:
                k_search[i, valid[:edge_cut]] = np.nan
                k_search[i, valid[-edge_cut:]] = np.nan
            else:
                k_search[i, :] = np.nan

        if np.all(np.isnan(k_search)):
            print("⚠️ 警告: 稳区太窄，已回退到无切除模式")
            flat_idx = np.nanargmax(k_map)
        else:
            flat_idx = np.nanargmax(k_search)
            
        best_idx = np.unravel_index(flat_idx, k_map.shape)
        best_z = z_scan[best_idx[1]]
        best_x = x_scan[best_idx[0]]
        max_k = k_map[best_idx]
        
        print(f"\n🌟 最佳工作点 (安全区内部): z={best_z*1e3:.3f}mm, x={best_x*1e3:.3f}mm, k={max_k:.4f}")
        
        # 运行一次详细追踪 (Trace Mode)
        bpm.run_simulation(best_z, best_x, best_theta, power=0, trace_mode=False) # 预热
        dist, cw_wx, cw_wy = bpm.run_simulation(best_z, best_x, best_theta, power=0, trace_mode=True)
        _, ml_wx, ml_wy = bpm.run_simulation(best_z, best_x, best_theta, power=cfg.P_peak, trace_mode=True)
        
        ax2 = plt.subplot(1, 2, 2)
        dist_mm = dist * 1e3
        
        # 绘制曲线
        ax2.plot(dist_mm, cw_wx*1e6, 'b--', alpha=0.5, label='CW Tangential (X)')
        ax2.plot(dist_mm, ml_wx*1e6, 'b-', label='ML Tangential (X)', linewidth=2)
        ax2.plot(dist_mm, cw_wy*1e6, 'r--', alpha=0.5, label='CW Sagittal (Y)')
        ax2.plot(dist_mm, ml_wy*1e6, 'r-', label='ML Sagittal (Y)', linewidth=2)
        
        # --- [还原] 器件色块示意 (完整版) ---
        d_a = (best_z/2 + best_x - cfg.L_crys/2) * 1e3
        d_b = (best_z/2 - best_x - cfg.L_crys/2) * 1e3
        L1_mm = cfg.L1 * 1e3
        L2_mm = cfg.L2 * 1e3
        Lc_half = (cfg.L_crys/2) * 1e3
        
        # 0. Crystal Center
        current = 0
        # 1. Crystal Back
        p_cry1_end = current + Lc_half
        ax2.axvspan(current, p_cry1_end, color='gray', alpha=0.2, label='Crystal')
        current = p_cry1_end
        # 2. M3 (Gap to M2 mirror)
        p_m2 = current + d_b
        ax2.axvline(p_m2, color='cyan', linestyle=':', alpha=0.8, linewidth=2, label='Curved Mirror')
        current = p_m2
        # 3. Long Arm to M3
        p_end_mirror = current + L2_mm
        ax2.axvline(p_end_mirror, color='orange', linestyle='-', alpha=0.8, linewidth=3, label='Flat Mirror (End/OC)')
        current = p_end_mirror
        # 4. Return Long Arm
        p_m2_ret = current + L2_mm
        ax2.axvline(p_m2_ret, color='cyan', linestyle=':', alpha=0.8, linewidth=2)
        current = p_m2_ret
        # 5. Return Gap + Crystal Full
        p_cry_full_start = current + d_b
        p_cry_full_end = p_cry_full_start + 2*Lc_half
        ax2.axvspan(p_cry_full_start, p_cry_full_end, color='gray', alpha=0.2)
        current = p_cry_full_end
        # 6. M1 (Gap to M1 mirror)
        p_m1 = current + d_a
        ax2.axvline(p_m1, color='cyan', linestyle=':', alpha=0.8, linewidth=2)
        current = p_m1
        # 7. Short Arm to OC
        p_oc = current + L1_mm
        ax2.axvline(p_oc, color='orange', linestyle='-', alpha=0.8, linewidth=3)
        current = p_oc
        
        handles, labels = ax2.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax2.legend(by_label.values(), by_label.keys(), loc='upper right')
        
        ax2.set_title(f'Intracavity Beam Evolution (z={best_z*1e3:.2f}mm)')
        ax2.set_xlabel('Propagation Distance (mm)')
        ax2.set_ylabel('Beam Radius (um)')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    print("Done.")