import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from cupy.fft import fft2, ifft2
from cupyx.scipy.ndimage import zoom, gaussian_filter
import time

# ==========================================
# 1. 配置参数
# ==========================================
c = 2.99792458e8
epsilon0 = 8.854e-12

# --- 核心参数设定 ---
# d_eff: 保持理论上限 2.3 pm/V，确保增益足够补偿仿真损耗
d_eff = 2.05e-12 

class PaperConfig:
    def __init__(self):
        # 基础参数
        self.lambda_p = 532e-9   
        self.lambda_s = 808e-9   
        self.tau_p = 2.2e-9      
        
        # 能量参数 (Paper values)
        self.energy_p1 = 80e-3    
        self.energy_p2 = 370e-3   
        
        # 光斑直径
        self.beam_diam_p1 = 4.0e-3 
        self.beam_diam_p2 = 8.5e-3
        
        # 信号光
        self.super_gaussian_order = 3.0 
        self.tau_s_stretched = 2.55e-9 
        self.bw_s_nm = 100.0     
        self.energy_s_in = 0.2e-9  
        self.beam_diam_s = 5.0e-3  
        
        # 角度配置
        self.alpha_deg = 2.37    
        self.theta_deg = 23.84   
        self.rho_deg = 3.2       
        
        # 晶体长度 (L-type)
        self.L_BBO1 = 23.0e-3
        self.L_BBO2 = 23.0e-3
        # Stage 2:
        self.L_BBO3 = 16.0e-3    
        # 修正: 设定为 10.0mm (论文 Point A 稳定点)
        # 稍微加长有助于在去除 Detuning 后彻底压榨泵浦能量
        self.L_BBO4 = 9.0e-3     
        
        # 分辨率与滤波
        self.N_xy = 512          
        self.XY_window = 16e-3   
        self.dz = 100e-6         
        self.num_slices = 101    
        self.time_window = 6.0e-9 
        self.inter_stage_transmission = 0.85 

# ==========================================
# 2. BBO 材质 (GPU)
# ==========================================
class MaterialBBO_GPU:
    def get_refractive_index(self, lam_array, axis='o'):
        lam_um = lam_array * 1e6
        if axis == 'o':
            n_sq = 2.7359 + 0.01878 / (lam_um**2 - 0.01822) - 0.01354 * lam_um**2
        elif axis == 'e':
            n_sq = 2.3753 + 0.01224 / (lam_um**2 - 0.01667) - 0.01516 * lam_um**2
        return cp.sqrt(n_sq)

    def get_n_eff(self, lam, theta_rad):
        no = self.get_refractive_index(lam, 'o')
        ne = self.get_refractive_index(lam, 'e')
        return 1.0 / cp.sqrt((cp.cos(theta_rad)/no)**2 + (cp.sin(theta_rad)/ne)**2)

    # === 新增：精确计算双折射走离角 rho ===
    def get_birefringent_walkoff(self, lam, theta_rad):
        # 获取主轴折射率
        no = self.get_refractive_index(lam, 'o')
        ne = self.get_refractive_index(lam, 'e')
        
        # === 修正：BBO是负单轴晶体，正确公式分子是 no^2, 分母是 ne^2 ===
        # 原代码: term = (ne**2 / no**2) * cp.tan(theta_rad)  <-- 错误
        term = (no**2 / ne**2) * cp.tan(theta_rad)          # <-- 正确
        
        rho = cp.arctan(term) - theta_rad
        if float(rho) < 0:
            rho = -rho
        return float(rho) # 返回弧度值

# ==========================================
# 3. Time-Slice Solver
# ==========================================
def run_slice_simulation():
    cfg = PaperConfig()
    bbo = MaterialBBO_GPU()
    start_time = time.time()
    
    print(f"--- 2+1D Quasi-Static Simulation (Correction: Removing Phase Detuning) ---")
    print(f"Resolution: {cfg.N_xy}x{cfg.N_xy}, d_eff: {d_eff*1e12} pm/V")
    print(f"L_BBO4: {cfg.L_BBO4*1000} mm (Standard Phase Matching)")
    print(f"Stage 1 Pump: {cfg.energy_p1*1000} mJ")
    print(f"Stage 2 Pump: {cfg.energy_p2*1000} mJ")

    # 1. 基础物理量
    t_axis = np.linspace(-cfg.time_window/2, cfg.time_window/2, cfg.num_slices)
    dt = t_axis[1] - t_axis[0]
    
    omega_p = 2 * np.pi * c / cfg.lambda_p
    omega_s0 = 2 * np.pi * c / cfg.lambda_s
    dw_fwhm = 2 * np.pi * c * (cfg.bw_s_nm * 1e-9) / (cfg.lambda_s**2)
    GDD = cfg.tau_s_stretched / dw_fwhm
    omega_s_list = omega_s0 + t_axis / GDD
    omega_i_list = omega_p - omega_s_list
    
    lam_s_list = 2 * np.pi * c / omega_s_list
    lam_i_list = 2 * np.pi * c / omega_i_list
    
    theta_rad = np.radians(cfg.theta_deg)
    alpha_rad = np.radians(cfg.alpha_deg)
    
    # 1. 计算 BBO 本身的双折射走离 rho_birefringence
    # 注意：只针对泵浦光 (532nm)
    rho_birefringence = bbo.get_birefringent_walkoff(cp.array(cfg.lambda_p), theta_rad)
    print(f"Calculated Birefringent Walk-off (rho): {np.degrees(rho_birefringence):.3f} deg")
    
    # 2. 计算净走离角 Net Walk-off
    # 文献暗示补偿效应，因此取差值。
    # 几何角 alpha 让泵浦往一个方向偏，双折射 rho 让它往回偏。
    # 假设 alpha=2.37 deg, rho ~3.2 deg -> net ~ 0.83 deg
    walkoff_net_rad = rho_birefringence - alpha_rad 
    
    print(f"Geometric Angle (alpha): {cfg.alpha_deg:.3f} deg")
    print(f"Net Effective Walk-off: {np.degrees(walkoff_net_rad):.3f} deg")
    
    # 保留原有的 rho_rad 变量用于兼容性（后续会替换为 walkoff_net_rad）
    rho_rad = np.radians(cfg.rho_deg)
    
    lam_s_cp = cp.array(lam_s_list)
    lam_i_cp = cp.array(lam_i_list)
    
    n_p = float(bbo.get_n_eff(cp.array(cfg.lambda_p), theta_rad))
    kp = n_p * omega_p / c
    
    ns_cp = bbo.get_refractive_index(lam_s_cp, 'o')
    ni_cp = bbo.get_refractive_index(lam_i_cp, 'o')
    ks_cp = ns_cp * cp.array(omega_s_list) / c
    ki_cp = ni_cp * cp.array(omega_i_list) / c
    
    # === 新增/封装函数：计算 Delta k 数组 ===
    def calculate_dk_array(theta_degrees):
        theta_rad_loc = np.radians(theta_degrees)
        n_p_loc = float(bbo.get_n_eff(cp.array(cfg.lambda_p), theta_rad_loc))
        kp_loc = n_p_loc * omega_p / c
        
        # 使用类中定义的参数
        dk_cp_loc = kp_loc * np.cos(alpha_rad) - ks_cp - cp.sqrt(ki_cp**2 - (kp_loc * np.sin(alpha_rad))**2)
        return cp.asnumpy(dk_cp_loc) # 返回 numpy 数组以便在循环中使用
    # =====================================

    # 计算标准 dk (用于 Stage 1 和 BBO3)
    dk_standard_list = calculate_dk_array(cfg.theta_deg)
    
    # 计算失配 dk (专门用于 BBO4)
    # 物理估计：为了增强长波长(840nm)，通常需要略微减小角度或增加角度，取决于晶体轴向
    # 建议尝试 -0.05 到 -0.1 度，或者 +0.05 到 +0.1 度。
    # 根据BBO特性，建议先尝试减小角度
    # 之前 -0.08 导致能量剧降，说明严重偏离了增益中心
    # === 修正：减小失配角，或者尝试正值 ===
    # 建议设置为 -0.02 到 -0.03，或者暂时设为 0.0 先跑通基准
    detuning_angle = -0.04  # 改为更温和的失配
    dk_tuned_list = calculate_dk_array(cfg.theta_deg + detuning_angle)
    
    # 2. 空间网格
    x = cp.linspace(-cfg.XY_window/2, cfg.XY_window/2, cfg.N_xy)
    y = cp.linspace(-cfg.XY_window/2, cfg.XY_window/2, cfg.N_xy)
    X, Y = cp.meshgrid(x, y)
    dx = x[1] - x[0]
    r_sq = X**2 + Y**2
    
    kx = 2 * cp.pi * cp.fft.fftfreq(cfg.N_xy, dx)
    ky = 2 * cp.pi * cp.fft.fftfreq(cfg.N_xy, dx)
    KX, KY = cp.meshgrid(kx, ky)
    K_perp_sq = KX**2 + KY**2

    def get_propagator(k_val, rho, dz):
        diff = -1j * K_perp_sq / (2 * k_val) * dz
        walk = 1j * KY * rho * dz
        return cp.exp(diff + walk)

    # 3. 初始化光斑 (修改部分)
    # 建议：先暂时将 super_gaussian_order 降为 4，或者保持 6 但必须加滤波
    spatial_p1 = cp.exp(-(r_sq / (cfg.beam_diam_p1/2)**2)**cfg.super_gaussian_order)
    spatial_p2 = cp.exp(-(r_sq / (cfg.beam_diam_p2/2)**2)**cfg.super_gaussian_order)
    spatial_s_in = cp.exp(-(r_sq / (cfg.beam_diam_s/2)**2)**cfg.super_gaussian_order)

    # === 新增代码：应用高斯滤波软化硬边 ===
    # sigma=1.0 对应的物理尺寸约为 1-2 个像素，足以消除振荡而不改变光斑大小
    # === 修正：稍微加大 sigma ===
    # 将 sigma 从 1.0 增加到 2.0，确保边缘平滑过渡，防止FFT中的硬边效应
    spatial_p1 = gaussian_filter(spatial_p1, sigma=3.0)
    spatial_p2 = gaussian_filter(spatial_p2, sigma=3.0)
    spatial_s_in = gaussian_filter(spatial_s_in, sigma=3.0)
    # ==================================

    # 4. 归一化振幅
    pump_temp_profile = np.exp(-(t_axis / (cfg.tau_p/2))**4)
    spec_env = np.exp(-4 * np.log(2) * (omega_s_list - omega_s0)**2 / (dw_fwhm)**2)
    signal_temp_profile = np.sqrt(spec_env)

    def get_amplitude(energy, temp_prof, spat_prof, n_idx):
        E_time = np.sum(temp_prof**2) * dt
        E_space = cp.sum(spat_prof**2) * dx**2
        E_total_unit = E_time * E_space * n_idx * epsilon0 * c / 2
        return cp.sqrt(energy / E_total_unit)

    n_s0 = float(bbo.get_refractive_index(cp.array(cfg.lambda_s), 'o'))
    n_i0 = float(bbo.get_refractive_index(cp.array(2*np.pi*c/(omega_p-omega_s0)), 'o'))
    
    Amp_p1 = get_amplitude(cfg.energy_p1, pump_temp_profile, spatial_p1, n_p)
    Amp_p2 = get_amplitude(cfg.energy_p2, pump_temp_profile, spatial_p2, n_p)
    Amp_s_in = get_amplitude(cfg.energy_s_in, signal_temp_profile, spatial_s_in, n_s0)

    # 耦合系数
    kappa_p = 2j * omega_p * d_eff / (n_p * c)
    kappa_s = 2j * omega_s0 * d_eff / (n_s0 * c)
    kappa_i = 2j * (omega_p - omega_s0) * d_eff / (n_i0 * c)

    slice_energy_out = []
    center_slice_fluence = None
    center_slice_idx = cfg.num_slices // 2
    
    print("Running slices...")
    for idx in range(cfg.num_slices):
        if pump_temp_profile[idx] < 1e-3:
            slice_energy_out.append(0.0)
            continue
            
        t_val = t_axis[idx]
        dk_val = dk_standard_list[idx]       # 标准 dk
        dk_val_tuned = dk_tuned_list[idx]    # 调谐后的 dk
        
        # --- Stage 1 ---
        Ep = Amp_p1 * pump_temp_profile[idx] * spatial_p1
        Es = Amp_s_in * signal_temp_profile[idx] * spatial_s_in
        Ei = cp.zeros_like(Es)

        # Propagate Helper
        def propagate_crystal(E_p, E_s, E_i, length, rho, dk):
            steps = int(length / cfg.dz)
            h = cfg.dz
            H_p = get_propagator(kp, rho, h/2)
            H_s = get_propagator(ks_cp[idx], 0, h/2)
            H_i = get_propagator(ki_cp[idx], 0, h/2)
            
            z_loc = 0
            for _ in range(steps):
                E_p = ifft2(fft2(E_p) * H_p)
                E_s = ifft2(fft2(E_s) * H_s)
                E_i = ifft2(fft2(E_i) * H_i)
                
                def get_d(Ap, As, Ai, z):
                    p = cp.exp(1j * dk * z)
                    pc = cp.conj(p)
                    return (kappa_p*As*Ai*pc, kappa_s*Ap*cp.conj(Ai)*p, kappa_i*Ap*cp.conj(As)*p)

                k1p, k1s, k1i = get_d(E_p, E_s, E_i, z_loc)
                k2p, k2s, k2i = get_d(E_p+h*k1p/2, E_s+h*k1s/2, E_i+h*k1i/2, z_loc+h/2)
                k3p, k3s, k3i = get_d(E_p+h*k2p/2, E_s+h*k2s/2, E_i+h*k2i/2, z_loc+h/2)
                k4p, k4s, k4i = get_d(E_p+h*k3p, E_s+h*k3s, E_i+h*k3i, z_loc+h)
                
                E_p += h/6*(k1p+2*k2p+2*k3p+k4p)
                E_s += h/6*(k1s+2*k2s+2*k3s+k4s)
                E_i += h/6*(k1i+2*k2i+2*k3i+k4i)
                z_loc += h
                
                E_p = ifft2(fft2(E_p) * H_p)
                E_s = ifft2(fft2(E_s) * H_s)
                E_i = ifft2(fft2(E_i) * H_i)
            return E_p, E_s, E_i

        # Run Stage 1
        Ep, Es, Ei = propagate_crystal(Ep, Es, Ei, cfg.L_BBO1, walkoff_net_rad, dk_val)
        Ei = cp.zeros_like(Ei) 
        Ep, Es, Ei = propagate_crystal(Ep, Es, Ei, cfg.L_BBO2, -walkoff_net_rad, dk_val)

        # --- Inter-Stage ---
        Es *= np.sqrt(cfg.inter_stage_transmission)
        
        # Expander
        mag_factor = cfg.beam_diam_p2 / cfg.beam_diam_p1 
        if mag_factor > 1.0:
            # === 修正：改回 order=1 (线性插值) ===
            # order=3 虽然平滑，但对陡峭边缘会产生振荡(Over-shoot)
            # order=1 虽然有棱角，但绝对稳定，不会产生同心圆纹路
            Es_real = zoom(Es.real, mag_factor, order=1) 
            Es_imag = zoom(Es.imag, mag_factor, order=1)
            
            Es_expanded = Es_real + 1j * Es_imag
            Es_expanded /= mag_factor 
            
            new_size = Es_expanded.shape[0]
            start_idx = (new_size - cfg.N_xy) // 2
            end_idx = start_idx + cfg.N_xy
            Es = Es_expanded[start_idx:end_idx, start_idx:end_idx]
            if Es.shape[0] != cfg.N_xy:
                 Es = Es[:cfg.N_xy, :cfg.N_xy]

        # Spatial Filter (Gaussian Blur) - KEEP THIS
        Es_real = gaussian_filter(Es.real, sigma=3.0)
        Es_imag = gaussian_filter(Es.imag, sigma=3.0)
        Es = Es_real + 1j * Es_imag

        Ei = cp.zeros_like(Ei)
        Ep = Amp_p2 * pump_temp_profile[idx] * spatial_p2
        
        # --- Stage 2 ---
        Ep, Es, Ei = propagate_crystal(Ep, Es, Ei, cfg.L_BBO3, walkoff_net_rad, dk_val) # BBO3 使用标准匹配
        Ei = cp.zeros_like(Ei)
        
        # === 修改处：BBO4 使用调谐后的 dk ===
        # 注意：这里传入 dk_val_tuned
        Ep, Es, Ei = propagate_crystal(Ep, Es, Ei, cfg.L_BBO4, -walkoff_net_rad, dk_val_tuned)
        
        power_s = cp.sum(cp.abs(Es)**2) * dx**2 * n_s0 * epsilon0 * c / 2
        slice_energy_out.append(float(power_s * dt))
        
        if idx == center_slice_idx:
            center_slice_fluence = cp.abs(Es)**2

        print(f"Slice {idx}/{cfg.num_slices} done.", end='\r')

    total_energy_out = sum(slice_energy_out)
    print(f"\n\nTotal Output Energy: {total_energy_out*1000:.2f} mJ")
    print(f"Sim Time: {time.time()-start_time:.2f} s")
    
    slice_energy_arr = np.array(slice_energy_out)
    spec_intensity = slice_energy_arr / np.max(slice_energy_arr)
    
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    if center_slice_fluence is not None:
        fl_cpu = cp.asnumpy(center_slice_fluence)
        extent = [-cfg.XY_window/2*1e3, cfg.XY_window/2*1e3, -cfg.XY_window/2*1e3, cfg.XY_window/2*1e3]
        plt.imshow(fl_cpu, extent=extent, cmap='jet', origin='lower')
        plt.colorbar(label='Intensity (a.u.)')
        plt.title(f"Center Slice Beam Profile\n(t=0, {cfg.lambda_s*1e9:.1f}nm)")
        plt.xlabel("x (mm)")
        plt.ylabel("y (mm)")
    
    plt.subplot(1, 2, 2)
    plt.plot(lam_s_list * 1e9, spec_intensity, 'r-', linewidth=2)
    plt.title(f"Output Spectrum (E = {total_energy_out*1000:.1f} mJ)")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Normalized Intensity")
    plt.grid(True)
    plt.xlim(750, 860) 
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_slice_simulation()
