import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from cupy.fft import fft2, ifft2
from cupyx.scipy.ndimage import zoom  # <--- 新增这行
import time

# ==========================================
# 1. 配置参数
# ==========================================
c = 2.99792458e8
epsilon0 = 8.854e-12
d_eff = 2.0e-12 

class PaperConfig:
    def __init__(self):
        # 基础参数
        self.lambda_p = 532e-9   
        self.lambda_s = 808e-9   
        self.tau_p = 2.2e-9      
        
        # --- 修正 1: 泵浦光能量分配 ---
        # 论文: Stage 1 = 80 mJ, Stage 2 = 370 mJ
        self.energy_p1 = 80e-3    
        self.energy_p2 = 370e-3   
        
        self.beam_diam_p1 = 4.0e-3 # 论文提及 Stage 1 泵浦光斑较小 (Fig 3 caption mention 4mm?) 
        # 论文 Page 3: "pump beam apertures... for OPCPA stages 1 and 2 are 4 mm and 8.5 mm"
        self.beam_diam_p2 = 8.5e-3
        
        self.super_gaussian_order = 6 
        
        self.tau_s_stretched = 2.55e-9 
        self.bw_s_nm = 100.0     
        
        # --- 修正 2: 种子光能量 ---
        # 论文: Oscillator(2nJ) -> Stretcher -> 0.2 nJ
        self.energy_s_in = 0.2e-9  
        # [cite: 118] "initial signal beam aperture at FWHM is 5 mm"
        # 原代码是 4.0e-3，需改为 5.0e-3
        self.beam_diam_s = 5.0e-3  # 种子光在 Stage 1 通常匹配泵浦光斑 (4mm)
        
        # Stage 2 信号光扩束后直径 (匹配 Stage 2 泵浦)
        self.beam_diam_s_stage2 = 8.5e-3

        self.alpha_deg = 2.37    
        # --- 修正 3: 角度微调 ---
        self.theta_deg = 23.84   # 论文设计值
        self.rho_deg = 3.2       
        
        # --- Stage 1 Crystals (Total 46mm split into 23+23 for L-type) ---
        self.L_BBO1 = 23.0e-3
        self.L_BBO2 = 23.0e-3
        
        # --- Stage 2 Crystals (16+9 optimized) ---
        self.L_BBO3 = 16.0e-3    
        self.L_BBO4 = 9.0e-3     
        
        self.N_xy = 128          
        self.XY_window = 16e-3   # 扩大窗口，留出更多余量，防止边界反射
        self.dz = 150e-6         # 稍微减小步长提高精度
        self.num_slices = 101    
        self.time_window = 6.0e-9 
        
        # 级间损耗 (Beam Expander + Filter)
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

# ==========================================
# 3. Time-Slice Solver
# ==========================================
def run_slice_simulation():
    cfg = PaperConfig()
    bbo = MaterialBBO_GPU()
    start_time = time.time()
    
    print(f"--- 2+1D Quasi-Static Simulation (Corrected Physics) ---")
    print(f"Stage 1 Pump: {cfg.energy_p1*1000} mJ, Seed: {cfg.energy_s_in*1e9} nJ")
    print(f"Stage 2 Pump: {cfg.energy_p2*1000} mJ")

    # 1. 时间轴与频率参数 (保持不变)
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
    
    # 2. 计算 dk(t) (保持不变)
    theta_rad = np.radians(cfg.theta_deg)
    alpha_rad = np.radians(cfg.alpha_deg)
    rho_rad = np.radians(cfg.rho_deg)
    
    lam_s_cp = cp.array(lam_s_list)
    lam_i_cp = cp.array(lam_i_list)
    
    n_p = float(bbo.get_n_eff(cp.array(cfg.lambda_p), theta_rad))
    kp = n_p * omega_p / c
    
    ns_cp = bbo.get_refractive_index(lam_s_cp, 'o')
    ni_cp = bbo.get_refractive_index(lam_i_cp, 'o')
    ks_cp = ns_cp * cp.array(omega_s_list) / c
    ki_cp = ni_cp * cp.array(omega_i_list) / c
    
    dk_cp = kp * np.cos(alpha_rad) - ks_cp - cp.sqrt(ki_cp**2 - (kp * np.sin(alpha_rad))**2)
    dk_list = cp.asnumpy(dk_cp)
    
    # 3. 空间网格 (GPU)
    x = cp.linspace(-cfg.XY_window/2, cfg.XY_window/2, cfg.N_xy)
    y = cp.linspace(-cfg.XY_window/2, cfg.XY_window/2, cfg.N_xy)
    X, Y = cp.meshgrid(x, y)
    dx = x[1] - x[0]
    r_sq = X**2 + Y**2
    
    kx = 2 * cp.pi * cp.fft.fftfreq(cfg.N_xy, dx)
    ky = 2 * cp.pi * cp.fft.fftfreq(cfg.N_xy, dx)
    KX, KY = cp.meshgrid(kx, ky)
    K_perp_sq = KX**2 + KY**2

    # 4. Propagator 生成器
    def get_propagator(k_val, rho, dz):
        diff = -1j * K_perp_sq / (2 * k_val) * dz
        walk = 1j * KY * rho * dz
        return cp.exp(diff + walk)

    # 5. 准备两种泵浦光和初始种子光的空间分布
    # Pump 1 (4mm)
    spatial_p1 = cp.exp(-(r_sq / (cfg.beam_diam_p1/2)**2)**cfg.super_gaussian_order)
    # Pump 2 (8.5mm)
    spatial_p2 = cp.exp(-(r_sq / (cfg.beam_diam_p2/2)**2)**cfg.super_gaussian_order)
    # Seed (5mm, Super Gaussian)
    # [cite: 115] "The spatial ... profiles of the signal pulse are super Gaussian"
    # 原代码: spatial_s_in = cp.exp(-(r_sq / (cfg.beam_diam_s/2)**2))
    spatial_s_in = cp.exp(-(r_sq / (cfg.beam_diam_s/2)**2)**cfg.super_gaussian_order)

    # 6. 计算归一化振幅 (Amplitudes)
    pump_temp_profile = np.exp(-(t_axis / (cfg.tau_p/2))**4)
    spec_env = np.exp(-4 * np.log(2) * (omega_s_list - omega_s0)**2 / (dw_fwhm)**2)
    signal_temp_profile = np.sqrt(spec_env)

    # Helper: Normalize Energy -> Amplitude
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
    kappa_p = 1j * omega_p * d_eff / (n_p * c)
    kappa_s = 1j * omega_s0 * d_eff / (n_s0 * c)
    kappa_i = 1j * (omega_p - omega_s0) * d_eff / (n_i0 * c)

    slice_energy_out = []
    
    # 记录中心切片的光斑用于绘图
    center_slice_idx = cfg.num_slices // 2
    center_slice_fluence = None
    
    print("Running slices...")
    for idx in range(cfg.num_slices):
        if pump_temp_profile[idx] < 1e-3:
            slice_energy_out.append(0.0)
            continue
            
        t_val = t_axis[idx]
        dk_val = dk_list[idx]
        
        # ==========================================================
        # STAGE 1: High Gain Pre-Amp (BBO1 + BBO2)
        # ==========================================================
        
        # Input: Oscillator Seed (0.2 nJ) + Pump 1
        Ep = Amp_p1 * pump_temp_profile[idx] * spatial_p1
        Es = Amp_s_in * signal_temp_profile[idx] * spatial_s_in
        Ei = cp.zeros_like(Es)

        # --- Propagate Function (Same as before) ---
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

        # 1. BBO1 Propagate (Walk-off: +rho)
        Ep, Es, Ei = propagate_crystal(Ep, Es, Ei, cfg.L_BBO1, rho_rad, dk_val)
        
        # 2. 'L' Configuration: DUMP IDLER
        Ei = cp.zeros_like(Ei) 
        
        # 3. BBO2 Propagate (Walk-off: -rho for compensation)
        Ep, Es, Ei = propagate_crystal(Ep, Es, Ei, cfg.L_BBO2, -rho_rad, dk_val)

        # ==========================================================
        # INTER-STAGE: Expansion & Filtering
        # ==========================================================
        
        # 1. 级间传输损耗
        Es *= np.sqrt(cfg.inter_stage_transmission)
        
        # 2. 光束扩束 (Beam Expansion) - 关键修正
        # 目标：将 Stage 1 的细光束 (4mm) 放大以匹配 Stage 2 的泵浦光 (8.5mm)
        # [cite_start]论文依据[cite: 191]: "signal beam from stage 1 is expanded... match with the pump beam aperture"
        
        # --- 修改处：强制倍率为 5.0 ---
        # 原代码: mag_factor = cfg.beam_diam_p2 / cfg.beam_diam_p1
        #mag_factor = 5.0  # 论文明确提到 Stage 1 后的信号光被扩束了 5倍 (5-fold) [cite: 191]
        mag_factor = cfg.beam_diam_p2 / cfg.beam_diam_p1
        
        if mag_factor > 1.0:
            # 分别对实部和虚部进行插值放大
            # order=1 (双线性插值) 速度快且对光斑足够平滑
            Es_real = zoom(Es.real, mag_factor, order=1)
            Es_imag = zoom(Es.imag, mag_factor, order=1)
            Es_expanded = Es_real + 1j * Es_imag
            
            # --- 能量守恒校正 ---
            # 光束变宽，能量分布在更大面积上，单位面积光强(Intensity)下降
            # Intensity_new = Intensity_old / (mag^2)
            # Amplitude_new = Amplitude_old / mag
            Es_expanded /= mag_factor
            
            # --- 裁剪回原网格尺寸 (N_xy) ---
            # zoom 后数组尺寸变大 (e.g. 128 -> 272)，我们需要取中心部分的 128x128
            new_size = Es_expanded.shape[0]
            start_idx = (new_size - cfg.N_xy) // 2
            end_idx = start_idx + cfg.N_xy
            
            # 防止奇偶数导致的 1 像素误差，强制切片
            Es = Es_expanded[start_idx:end_idx, start_idx:end_idx]
            
            # 确保最终尺寸严格匹配
            if Es.shape[0] != cfg.N_xy:
                 Es = Es[:cfg.N_xy, :cfg.N_xy]

        # Reset Idler for Stage 2 (Stage 1 的 Idler 被 Filter 滤除)
        Ei = cp.zeros_like(Ei)
        
        # LOAD NEW PUMP for Stage 2 (370 mJ)
        Ep = Amp_p2 * pump_temp_profile[idx] * spatial_p2
        
        # ==========================================================
        # STAGE 2: Power Amp (BBO3 + BBO4)
        # ==========================================================

        # 4. BBO3 Propagate (Walk-off: +rho)
        Ep, Es, Ei = propagate_crystal(Ep, Es, Ei, cfg.L_BBO3, rho_rad, dk_val)
        
        # 5. 'L' Configuration: DUMP IDLER
        Ei = cp.zeros_like(Ei)
        
        # 6. BBO4 Propagate (Walk-off: -rho)
        Ep, Es, Ei = propagate_crystal(Ep, Es, Ei, cfg.L_BBO4, -rho_rad, dk_val)
        
        # ==========================================================
        # FINAL CALCULATION
        # ==========================================================
        
        power_s = cp.sum(cp.abs(Es)**2) * dx**2 * n_s0 * epsilon0 * c / 2
        slice_energy_out.append(float(power_s * dt))
        
        if idx == center_slice_idx:
            center_slice_fluence = cp.abs(Es)**2

        print(f"Slice {idx}/{cfg.num_slices} done.", end='\r')

    # ============================================
    # 结果统计与绘图
    # ============================================
    total_energy_out = sum(slice_energy_out)
    print(f"\n\nTotal Output Energy: {total_energy_out*1000:.2f} mJ")
    print(f"Sim Time: {time.time()-start_time:.2f} s")
    
    # 重建光谱 (Intensity vs Wavelength)
    # 因为采用了 Quasi-static 近似，每个时间切片对应一个特定波长
    # Slice Energy 分布其实就直接反映了光谱形状 (考虑了 chirp 映射)
    # Plot Energy(t) vs Wavelength(t)
    
    slice_energy_arr = np.array(slice_energy_out)
    # 归一化光谱
    spec_intensity = slice_energy_arr / np.max(slice_energy_arr)
    
    # 绘图
    plt.figure(figsize=(12, 5))
    
    # 1. 空间光斑 (Center Slice)
    plt.subplot(1, 2, 1)
    if center_slice_fluence is not None:
        fl_cpu = cp.asnumpy(center_slice_fluence)
        extent = [-cfg.XY_window/2*1e3, cfg.XY_window/2*1e3, -cfg.XY_window/2*1e3, cfg.XY_window/2*1e3]
        plt.imshow(fl_cpu, extent=extent, cmap='jet', origin='lower')
        plt.colorbar(label='Intensity (a.u.)')
        plt.title(f"Center Slice Beam Profile\n(t=0, {cfg.lambda_s*1e9:.1f}nm)")
        plt.xlabel("x (mm)")
        plt.ylabel("y (mm)")
    
    # 2. 光谱
    plt.subplot(1, 2, 2)
    plt.plot(lam_s_list * 1e9, spec_intensity, 'r-', linewidth=2)
    plt.title("Output Spectrum (Reconstructed from Slices)")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Normalized Intensity")
    plt.grid(True)
    plt.xlim(750, 860) # Set proper range
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_slice_simulation()
