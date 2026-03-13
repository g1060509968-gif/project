import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftshift, fftfreq, ifftshift
import time

# ==========================================
# 1. 物理常数与配置
# ==========================================
c = 2.99792458e8
epsilon0 = 8.854e-12
d_eff = 2.0e-12 

class PaperConfig:
    def __init__(self):
        self.lambda_p = 532e-9   
        self.lambda_s = 808e-9   
        
        self.tau_p = 2.2e-9      
        self.energy_p = 370e-3   
        self.beam_diam_p = 8.5e-3 
        
        self.tau_s_stretched = 2.55e-9 
        self.bw_s_nm = 100.0     
        self.energy_s_in = 5e-3  
        self.beam_diam_s = 8.5e-3 

        self.alpha_deg = 2.37    # 非共线角
        self.L_BBO3 = 16.0e-3    
        self.L_BBO4 = 9.0e-3     
        
        # 保持高采样率以解析宽带光谱
        self.time_window = 6e-9  
        self.N = 2**19           # ~520,000 点
        self.dz = 50e-6          # 50um 步长

# ==========================================
# 2. BBO 晶体模型
# ==========================================
class MaterialBBO:
    def __init__(self):
        pass
    
    def get_refractive_index(self, lam, axis='o'):
        lam_um = lam * 1e6
        if axis == 'o':
            n_sq = 2.7359 + 0.01878 / (lam_um**2 - 0.01822) - 0.01354 * lam_um**2
        elif axis == 'e':
            n_sq = 2.3753 + 0.01224 / (lam_um**2 - 0.01667) - 0.01516 * lam_um**2
        else:
            raise ValueError("Axis must be 'o' or 'e'")
        return np.sqrt(n_sq)

    def get_n_eff(self, lam, theta_rad):
        no = self.get_refractive_index(lam, 'o')
        ne = self.get_refractive_index(lam, 'e')
        return 1.0 / np.sqrt((np.cos(theta_rad)/no)**2 + (np.sin(theta_rad)/ne)**2)

# ==========================================
# 3. 求解器 (Simplified for ns pulses)
# ==========================================
def run_simulation():
    start_time = time.time()
    cfg = PaperConfig()
    bbo = MaterialBBO()
    
    # --- A. 基础计算 ---
    omega_p = 2 * np.pi * c / cfg.lambda_p
    omega_s0 = 2 * np.pi * c / cfg.lambda_s
    omega_i0 = omega_p - omega_s0
    lambda_i0 = 2 * np.pi * c / omega_i0
    
    print(f"--- Simulation Start (Pure RK4) ---")
    
    # 计算相位匹配角 (使用简单的矢量近似)
    alpha_rad = np.radians(cfg.alpha_deg)
    
    def get_k_vector(lam, theta, axis):
        n = bbo.get_n_eff(lam, theta) if axis=='e' else bbo.get_refractive_index(lam, axis)
        return 2 * np.pi * n / lam

    # 扫描寻找最佳 theta
    thetas = np.linspace(23.0, 24.5, 200) # 增加精度
    dks = []
    for th_deg in thetas:
        th = np.radians(th_deg)
        kp = get_k_vector(cfg.lambda_p, th, 'e')
        ks = get_k_vector(cfg.lambda_s, 0, 'o') 
        ki = get_k_vector(lambda_i0, 0, 'o')
        # dk = kp*cos(alpha) - ks - sqrt(ki^2 - (kp*sin(alpha))^2)
        dk = kp * np.cos(alpha_rad) - ks - np.sqrt(ki**2 - (kp * np.sin(alpha_rad))**2)
        dks.append(abs(dk))
    
    best_theta_deg = thetas[np.argmin(dks)]
    best_theta_rad = np.radians(best_theta_deg)
    # 重新计算最佳点的 dk 值用于仿真
    kp_best = get_k_vector(cfg.lambda_p, best_theta_rad, 'e')
    ks_best = get_k_vector(cfg.lambda_s, 0, 'o')
    ki_best = get_k_vector(lambda_i0, 0, 'o')
    dk_val = kp_best * np.cos(alpha_rad) - ks_best - np.sqrt(ki_best**2 - (kp_best * np.sin(alpha_rad))**2)

    print(f"Optimal Theta: {best_theta_deg:.3f} deg")
    print(f"Residual dk: {dk_val:.2f} m^-1")

    # --- B. 脉冲初始化 ---
    t = np.linspace(-cfg.time_window/2, cfg.time_window/2, cfg.N)
    dt = t[1] - t[0]
    freqs = fftshift(fftfreq(cfg.N, dt))
    omega_axis = 2 * np.pi * freqs
    
    # 1. Pump (Time Domain)
    area_p = np.pi * (cfg.beam_diam_p/2)**2
    I0_p = cfg.energy_p / (area_p * cfg.tau_p)
    Ep_profile = np.exp(- (t / (cfg.tau_p/2))**8 ) # Flat top
    n_p = bbo.get_n_eff(cfg.lambda_p, best_theta_rad)
    A0_p = np.sqrt(2 * I0_p / (n_p * epsilon0 * c))
    Ep = A0_p * Ep_profile.astype(complex)

    # 2. Signal (Stretched)
    dw_fwhm = 2 * np.pi * c * (cfg.bw_s_nm * 1e-9) / (cfg.lambda_s**2)
    spec_env = np.exp(-4 * np.log(2) * omega_axis**2 / dw_fwhm**2)
    GDD = cfg.tau_s_stretched / dw_fwhm
    print(f"Applying GDD: {GDD:.2e} s^2")
    
    Es_freq = spec_env * np.exp(1j * 0.5 * GDD * omega_axis**2)
    Es = fftshift(ifft(ifftshift(Es_freq)))
    
    # Normalize Energy
    area_s = np.pi * (cfg.beam_diam_s/2)**2
    n_s = bbo.get_refractive_index(cfg.lambda_s, 'o')
    curr_en = np.sum(np.abs(Es)**2) * dt * area_s * n_s * epsilon0 * c / 2
    Es = Es * np.sqrt(cfg.energy_s_in / curr_en)
    
    Ei = np.zeros_like(Es)

    # --- C. 耦合系数 ---
    # 忽略色散算子，只保留耦合系数和相位失配
    n_i = bbo.get_refractive_index(lambda_i0, 'o')
    kappa_s = 1j * omega_s0 * d_eff / (n_s * c)
    kappa_i = 1j * omega_i0 * d_eff / (n_i * c)
    kappa_p = 1j * omega_p * d_eff / (n_p * c)

    # --- D. 仿真循环 (纯 RK4) ---
    def run_crystal(E_p, E_s, E_i, length):
        steps = int(length / cfg.dz)
        z_record = []
        e_record = []
        
        local_z = 0
        
        # 预计算相位因子数组 (如果 dz 固定)
        # 但 RK4 中 z 会变动 (z, z+h/2, z+h)，所以动态计算
        
        current_Ep = E_p.copy()
        current_Es = E_s.copy()
        current_Ei = E_i.copy()
        
        for i in range(steps):
            # 这里的 z 必须是累积的物理距离，用于计算 exp(i*dk*z)
            # RK4 Solver
            h = cfg.dz
            
            def derivatives(z_pos, Ap, As, Ai):
                # Standard Coupled Wave Equations with Phase Mismatch
                # dAs/dz = i * kappa * Ap * Ai* * exp(i * dk * z)
                phase = np.exp(1j * dk_val * z_pos)
                phase_c = np.conj(phase)
                
                dAp = kappa_p * As * Ai * phase_c  # exp(-i dk z)
                dAs = kappa_s * Ap * np.conj(Ai) * phase  # exp(i dk z)
                dAi = kappa_i * Ap * np.conj(As) * phase  # exp(i dk z)
                return dAp, dAs, dAi

            # K1
            k1p, k1s, k1i = derivatives(local_z, current_Ep, current_Es, current_Ei)
            
            # K2
            k2p, k2s, k2i = derivatives(local_z + h/2, 
                                        current_Ep + h*k1p/2, 
                                        current_Es + h*k1s/2, 
                                        current_Ei + h*k1i/2)
            
            # K3
            k3p, k3s, k3i = derivatives(local_z + h/2, 
                                        current_Ep + h*k2p/2, 
                                        current_Es + h*k2s/2, 
                                        current_Ei + h*k2i/2)
                                        
            # K4
            k4p, k4s, k4i = derivatives(local_z + h, 
                                        current_Ep + h*k3p, 
                                        current_Es + h*k3s, 
                                        current_Ei + h*k3i)
            
            current_Ep += (h/6)*(k1p + 2*k2p + 2*k3p + k4p)
            current_Es += (h/6)*(k1s + 2*k2s + 2*k3s + k4s)
            current_Ei += (h/6)*(k1i + 2*k2i + 2*k3i + k4i)
            
            local_z += h
            z_record.append(local_z)
            
            # Energy check
            en = np.sum(np.abs(current_Es)**2) * dt * area_s * n_s * epsilon0 * c / 2
            e_record.append(en)
            
            if i % 50 == 0:
                print(f"Step {i}/{steps}, Signal Energy: {en*1000:.1f} mJ", end='\r')

        return current_Ep, current_Es, current_Ei, z_record, e_record

    # --- 执行 ---
    print("\nRunning Crystal 1 (16mm)...")
    Ep, Es, Ei, z1, en1 = run_crystal(Ep, Es, Ei, cfg.L_BBO3)
    
    print("\nDumping Idler...")
    Ei = np.zeros_like(Ei)
    
    print("\nRunning Crystal 2 (9mm)...")
    # 重要: 重置 z=0。在 L 型构型中，第二块晶体通常会重新对准相位(adjust air gap or crystal angle)
    # 以确保 Idler 从零开始时与 Pump/Signal 再次匹配。
    # 如果让 z 继续增加，exp(i dk z) 可能会从错误的相位开始。
    # 假设重新相位匹配 -> z 从 0 开始计算相对相位。
    Ep, Es, Ei, z2, en2 = run_crystal(Ep, Es, Ei, cfg.L_BBO4)
    
    total_en = en1 + en2
    total_z = np.concatenate([np.array(z1), np.array(z2) + cfg.L_BBO3])
    
    print(f"\nFinal Energy: {total_en[-1]*1000:.2f} mJ")
    print(f"Efficiency: {total_en[-1]/cfg.energy_p*100:.1f}%")
    print(f"Sim Time: {time.time()-start_time:.1f}s")
    
    # --- 绘图 ---
    plt.figure(figsize=(15, 10))
    
    # Spectrum
    plt.subplot(2,2,1)
    spec_out = np.abs(fftshift(fft(Es)))**2
    wl_axis = 2*np.pi*c / (omega_s0 + omega_axis) * 1e9
    mask = (wl_axis > 750) & (wl_axis < 860)
    plt.plot(wl_axis[mask], spec_out[mask], 'r')
    plt.title("Output Spectrum")
    plt.xlabel("Wavelength (nm)")
    plt.grid(True)
    
    # Time
    plt.subplot(2,2,2)
    # 计算光强 (W/m^2)
    I_s = np.abs(Es)**2 * n_s * epsilon0 * c / 2
    I_p = np.abs(Ep)**2 * n_p * epsilon0 * c / 2
    plt.plot(t*1e9, I_s/1e4/1e9, 'r', label='Signal (GW/cm^2)')
    plt.plot(t*1e9, I_p/1e4/1e9, 'b', label='Pump (GW/cm^2)')
    plt.title("Time Domain")
    plt.xlabel("Time (ns)")
    plt.ylabel("Intensity (GW/cm^2)")
    plt.legend()
    plt.grid(True)
    
    # Energy
    plt.subplot(2,2,3)
    plt.plot(total_z*1000, np.array(total_en)*1000, 'g', lw=2)
    plt.axvline(cfg.L_BBO3*1000, color='k', ls='--')
    plt.title("Energy Growth")
    plt.ylabel("Energy (mJ)")
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_simulation()