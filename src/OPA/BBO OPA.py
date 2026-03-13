import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftshift, fftfreq

# ==========================================
# 1. 物理常数与配置
# ==========================================
c = 2.99792458e8  # 光速 (m/s)
epsilon0 = 8.854e-12
d_eff = 2.0e-12   # BBO 有效非线性系数 (approx 2 pm/V)

class SimulationConfig:
    def __init__(self):
        # 脉冲参数
        self.lambda_p = 400e-9   # Pump 波长 (m)
        self.lambda_s = 800e-9   # Signal 波长 (m)
        self.tau_p = 100e-15     # Pump 脉宽 (FWHM, s)
        self.tau_s = 50e-15      # Signal 脉宽 (FWHM, s)
        self.I_p_peak = 20e9 * 1e4 # Pump 峰值光强 20 GW/cm^2 -> W/m^2
        self.I_s_peak = 1e7 * 1e4  # Signal 种子光强 (较弱)
        
        # 晶体与网格
        self.L = 2.0e-3          # 晶体长度 2mm
        self.dz = 5e-6           # 步长 5um
        self.time_window = 2e-12 # 时间窗口 2ps
        self.N = 2048            # 采样点数

# ==========================================
# 2. BBO 晶体模型 (Sellmeier Equations)
# ==========================================
class MaterialBBO:
    def __init__(self):
        pass
    
    def get_refractive_index(self, lam, axis='o'):
        # Eimerl (1987) Sellmeier equations for BBO
        # lam in microns for the formula, convert input (meters) to microns
        lam_um = lam * 1e6
        if axis == 'o':
            n_sq = 2.7359 + 0.01878 / (lam_um**2 - 0.01822) - 0.01354 * lam_um**2
        elif axis == 'e':
            n_sq = 2.3753 + 0.01224 / (lam_um**2 - 0.01667) - 0.01516 * lam_um**2
        else:
            raise ValueError("Axis must be 'o' or 'e'")
        return np.sqrt(n_sq)

    def get_n_eff(self, lam, theta_deg):
        # 计算特定角度下的非常光折射率 n_e(theta)
        theta = np.radians(theta_deg)
        no = self.get_refractive_index(lam, 'o')
        ne = self.get_refractive_index(lam, 'e')
        # index ellipsoid equation
        return 1.0 / np.sqrt((np.cos(theta)/no)**2 + (np.sin(theta)/ne)**2)

    def get_group_velocity(self, lam, axis, theta_deg=None):
        # 数值微分计算 domega/dk = vg
        d_lam = 1e-9
        def get_k(l): 
            n = self.get_n_eff(l, theta_deg) if axis == 'eff' else self.get_refractive_index(l, axis)
            return 2 * np.pi * n / l
        
        k_center = get_k(lam)
        k_plus = get_k(lam + d_lam)
        # vg = d_omega / d_k approx - (lambda^2 / 2pi*c) * (d_omega/d_lambda) ... 
        # Easier: vg = c / (n - lambda * dn/dlambda) (Group Index)
        # Let's use finite difference on k directly:
        # vg = 1 / (dk/domega)
        omega_c = 2 * np.pi * c / lam
        omega_p = 2 * np.pi * c / (lam + d_lam)
        vg = (omega_p - omega_c) / (k_plus - k_center)
        return vg

    def get_gdd(self, lam, axis, theta_deg=None, length=1.0):
        # 计算 GDD (s^2) = L * d^2k / domega^2
        # 使用三点微分法
        d_omega = 1e12 # 1 THz perturbation
        omega_0 = 2 * np.pi * c / lam
        
        def get_k_from_omega(w):
            l = 2 * np.pi * c / w
            n = self.get_n_eff(l, theta_deg) if axis == 'eff' else self.get_refractive_index(l, axis)
            return w * n / c
            
        k0 = get_k_from_omega(omega_0)
        kp = get_k_from_omega(omega_0 + d_omega)
        km = get_k_from_omega(omega_0 - d_omega)
        
        beta2 = (kp - 2*k0 + km) / (d_omega**2)
        return beta2 * length

# ==========================================
# 3. 核心算法实现
# ==========================================

def run_simulation():
    cfg = SimulationConfig()
    bbo = MaterialBBO()
    
    # 1. 计算频率与波长
    omega_p = 2 * np.pi * c / cfg.lambda_p
    omega_s = 2 * np.pi * c / cfg.lambda_s
    omega_i = omega_p - omega_s
    lambda_i = 2 * np.pi * c / omega_i
    
    print(f"Pump: {cfg.lambda_p*1e9:.1f}nm, Signal: {cfg.lambda_s*1e9:.1f}nm, Idler: {lambda_i*1e9:.1f}nm")

    # 2. 寻找相位匹配角 (Type I: Pump(e) -> Signal(o) + Idler(o))
    # 目标: n_e(theta, wp)/lp - n_o(ws)/ls - n_o(wi)/li = 0
    thetas = np.linspace(20, 30, 1000)
    delta_ks = []
    for th in thetas:
        np_eff = bbo.get_n_eff(cfg.lambda_p, th)
        ns = bbo.get_refractive_index(cfg.lambda_s, 'o')
        ni = bbo.get_refractive_index(lambda_i, 'o')
        dk = (2*np.pi*np_eff/cfg.lambda_p) - (2*np.pi*ns/cfg.lambda_s) - (2*np.pi*ni/lambda_i)
        delta_ks.append(abs(dk))
    
    best_theta = thetas[np.argmin(delta_ks)]
    print(f"Optimal Phase Matching Angle: {best_theta:.2f} degrees")
    
    # 3. 计算参考系参数 (以 Signal 为参考系)
    # Pump (e-wave at theta)
    vg_p = bbo.get_group_velocity(cfg.lambda_p, 'eff', best_theta)
    beta2_p = bbo.get_gdd(cfg.lambda_p, 'eff', best_theta, length=1) # per meter
    
    # Signal (o-wave)
    vg_s = bbo.get_group_velocity(cfg.lambda_s, 'o')
    beta2_s = bbo.get_gdd(cfg.lambda_s, 'o', length=1)
    
    # Idler (o-wave)
    vg_i = bbo.get_group_velocity(lambda_i, 'o')
    beta2_i = bbo.get_gdd(lambda_i, 'o', length=1)
    
    # 群速度失配 (相对于 Signal)
    delta_beta1_p = 1/vg_p - 1/vg_s
    delta_beta1_i = 1/vg_i - 1/vg_s
    delta_beta1_s = 0.0 # 参考系
    
    print(f"GVM (Pump-Signal): {delta_beta1_p*1e12:.1f} ps/m")
    print(f"GVM (Idler-Signal): {delta_beta1_i*1e12:.1f} ps/m")

    # 4. 初始化场 (Time Domain)
    t = np.linspace(-cfg.time_window/2, cfg.time_window/2, cfg.N)
    dt = t[1] - t[0]
    
    # 定义高斯脉冲 (Field Amplitude, V/m)
    # E = sqrt(2 * I / (n * epsilon0 * c))
    n_p = bbo.get_n_eff(cfg.lambda_p, best_theta)
    n_s = bbo.get_refractive_index(cfg.lambda_s, 'o')
    n_i = bbo.get_refractive_index(lambda_i, 'o')
    
    A0_p = np.sqrt(2 * cfg.I_p_peak / (n_p * epsilon0 * c))
    A0_s = np.sqrt(2 * cfg.I_s_peak / (n_s * epsilon0 * c))
    
    Ep = A0_p * np.exp(-2 * np.log(2) * t**2 / cfg.tau_p**2)
    Es = A0_s * np.exp(-2 * np.log(2) * t**2 / cfg.tau_s**2)
    Ei = np.zeros_like(t, dtype=complex) # Idler starts at 0
    
    # 转换为复数数组
    Ep = Ep.astype(complex)
    Es = Es.astype(complex)
    
    # 5. 准备 SSFM 频域算子
    # frequency axis (angular frequency shift Omega)
    omega_axis = fftshift(fftfreq(cfg.N, dt)) * 2 * np.pi
    
    # 色散算子 D = exp(-i * (delta_beta1 * w + beta2/2 * w^2) * dz)
    def get_dispersion_op(db1, b2, dz):
        return np.exp(-1j * (db1 * omega_axis + 0.5 * b2 * omega_axis**2) * dz)
        
    Disp_p = get_dispersion_op(delta_beta1_p, beta2_p, cfg.dz)
    Disp_s = get_dispersion_op(delta_beta1_s, beta2_s, cfg.dz)
    Disp_i = get_dispersion_op(delta_beta1_i, beta2_i, cfg.dz)
    
    # 耦合系数
    kappa_p = 1j * omega_p * d_eff / (n_p * c)
    kappa_s = 1j * omega_s * d_eff / (n_s * c)
    kappa_i = 1j * omega_i * d_eff / (n_i * c)
    
    # 计算相位失配 delta_k (在中心频率处)
    # 注意：如果前面算对了最佳角度，这里应该是 0。保留它是为了通用性。
    dk_val = (2*np.pi*n_p/cfg.lambda_p) - (2*np.pi*n_s/cfg.lambda_s) - (2*np.pi*n_i/lambda_i)
    
    # 用于记录能量随距离变化
    z_axis = []
    en_s_axis = []

    # ==========================================
    # 主循环: Split-Step Fourier Method
    # ==========================================
    num_steps = int(cfg.L / cfg.dz)
    
    # RK4 辅助函数 (非线性耦合方程)
    def nonlinear_derivs(z_local, Ap, As, Ai):
        # 耦合波方程 (Coupled Wave Equations)
        # dAp/dz = i * kappa_p * As * Ai * exp(i * dk * z)
        # dAs/dz = i * kappa_s * Ap * conj(Ai) * exp(-i * dk * z)
        # dAi/dz = i * kappa_i * Ap * conj(As) * exp(-i * dk * z)
        
        phase_pos = np.exp(1j * dk_val * z_local)
        phase_neg = np.exp(-1j * dk_val * z_local)
        
        dAp = kappa_p * As * Ai * phase_pos
        dAs = kappa_s * Ap * np.conj(Ai) * phase_neg
        dAi = kappa_i * Ap * np.conj(As) * phase_neg
        
        return dAp, dAs, dAi

    current_z = 0
    
    for step in range(num_steps):
        # A. 线性步 (Linear Step - Frequency Domain)
        # FFT -> Apply Dispersion -> IFFT
        Ep = ifft(fftshift(Disp_p * fftshift(fft(Ep))))
        Es = ifft(fftshift(Disp_s * fftshift(fft(Es))))
        Ei = ifft(fftshift(Disp_i * fftshift(fft(Ei))))
        
        # B. 非线性步 (Nonlinear Step - Time Domain - RK4)
        h = cfg.dz
        
        # k1
        k1_p, k1_s, k1_i = nonlinear_derivs(current_z, Ep, Es, Ei)
        
        # k2
        k2_p, k2_s, k2_i = nonlinear_derivs(current_z + h/2, 
                                            Ep + h*k1_p/2, 
                                            Es + h*k1_s/2, 
                                            Ei + h*k1_i/2)
        
        # k3
        k3_p, k3_s, k3_i = nonlinear_derivs(current_z + h/2, 
                                            Ep + h*k2_p/2, 
                                            Es + h*k2_s/2, 
                                            Ei + h*k2_i/2)
        
        # k4
        k4_p, k4_s, k4_i = nonlinear_derivs(current_z + h, 
                                            Ep + h*k3_p, 
                                            Es + h*k3_s, 
                                            Ei + h*k3_i)
                                            
        Ep = Ep + (h/6) * (k1_p + 2*k2_p + 2*k3_p + k4_p)
        Es = Es + (h/6) * (k1_s + 2*k2_s + 2*k3_s + k4_s)
        Ei = Ei + (h/6) * (k1_i + 2*k2_i + 2*k3_i + k4_i)
        
        current_z += h
        
        # 记录数据
        z_axis.append(current_z * 1000) # mm
        # 简单的能量积分 (sum |A|^2)
        en_s_axis.append(np.sum(np.abs(Es)**2))

    # ==========================================
    # 4. 绘图结果
    # ==========================================
    plt.figure(figsize=(12, 8))
    
    # Plot 1: Pulse Evolution (Time)
    plt.subplot(2, 2, 1)
    plt.plot(t*1e12, np.abs(Es)**2, label='Signal Out', color='red')
    plt.plot(t*1e12, np.abs(Ei)**2, label='Idler Out', color='orange', linestyle='--')
    # 缩放初始信号以便对比
    plt.plot(t*1e12, np.abs(A0_s * np.exp(-2 * np.log(2) * t**2 / cfg.tau_s**2))**2 * 50, 
             label='Signal In (x50)', color='black', alpha=0.3)
    plt.xlabel('Time (ps)')
    plt.ylabel('Intensity (a.u.)')
    plt.title(f'Output Pulses (L={cfg.L*1000}mm)')
    plt.legend()
    plt.grid(True)
    
    # Plot 2: Spectral Broadening/Gain
    plt.subplot(2, 2, 2)
    spec_s = np.abs(fftshift(fft(Es)))**2
    freq_axis_thz = omega_axis / (2*np.pi) * 1e-12
    # Convert to wavelength roughly for x-axis labels
    wl_axis_nm = (2*np.pi*c / (omega_s + omega_axis)) * 1e9
    
    plt.plot(wl_axis_nm, spec_s / np.max(spec_s), color='red')
    plt.xlabel('Wavelength (nm)')
    plt.title('Normalized Signal Spectrum')
    plt.xlim(750, 850)
    plt.grid(True)
    
    # Plot 3: Pump Depletion (Time)
    plt.subplot(2, 2, 3)
    plt.plot(t*1e12, np.abs(Ep)**2, color='blue', label='Pump Out')
    plt.plot(t*1e12, np.abs(A0_p * np.exp(-2 * np.log(2) * t**2 / cfg.tau_p**2))**2, 
             color='cyan', linestyle='--', alpha=0.5, label='Pump In')
    plt.xlabel('Time (ps)')
    plt.title('Pump Depletion')
    plt.legend()
    plt.grid(True)
    
    # Plot 4: Gain Curve vs Z
    plt.subplot(2, 2, 4)
    plt.plot(z_axis, en_s_axis, color='green')
    plt.xlabel('Crystal Length (mm)')
    plt.ylabel('Signal Energy (a.u.)')
    plt.title('Amplification Process')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    run_simulation()