"""
ERK4(3)-IP Full Dispersion Solver (Enhanced Input Version)
基于论文: Balac & Mahé, "Embedded Runge-Kutta scheme for step-size control in the interaction picture method", CPC (2013)
包含:
1. Sellmeier 全波长色散
2. 高阶非线性 (自陡峭 + Raman)
3. 高级光束生成 (高斯脉冲/光谱重建)
"""

import numpy as np
from scipy.fft import fft, ifft, fftfreq, fftshift, ifftshift
from scipy.interpolate import interp1d

class ERK43IP_FullDispersion:
    """
    ERK4(3)-IP 求解器
    支持全波长色散、自陡峭、拉曼效应，以及通过物理参数生成输入光束。
    """
    
    def __init__(self, material='fused_silica', 
                 gamma=None, n2=None, beam_radius=None, 
                 alpha=0.0, center_wavelength=1064e-9, 
                 use_raman=True, f_R=0.18, tau1=12.2e-15, tau2=32e-15,
                 use_self_steepening=True):
        """
        初始化仿真器
        
        Parameters:
        -----------
        material : str
            材料名称 ('fused_silica', 'sapphire', 'yag')
        gamma : float, optional
            非线性系数 (W^-1 m^-1)。如果提供，优先使用此值。
        n2 : float, optional
            非线性折射率 (m^2/W)。需配合 beam_radius 使用。
        beam_radius : float, optional
            光斑半径 (m)。需配合 n2 使用。
        alpha : float
            损耗系数 (m^-1)
        center_wavelength : float
            中心波长 lambda0 (m)
        use_raman : bool
            是否开启拉曼效应
        f_R : float
            拉曼分数
        tau1, tau2 : float
            拉曼响应时间常数 (s)
        use_self_steepening : bool
            是否开启自陡峭效应
        """
        self.material = material
        self.alpha = alpha
        self.lambda0 = center_wavelength
        
        # 物理常数
        self.c = 299792458.0  # m/s
        self.omega0 = 2 * np.pi * self.c / self.lambda0
        
        # --- 自动计算非线性系数 gamma ---
        if gamma is not None:
            self.gamma = gamma
        elif n2 is not None and beam_radius is not None:
            # 有效模场面积 A_eff = pi * w^2
            self.beam_area = np.pi * beam_radius**2
            # gamma = 2*pi*n2 / (lambda * A_eff)
            self.gamma = (2 * np.pi / self.lambda0) * (n2 / self.beam_area)
            print(f"初始化: 根据参数 n2={n2:.2e}, r={beam_radius*1e6:.1f}um 计算 gamma={self.gamma:.6f} W^-1 m^-1")
        else:
            self.gamma = 0.0
            print("警告: 未提供 gamma 或 (n2, beam_radius)，非线性系数默认为 0")
        
        # 高阶非线性开关与参数
        self.use_raman = use_raman
        self.f_R = f_R
        self.tau1 = tau1
        self.tau2 = tau2
        self.use_self_steepening = use_self_steepening
        
        # 获取 Sellmeier 系数
        self.sellmeier_coeffs = self._get_sellmeier_coeffs(material)
        
        # 缓存变量
        self._cached_D = None
        self._cached_omega_grid = None
        self._cached_raman_response = None 

    def _get_sellmeier_coeffs(self, material):
        """定义常用材料的 Sellmeier 系数"""
        coeffs = {
            'fused_silica': {
                'B': [0.6961663, 0.4079426, 0.8974794],
                'C': [0.0684043**2, 0.1162414**2, 9.896161**2], # C单位是 um^2
                'valid_range': (0.21e-6, 6.7e-6)
            },
            'sapphire': {
                'B': [1.4313493, 0.65054713, 5.3414021],
                'C': [0.0726631**2, 0.1193242**2, 18.028251**2],
                'valid_range': (0.2e-6, 5.5e-6)
            },
            'yag': {
                'B': [2.282, 3.27644, 4.44174],
                'C': [0.01185**2, 0.0282**2, 282.734**2],
                'valid_range': (0.4e-6, 5e-6)
            }
        }
        return coeffs.get(material, coeffs['fused_silica'])

    def _compute_refractive_index(self, wavelength):
        """使用 Sellmeier 方程计算折射率 n(lambda)"""
        coeffs = self.sellmeier_coeffs
        B, C = coeffs['B'], coeffs['C']
        wl_um = wavelength * 1e6
        wl_sq = wl_um**2
        n_squared = np.ones_like(wavelength, dtype=float)
        for i in range(3):
            n_squared += (B[i] * wl_sq) / (wl_sq - C[i])
        return np.sqrt(n_squared)

    def _compute_dispersion_operator(self, omega_grid):
        """计算全波长色散算子 D(omega)"""
        omega_abs = self.omega0 + omega_grid
        # 避免除以零
        omega_safe = np.where(np.abs(omega_abs) < 1e-10, 1e-10, omega_abs)
        wavelengths = 2 * np.pi * self.c / np.abs(omega_safe)
        
        valid_range = self.sellmeier_coeffs['valid_range']
        wavelengths = np.clip(wavelengths, valid_range[0], valid_range[1])
        
        n_omega = self._compute_refractive_index(wavelengths)
        k_omega = n_omega * omega_abs / self.c
        
        # 计算 k0 和 k1 (群速度倒数)
        d_omega = self.omega0 * 1e-5
        w_center = np.array([self.lambda0])
        w_plus = 2 * np.pi * self.c / (self.omega0 + d_omega)
        w_minus = 2 * np.pi * self.c / (self.omega0 - d_omega)
        
        n0 = self._compute_refractive_index(w_center)[0]
        n_plus = self._compute_refractive_index(np.array([w_plus]))[0]
        n_minus = self._compute_refractive_index(np.array([w_minus]))[0]
        
        k0 = n0 * self.omega0 / self.c
        k_plus = n_plus * (self.omega0 + d_omega) / self.c
        k_minus = n_minus * (self.omega0 - d_omega) / self.c
        
        k1 = (k_plus - k_minus) / (2 * d_omega)
        
        D = 1j * (k_omega - k0 - k1 * omega_grid) - self.alpha / 2
        return D

    def _update_raman_response(self, omega_grid):
        """计算频域拉曼响应函数 H_R(omega)"""
        N = len(omega_grid)
        if N > 1:
            d_omega = omega_grid[1] - omega_grid[0]
            dt = 2 * np.pi / (N * np.abs(d_omega))
        else:
            dt = 1e-15
        dt = max(dt, 1e-15)
        t = np.arange(N) * dt
        
        tau1_safe = max(self.tau1, 1e-20)
        tau2_safe = max(self.tau2, 1e-20)
        
        h_t = (tau1_safe**2 + tau2_safe**2) / (tau1_safe * tau2_safe**2) * \
              np.exp(-t / tau2_safe) * np.sin(t / tau1_safe)
        
        h_t[0] = 0.0
        H_omega = fft(h_t) * dt
        
        if np.abs(H_omega[0]) > 1e-15:
            H_omega = H_omega / H_omega[0]
        else:
            H_omega = np.ones_like(H_omega, dtype=complex)
        
        self._cached_raman_response = H_omega

    def linear_operator(self, A, omega, h):
        """应用线性算子 exp(h*D)"""
        if self._cached_D is None or \
           len(omega) != len(self._cached_omega_grid) or \
           not np.allclose(omega, self._cached_omega_grid):
            
            self._cached_omega_grid = omega.copy()
            self._cached_D = self._compute_dispersion_operator(omega)
            
            if self.use_raman:
                self._update_raman_response(omega)
            
        A_freq = fft(A)
        A_freq_new = A_freq * np.exp(h * self._cached_D)
        return ifft(A_freq_new)

    def nonlinear_operator(self, A, omega=None):
        """高阶非线性算子 N(A)"""
        Intensity = np.abs(A)**2
        
        if self.use_raman:
            if self._cached_raman_response is None and omega is not None:
                 self._update_raman_response(omega)
            
            if self._cached_raman_response is not None:
                I_omega = fft(Intensity)
                I_delayed = ifft(I_omega * self._cached_raman_response)
                I_eff = (1 - self.f_R) * Intensity + self.f_R * I_delayed
            else:
                I_eff = Intensity
        else:
            I_eff = Intensity
            
        P_NL = A * I_eff
        
        if self.use_self_steepening:
            if omega is None: omega = self._cached_omega_grid
            omega_abs = self.omega0 + omega
            shock_term = omega_abs / self.omega0
            P_NL_freq = fft(P_NL)
            nonlinear_term = 1j * self.gamma * ifft(shock_term * P_NL_freq)
        else:
            nonlinear_term = 1j * self.gamma * P_NL
            
        return nonlinear_term

    def erk43_step(self, A, omega, h, compute_error=True, alpha5_prev=None, h_prev=None):
        """
        执行一步 ERK4(3) 方法 
        利用 FSAL (First Same As Last) 性质 [cite: 282-283]
        """
        v_ip = self.linear_operator(A, omega, h/2)
        
        if alpha5_prev is not None and h_prev is not None and abs(h - h_prev) < 1e-10:
            alpha1 = self.linear_operator(alpha5_prev, omega, h/2)
        else:
            alpha1 = self.linear_operator(self.nonlinear_operator(A, omega), omega, h/2)
        
        alpha2 = self.nonlinear_operator(v_ip + h/2 * alpha1, omega)
        alpha3 = self.nonlinear_operator(v_ip + h/2 * alpha2, omega)
        
        temp = self.linear_operator(v_ip + h * alpha3, omega, h/2)
        alpha4_prime = self.nonlinear_operator(temp, omega)
        
        beta = self.linear_operator(
            v_ip + h/6 * (alpha1 + 2*alpha2 + 2*alpha3), 
            omega, h/2
        )
        
        A4 = beta + h/6 * alpha4_prime
        alpha5_prime = self.nonlinear_operator(A4, omega)
        
        if compute_error:
            # 3阶嵌入解用于误差估计 [cite: 321]
            A3 = beta + (h/30) * (2*alpha4_prime + 3*alpha5_prime)
            norm_A = np.sqrt(np.sum(np.abs(A4)**2))
            if norm_A < 1e-20: norm_A = 1.0
            error = np.sqrt(np.sum(np.abs(A4 - A3)**2)) / norm_A
            return A4, error, alpha5_prime
        else:
            return A4, 0.0, alpha5_prime

    def propagate(self, A0, t, L, tol=1e-4, max_step=1e-2, min_step=1e-11):
        """自适应步长传播主循环"""
        N = len(t)
        dt = t[1] - t[0]
        omega = 2 * np.pi * fftfreq(N, dt)
        
        z = 0.0
        h = max_step / 100
        A = A0.copy()
        
        z_history = [0.0]
        A_history = [A0.copy()]
        
        error_prev = None
        alpha5_prev = None
        h_prev = None
        
        print(f"开始传播: L={L:.4f}m, 初始步长={h:.2e}m")
        
        while z < L:
            if z + h > L:
                h = L - z
            
            A_new, error, alpha5_new = self.erk43_step(
                A, omega, h, 
                compute_error=True, 
                alpha5_prev=alpha5_prev, 
                h_prev=h_prev
            )
            
            # 步长控制 [cite: 358-359]
            if error < tol:
                z += h
                A = A_new
                alpha5_prev = alpha5_new
                h_prev = h
                
                if len(z_history) < 1000 or (z - z_history[-1]) > L/200:
                    z_history.append(z)
                    A_history.append(A.copy())
                
                factor = 0.95 * (tol / (error + 1e-30))**0.25
                if error_prev is not None and error_prev > 1e-30:
                    factor *= (error_prev / (error + 1e-30))**0.1
                error_prev = error
                h_new = h * min(factor, 2.0)
                h = min(h_new, max_step)
            else:
                factor = 0.95 * (tol / (error + 1e-30))**0.25
                h_new = h * max(factor, 0.1)
                h = max(h_new, min_step)
                
                if h <= min_step:
                    print(f"Warning: Step size collapsed to {h:.2e} at z={z:.4f}")
                    A_new, error, alpha5_new = self.erk43_step(
                        A, omega, h, 
                        compute_error=True, 
                        alpha5_prev=alpha5_prev, 
                        h_prev=h_prev
                    )
                    z += h
                    A = A_new
                    alpha5_prev = alpha5_new
                    h_prev = h
                    error_prev = None
        
        return np.array(z_history), np.array(A_history), omega

    # =================================================================
    # 新增功能: 脉冲生成 (高斯脉冲 & 光谱重建)
    # =================================================================
    
    def generate_gaussian_pulse(self, t, pulse_energy, pulse_fwhm, chirp=0):
        """
        生成高斯脉冲
        
        Parameters:
        -----------
        t : array
            时间网格 (s)
        pulse_energy : float
            脉冲能量 (J)
        pulse_fwhm : float
            脉冲宽度 FWHM (s)
        chirp : float
            啁啾参数 C
            
        Returns:
        --------
        A : array
            复数场幅 sqrt(W)
        """
        # 峰值功率 P0 = E / (T_FWHM * 1.064)
        factor = np.sqrt(np.pi / (4 * np.log(2))) 
        P0 = pulse_energy / (pulse_fwhm * factor)
        
        # 1/e 宽度 T0
        T0 = pulse_fwhm / (2 * np.sqrt(np.log(2)))
        
        # 场幅
        A = np.sqrt(P0) * np.exp(-(1 + 1j * chirp) * (t / T0)**2 / 2)
        
        print(f"生成高斯脉冲: E={pulse_energy*1e9:.2f}nJ, FWHM={pulse_fwhm*1e12:.2f}ps, P0={P0/1e3:.2f}kW")
        return A

    def generate_dispersed_pulse(self, t, pulse_energy, pulse_fwhm, GDD=0, TOD=0):
        """
        生成带有二阶(GDD)和三阶(TOD)色散的高斯脉冲
        (已修正: 增加 fftshift/ifftshift 以消除相位噪声)
        """
        # 1. 计算变换受限脉冲的参数
        factor = np.sqrt(np.pi / (4 * np.log(2)))
        P0 = pulse_energy / (pulse_fwhm * factor)
        T0 = pulse_fwhm / (2 * np.sqrt(np.log(2)))
        
        # 2. 生成初始时域场 (位于数组中心)
        A_tl = np.sqrt(P0) * np.exp(-(t/T0)**2 / 2)
        
        # 如果没有色散，直接返回
        if GDD == 0 and TOD == 0:
            print(f"生成变换受限脉冲: E={pulse_energy*1e9:.2f}nJ, FWHM={pulse_fwhm*1e12:.2f}ps")
            return A_tl
            
        # 3. 准备频域参数 (标准 FFT 顺序: 0, 1, ... -1)
        N = len(t)
        dt = t[1] - t[0]
        omega = 2 * np.pi * fftfreq(N, dt)
        
        # ==================== 修正开始 ====================
        
        # 4. 转换到频域
        # [关键步骤]: 使用 ifftshift 将脉冲中心从数组中间移到 index 0
        # 这样 fft 后的频谱是平滑的，没有快速振荡的线性相位
        A_shifted = fftshift(A_tl)  # 注意：这里对于中心对齐的输入，fftshift/ifftshift 通常互换使用，通常用 ifftshift 移到零点
        # 但由于通常 t 是 linspace(-T, T)，为了保险，我们用 ifftshift
        A_time_at_zero = ifftshift(A_tl)
        
        A_freq = fft(A_time_at_zero)
        
        # 5. 应用色散相位
        # 谱相位 Φ(ω) = (1/2)*GDD*ω^2 + (1/6)*TOD*ω^3
        dispersion_phase = 0.5 * GDD * omega**2 + (1.0/6.0) * TOD * omega**3
        A_dispersed_freq = A_freq * np.exp(1j * dispersion_phase)
        
        # 6. 转换回时域
        A_temp = ifft(A_dispersed_freq)
        
        # [关键步骤]: 恢复脉冲位置，从 index 0 移回数组中心
        A_final = fftshift(A_temp)
        
        # ==================== 修正结束 ====================
        
        # 7. 计算展宽后的参数用于显示
        power = np.abs(A_final)**2
        max_p = np.max(power)
        indices = np.where(power >= max_p/2)[0]
        if len(indices) > 1:
            current_fwhm = t[indices[-1]] - t[indices[0]]
        else:
            current_fwhm = 0
            
        stretch_ratio = current_fwhm / pulse_fwhm if pulse_fwhm > 0 else 0
        
        print(f"生成色散脉冲 (修正版):")
        print(f"  GDD = {GDD*1e30:.0f} fs², TOD = {TOD*1e45:.0f} fs³")
        print(f"  原始宽度 (TL): {pulse_fwhm*1e12:.3f} ps")
        print(f"  展宽后宽度:    {current_fwhm*1e12:.3f} ps (展宽比: {stretch_ratio:.2f})")
        
        return A_final

    def generate_pulse_from_spectrum(self, t, wavelengths_nm, intensities, pulse_energy, 
                                     GDD=0, TOD=0):  # <--- [改动1] 增加了 GDD 和 TOD 参数
        """
        从外部光谱数据重建脉冲，并允许添加额外的色散相位。
        
        Parameters:
        -----------
        ... (其他参数不变)
        GDD : float
            群延迟色散 (s^2), 默认为 0 (无啁啾)
        TOD : float
            三阶色散 (s^3), 默认为 0
        """
        print("正在从光谱数据重建脉冲 (含相位补偿)...")
        
        N = len(t)
        dt = t[1] - t[0]
        
        # --- 准备两套频率网格 ---
        
        # 1. 相对角频率 Omega (用于计算相位)
        # fftfreq 生成的就是相对频率 (0, df, 2df ... -df)
        omega_rel = 2 * np.pi * fftfreq(N, dt)
        
        # 2. 绝对频率 f_abs (用于插值光谱)
        # 绝对频率 = 相对频率 + 中心频率
        freqs_sim_abs = (omega_rel / (2*np.pi)) + (self.c / self.lambda0)
        
        # --- [步骤 A] 处理幅值 (原逻辑) ---
        
        # 整理输入数据
        idx = np.argsort(wavelengths_nm)
        wl_sorted = wavelengths_nm[idx]
        int_sorted = intensities[idx]
        
        # 转换为频率 (Hz)
        freqs_input = self.c / (wl_sorted * 1e-9)
        
        # 插值得到功率谱密度 S(f)
        f_interp = interp1d(freqs_input[::-1], int_sorted[::-1], 
                            kind='linear', bounds_error=False, fill_value=0.0)
        
        spec_density = f_interp(freqs_sim_abs) # 注意这里用绝对频率查表
        spec_density[spec_density < 0] = 0 
        
        # 得到幅值 |A(w)|
        A_freq_mag = np.sqrt(spec_density)
        
        # --- [步骤 B] 注入相位 (新逻辑) ---
        
        if GDD == 0 and TOD == 0:
            # 如果没有色散，相位因子就是 1
            phase_factor = 1.0
        else:
            # 计算色散相位 phi(w)
            # 注意：必须使用【相对角频率】omega_rel 来计算
            dispersion_phase = 0.5 * GDD * omega_rel**2 + (1.0/6.0) * TOD * omega_rel**3
            phase_factor = np.exp(1j * dispersion_phase)
            print(f"  应用相位: GDD={GDD*1e30:.0f}fs², TOD={TOD*1e45:.0f}fs³")
            
        # 合成复数频域场： 幅值 * 相位
        A_freq_complex = A_freq_mag * phase_factor
        
        # --- [步骤 C] 回到时域 (原逻辑) ---
        
        # IFFT
        A_temp = ifft(A_freq_complex)
        
        # Shift 到中心
        A_initial = fftshift(A_temp)
        
        # 能量归一化
        current_E = np.trapz(np.abs(A_initial)**2, t)
        
        if current_E > 0:
            scale = np.sqrt(pulse_energy / current_E)
            A_initial *= scale
            print(f"重建完成: 目标能量={pulse_energy*1e9:.2f}nJ")
        else:
            raise ValueError("光谱重建失败：计算能量为0")
            
        return A_initial
    
    # =================================================================
    # 新增功能: 相位残差计算 (Spectral Phase Analysis)
    # =================================================================

    def analyze_spectral_phase(self, A, t, fit_order=4, plot_threshold=0.01):
        """
        相位残差计算 (线性归一化版本)
        
        Parameters:
        -----------
        plot_threshold : float
            线性阈值 (0.0~1.0)，例如 0.01 表示只计算强度大于峰值 1% 的区域
        """

        N = len(t)
        dt = t[1] - t[0]
        
        # 1. 频域转换 (包络坐标系)
        # fftfreq 生成的直接就是相对中心频率的偏差 (delta_omega / 2pi)
        freq = fftfreq(N, dt)
        omega_rel = fftshift(2 * np.pi * freq) # 相对角频率
        
        # 频谱计算
        spectrum = fftshift(fft(A))
        spectral_intensity = np.abs(spectrum)**2
        
        # 归一化 (Normalize)
        max_val = np.max(spectral_intensity)
        if max_val > 0:
            spectral_intensity_norm = spectral_intensity / max_val
        else:
            spectral_intensity_norm = spectral_intensity

        # 2. 提取相位
        phi = np.unwrap(np.angle(spectrum))
        
        # 3. 确定有效范围 (Mask) - 使用线性阈值
        mask = spectral_intensity_norm > plot_threshold
        
        # 安全检查
        if np.sum(mask) < 5:
            print("有效点太少，跳过相位分析")
            return None

        # 提取有效数据用于拟合
        w_fit = omega_rel[mask]
        phi_fit = phi[mask]
        
        # 4. 多项式拟合
        try:
            p = np.polyfit(w_fit, phi_fit, fit_order)
        except Exception as e:
            print(f"相位拟合失败: {e}")
            return None
        
        # 计算全域拟合曲线 (用于减去)
        phi_polynomial = np.polyval(p, omega_rel)
        
        # 5. 计算残差
        phi_residual = phi - phi_polynomial
        
        # 对齐残差中心到 0 (以最大强度点为基准，美观用)
        center_index = np.argmax(spectral_intensity_norm)
        phi_residual = phi_residual - phi_residual[center_index]

        # 6. 转换波长轴
        # 绝对角频率 = 相对角频率 + 中心角频率
        omega_abs = omega_rel + self.omega0
        # 防止除零
        omega_abs[omega_abs < 1e-10] = 1e-10 
        wavelengths_nm = 2 * np.pi * self.c / omega_abs * 1e9
        
        # 简单的 GDD/TOD 估算输出
        if fit_order >= 2:
            gdd = 2 * p[-3] * 1e30 if fit_order >= 2 else 0
            tod = 6 * p[-4] * 1e45 if fit_order >= 3 else 0
            print(f"[相位分析] 估算 GDD: {gdd:.1f} fs², TOD: {tod:.1f} fs³")

        return {
            "wavelength_nm": wavelengths_nm,
            "spectral_intensity": spectral_intensity_norm, # 线性归一化强度
            "residual_phase": phi_residual,
            "mask": mask
        }

    def get_beta2(self):
        """计算中心波长处的 GVD (beta2) 用于显示"""
        # 利用中心差分估算 d^2k/dw^2
        w0 = self.omega0
        dw = w0 * 0.001
        
        # 计算三个点的波数 k = n*w/c
        def get_k(w):
            wl = 2 * np.pi * self.c / w
            n = self._compute_refractive_index(wl) # 假设这是内部方法，下面会修正
            return n * w / self.c
            
        # 注意：这里需要临时调用类内部的折射率计算逻辑
        # 为了方便，我们直接利用 self.sellmeier_coeffs
        coeffs = self.sellmeier_coeffs
        def n_calc(wl_m):
            wl_um = wl_m * 1e6
            B, C = coeffs['B'], coeffs['C']
            n_sq = 1.0
            for i in range(3):
                n_sq += (B[i] * wl_um**2) / (wl_um**2 - C[i])
            return np.sqrt(n_sq)
            
        k0 = n_calc(2*np.pi*self.c/w0) * w0 / self.c
        kp = n_calc(2*np.pi*self.c/(w0+dw)) * (w0+dw) / self.c
        km = n_calc(2*np.pi*self.c/(w0-dw)) * (w0-dw) / self.c
        
        beta2 = (kp - 2*k0 + km) / (dw**2)
        return beta2

class SimResultAdapter:
    """
    适配器类：将 ERK43IP 的仿真结果包装成 visualization 模块能识别的对象
    """
    def __init__(self, solver, z_array, t, A_evolution):
        # 1. 传递物理参数
        self.material = solver.material
        self.lambda0 = solver.lambda0
        self.omega0 = solver.omega0
        self.c = solver.c
        self.beam_radius = getattr(solver, 'beam_radius', 0) # 处理可能不存在的情况
        self.n2 = getattr(solver, 'n2', 0)
        self.gamma = solver.gamma
        
        # 2. 传递网格参数
        self.t = t
        self.Nt = len(t)
        self.dt = t[1] - t[0]
        self.T = self.Nt * self.dt
        
        # 3. 传递传播参数
        self.z = z_array
        self.L = z_array[-1]
        self.Nz = len(z_array)
        self.dz = z_array[1] - z_array[0] if len(z_array) > 1 else 0
        
        # 4. 计算/传递衍生参数
        try:
            # 尝试调用 solver 的方法计算 beta2
            self.beta2 = solver.get_beta2()
        except:
            # 如果 solver 没有该方法（旧版本），设为 0
            self.beta2 = 0.0
            
        self.A_evolution = A_evolution
    
    # 为了兼容 visualization 的某些内部调用
    def _compute_energy(self, A):
        return np.trapz(np.abs(A)**2, self.t)
        
    def _compute_fwhm(self, power, t):
        # 简单的 FWHM 计算
        max_p = np.max(power)
        indices = np.where(power >= max_p/2)[0]
        if len(indices) > 1:
            return t[indices[-1]] - t[indices[0]]
        return 0
