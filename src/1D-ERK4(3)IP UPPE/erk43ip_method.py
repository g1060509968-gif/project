"""
ERK4(3)-IP UPPE/FME Solver (Full Field Mode)
基于论文: Balac & Mahé (2013) & Goorjian (1992)
升级内容:
1. 实现了 UPPE/FME (Forward Maxwell Equation) 形式
2. 非线性项包含精确的 n(omega) 修正
3. 修复了步长崩溃时的数据丢失和可视化适配器的 bug
"""

import numpy as np
from scipy.fft import fft, ifft, fftfreq, fftshift, ifftshift
from scipy.interpolate import interp1d

class ERK43IP_UPPE:
    """
    ERK4(3)-IP 求解器 (UPPE 增强版)
    相比普通 GNLSE，本版本在非线性算子中引入了折射率色散修正 n(w0)/n(w)，
    使其适用于极宽光谱 (如 5fs 脉冲) 的仿真。
    """
    
    def __init__(self, material='fused_silica', 
                 gamma=None, n2=None, beam_radius=None, 
                 alpha=0.0, center_wavelength=1064e-9, 
                 use_raman=True, f_R=0.18, tau1=12.2e-15, tau2=32e-15,
                 use_self_steepening=True):
        
        self.material = material
        self.alpha = alpha
        self.lambda0 = center_wavelength
        
        # 物理常数
        self.c = 299792458.0  # m/s
        self.omega0 = 2 * np.pi * self.c / self.lambda0
        
        # --- 自动计算非线性系数 gamma ---
        if gamma is not None:
            self.gamma = gamma
            # 反推 n2 用于记录 (如果未提供)
            self.n2 = 0.0 
            self.beam_radius = 0.0
        elif n2 is not None and beam_radius is not None:
            self.beam_radius = beam_radius
            self.n2 = n2
            self.beam_area = np.pi * beam_radius**2
            self.gamma = (2 * np.pi / self.lambda0) * (n2 / self.beam_area)
            print(f"初始化: 根据参数 n2={n2:.2e}, r={beam_radius*1e6:.1f}um 计算 gamma={self.gamma:.6f} W^-1 m^-1")
        else:
            self.gamma = 0.0
            self.n2 = 0.0
            self.beam_radius = 0.0
            print("警告: 未提供 gamma 或 (n2, beam_radius)，非线性系数默认为 0")
        
        # 高阶非线性参数
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
        self._cached_n_omega = None # [UPPE 新增] 缓存全谱折射率
        self._cached_raman_response = None 
        
        # 计算中心折射率 n0 (用于归一化)
        self.n0 = self._compute_refractive_index(np.array([self.lambda0]))[0]

    def _get_sellmeier_coeffs(self, material):
        """定义常用材料的 Sellmeier 系数"""
        coeffs = {
            'fused_silica': {
                'B': [0.6961663, 0.4079426, 0.8974794],
                'C': [0.0684043**2, 0.1162414**2, 9.896161**2], 
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
        """
        计算全波长色散算子 D(omega)
        并缓存 n(omega) 供 UPPE 非线性项使用
        """
        # 1. 转换为绝对频率
        omega_abs = self.omega0 + omega_grid
        
        # 2. 转换为波长 (处理 omega=0 的奇点)
        omega_safe = np.where(np.abs(omega_abs) < 1e-10, 1e-10, np.abs(omega_abs))
        wavelengths = 2 * np.pi * self.c / omega_safe
        
        # 限制波长范围以防 Sellmeier 发散
        valid_range = self.sellmeier_coeffs['valid_range']
        wavelengths_clamped = np.clip(wavelengths, valid_range[0], valid_range[1])
        
        # 3. 计算全谱折射率 n(omega)
        n_omega = self._compute_refractive_index(wavelengths_clamped)
        
        # [UPPE 关键] 缓存 n(omega) 用于非线性算子校正
        self._cached_n_omega = n_omega
        
        # 4. 计算波数 k(omega)
        k_omega = n_omega * omega_abs / self.c
        
        # 5. 计算参考系参数 (k0, k1)
        d_omega = self.omega0 * 1e-5
        w_center = np.array([self.lambda0])
        w_plus = 2 * np.pi * self.c / (self.omega0 + d_omega) # 注意：频率增->波长减
        w_minus = 2 * np.pi * self.c / (self.omega0 - d_omega)
        
        n0 = self._compute_refractive_index(w_center)[0]
        n_plus = self._compute_refractive_index(np.array([w_plus]))[0]
        n_minus = self._compute_refractive_index(np.array([w_minus]))[0]
        
        k0 = n0 * self.omega0 / self.c
        k_plus = n_plus * (self.omega0 + d_omega) / self.c
        k_minus = n_minus * (self.omega0 - d_omega) / self.c
        
        k1 = (k_plus - k_minus) / (2 * d_omega)
        
        # 6. 构建线性算子 D
        # UPPE/FME 形式: i[k(w) - k0 - k1*w] - alpha/2
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
        
        # 时域响应 h(t)
        h_t = (tau1_safe**2 + tau2_safe**2) / (tau1_safe * tau2_safe**2) * \
              np.exp(-t / tau2_safe) * np.sin(t / tau1_safe)
        
        h_t[0] = 0.0 # 保证因果性
        
        # 转到频域
        H_omega = fft(h_t) * dt
        
        # 归一化
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
        """
        UPPE 增强版非线性算子 N(A)
        包含:
        1. 瞬时克尔效应 + 拉曼卷积
        2. 自陡峭效应 (精确频率依赖)
        3. 折射率色散修正 n(w0)/n(w)
        """
        Intensity = np.abs(A)**2
        
        # --- 1. 计算时域非线性极化强度 P_NL 的包络部分 ---
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
            
        P_NL_time = A * I_eff # 时域非线性极化 (近似)
        
        # --- 2. 转到频域应用 UPPE 修正 ---
        P_NL_freq = fft(P_NL_time)
        
        if self.use_self_steepening:
            if omega is None: omega = self._cached_omega_grid
            
            # 绝对频率
            omega_abs = self.omega0 + omega
            
            # [UPPE 修正]
            # 标准 GNLSE 自陡峭项: shock = omega_abs / omega0
            # UPPE 精确项: coeff ~ omega_abs / (c * n(omega))
            # 相对于 gamma 的修正因子: factor = (omega_abs / omega0) * (n0 / n(omega))
            
            # 确保 n_omega 已计算
            if self._cached_n_omega is None:
                self._compute_dispersion_operator(omega)
            
            n_w = self._cached_n_omega
            
            # 避免除以零 (极低频处)
            n_w_safe = np.where(n_w < 1.0, 1.0, n_w)
            
            # 构造 UPPE 耦合系数
            # 这里的逻辑是：gamma 包含 n0，我们需要除以 n(w) 换回真实的 eps(w)
            uppe_factor = (omega_abs / self.omega0) * (self.n0 / n_w_safe)
            
            nonlinear_term_freq = 1j * self.gamma * uppe_factor * P_NL_freq
            
            return ifft(nonlinear_term_freq)
        else:
            # 如果关闭自陡峭，退化为普通 GNLSE
            return 1j * self.gamma * P_NL_time
    def erk43_step(self, A, omega, h, compute_error=True, alpha5_prev=None, h_prev=None):
        """执行一步 ERK4(3) 方法"""
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
            A3 = beta + (h/30) * (2*alpha4_prime + 3*alpha5_prime)
            norm_A = np.sqrt(np.sum(np.abs(A4)**2))
            if norm_A < 1e-20: norm_A = 1.0
            error = np.sqrt(np.sum(np.abs(A4 - A3)**2)) / norm_A
            return A4, error, alpha5_prime
        else:
            return A4, 0.0, alpha5_prime

    def propagate(self, A0, t, L, tol=1e-4, max_step=1e-2, min_step=1e-11):
        """自适应步长传播主循环 (包含 Bug 修复)"""
        N = len(t)
        dt = t[1] - t[0]
        omega = 2 * np.pi * fftfreq(N, dt)
        
        z = 0.0
        h = max_step / 100
        A = A0.copy()
        
        # 初始化历史记录
        z_history = [0.0]
        A_history = [A0.copy()] 
        
        error_prev = None
        alpha5_prev = None
        h_prev = None
        
        #print(f"开始传播 (UPPE模式): L={L:.4f}m")
        
        while z < L:
            if z + h > L:
                h = L - z
            
            A_new, error, alpha5_new = self.erk43_step(
                A, omega, h, 
                compute_error=True, 
                alpha5_prev=alpha5_prev, 
                h_prev=h_prev
            )
            
            if error < tol:
                # === 成功步 ===
                z += h
                A = A_new
                alpha5_prev = alpha5_new
                h_prev = h
                
                # 保存策略：更密集的保存以捕捉细节
                if len(z_history) < 1000 or (z - z_history[-1]) > L/500:
                    z_history.append(z)
                    A_history.append(A.copy())
                
                # PID 控制器
                factor = 0.95 * (tol / (error + 1e-30))**0.25
                if error_prev is not None and error_prev > 1e-30:
                    factor *= (error_prev / (error + 1e-30))**0.1
                error_prev = error
                h_new = h * min(factor, 2.0)
                h = min(h_new, max_step)
            else:
                # === 失败步 ===
                factor = 0.95 * (tol / (error + 1e-30))**0.25
                h_new = h * max(factor, 0.1)
                h = max(h_new, min_step)
                
                # === 强制执行 (步长崩溃保护) ===
                if h <= min_step:
                    print(f"Warning: Step size collapsed to {h:.2e} at z={z:.4f}")
                    
                    # [修复1] 必须用现在的 min_step 重新计算物理演化
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
                    
                    # [修复2] 重置 PID 历史，防止震荡
                    error_prev = None 
                    
                    # [修复3] 强制步长时也需要保存数据，否则会断片
                    if len(z_history) < 1000 or (z - z_history[-1]) > L/500:
                        z_history.append(z)
                        A_history.append(A.copy())
        
        return np.array(z_history), np.array(A_history), omega

    # =================================================================
    # 工具函数保持不变 (generate_gaussian_pulse 等)
    # =================================================================
    
    def generate_gaussian_pulse(self, t, pulse_energy, pulse_fwhm, chirp=0):
        factor = np.sqrt(np.pi / (4 * np.log(2))) 
        P0 = pulse_energy / (pulse_fwhm * factor)
        T0 = pulse_fwhm / (2 * np.sqrt(np.log(2)))
        A = np.sqrt(P0) * np.exp(-(1 + 1j * chirp) * (t / T0)**2 / 2)
        print(f"生成高斯脉冲: E={pulse_energy*1e9:.2f}nJ, FWHM={pulse_fwhm*1e12:.2f}ps")
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

    def _calculate_fwhm_precise(self, x, y):
        """
        精确计算半高全宽（FWHM）
        使用线性插值实现亚像素级精度
        
        参数：
            x: 自变量数组（如波长、频率或时间）
            y: 因变量数组（如强度）
        
        返回：
            x_left: 左边界位置
            x_right: 右边界位置
            fwhm: 半高全宽
        """
        # 步骤1：确定半最大值
        half_max = np.max(y) / 2
        
        # 步骤2：寻找与半高线的交点
        above = y >= half_max
        crossings = np.where(np.diff(above.astype(int)))[0]
        
        # 步骤3：处理边界情况
        if len(crossings) < 2:
            # 回退策略：使用简单阈值法
            indices = np.where(y >= half_max)[0]
            if len(indices) < 2:
                return None, None, None
            return x[indices[0]], x[indices[-1]], x[indices[-1]] - x[indices[0]]
        
        # 步骤4：左边界线性插值（上升沿）
        idx_left = crossings[0]
        x1, x2 = x[idx_left], x[idx_left+1]
        y1, y2 = y[idx_left], y[idx_left+1]
        # 线性插值公式：x = x1 + (y_target - y1) * (x2 - x1) / (y2 - y1)
        x_left = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1)
        
        # 步骤5：右边界线性插值（下降沿）
        idx_right = crossings[-1]
        x1, x2 = x[idx_right], x[idx_right+1]
        y1, y2 = y[idx_right], y[idx_right+1]
        x_right = x1 + (half_max - y1) * (x2 - x1) / (y2 - y1)
        
        return x_left, x_right, x_right - x_left

    def _exact_fourier_transform(self, t, wavelengths_nm, intensities, use_jacobian=True, 
                               interp_kind='cubic', GDD=0, TOD=0):
        """
        精确傅里叶变换（数值积分方法）
        基于EXACT FOURIER TRANSFORM ANALYSIS copy.py中的高精度方法
        
        参数：
            t : array
                时间数组 (s)
            wavelengths_nm : array
                波长数组 (nm)
            intensities : array
                强度数组
            use_jacobian : bool
                是否应用雅可比变换
            interp_kind : str
                插值方法
            GDD : float
                群延迟色散 (s²)
            TOD : float
                三阶色散 (s³)
                
        返回：
            A_initial : array
                重建的时域场（未归一化）
            center_freq : float
                中心频率
            delta_nu : float
                光谱FWHM (Hz)
        """
        from scipy.integrate import simpson
        from scipy.interpolate import interp1d
        
        # 1. 准备输入数据
        idx = np.argsort(wavelengths_nm)
        wl_sorted = wavelengths_nm[idx]
        int_sorted = intensities[idx]
        
        # 归一化输入强度
        if np.max(int_sorted) > 0:
            int_sorted = int_sorted / np.max(int_sorted)
        
        # 2. 转换为频率并应用雅可比变换
        freqs_input = self.c / (wl_sorted * 1e-9)  # Hz
        
        if use_jacobian:
            # 雅可比变换: I(ν) = I(λ) × |dλ/dν|, 其中 dλ/dν = -λ²/c
            jacobian = (wl_sorted * 1e-9)**2 / self.c
            int_transformed = int_sorted * jacobian
            # 重新归一化
            if np.max(int_transformed) > 0:
                int_transformed = int_transformed / np.max(int_transformed)
        else:
            int_transformed = int_sorted
        
        # 3. 移除重复频率值
        unique_freqs, unique_indices = np.unique(freqs_input, return_index=True)
        if len(unique_freqs) < len(freqs_input):
            freqs_input = unique_freqs
            int_transformed = int_transformed[unique_indices]
        
        # 4. 频率插值
        if freqs_input[0] < freqs_input[-1]:
            freqs_input = freqs_input[::-1]
            int_transformed = int_transformed[::-1]
        
        # 对于cubic插值，需要至少4个点
        if interp_kind in ['cubic', 'quadratic'] and len(freqs_input) < 4:
            interp_kind_actual = 'linear'
        else:
            interp_kind_actual = interp_kind
        
        f_interp = interp1d(freqs_input, int_transformed, 
                            kind=interp_kind_actual, bounds_error=False, fill_value=0.0)
        
        # 5. 创建密集的频率网格用于精确积分
        freq_min = freqs_input.min()
        freq_max = freqs_input.max()
        N_freq_dense = min(4096, len(freqs_input) * 4)
        freq_dense = np.linspace(freq_min, freq_max, N_freq_dense)
        spec_dense = f_interp(freq_dense)
        spec_dense[spec_dense < 0] = 0
        
        # 6. 计算光谱FWHM
        freq_left, freq_right, delta_nu = self._calculate_fwhm_precise(freq_dense, spec_dense)
        if delta_nu:
            center_freq = (freq_left + freq_right) / 2
        else:
            center_freq = np.sum(freq_dense * spec_dense) / np.sum(spec_dense) if np.sum(spec_dense) > 0 else self.c / self.lambda0
        
        # 7. 精确傅里叶变换
        N_time = len(t)
        A_initial = np.zeros(N_time, dtype=complex)
        freq_centered = freq_dense - center_freq
        omega_dense = 2 * np.pi * freq_centered
        
        # 应用色散相位
        dispersion_phase = 0.5 * GDD * omega_dense**2 + (1.0/6.0) * TOD * omega_dense**3
        
        for i, ti in enumerate(t):
            integrand = np.sqrt(spec_dense) * np.exp(1j * dispersion_phase) * np.exp(-1j * omega_dense * ti)
            A_initial[i] = simpson(integrand, x=freq_centered)
        
        A_initial = fftshift(A_initial)
        
        return A_initial, center_freq, delta_nu

    def generate_pulse_from_spectrum(self, t, wavelengths_nm, intensities, pulse_energy, GDD=0, TOD=0, 
                                   use_jacobian=True, interp_kind='cubic', exact_fourier=False,
                                   time_window_auto=True, N_time_points=4096):
        """
        从光谱数据重建脉冲（增强版）
        
        参数：
        -----------
        t : array or None
            时间数组 (s)，如果为None则自动计算
        wavelengths_nm : array
            波长数组 (nm)
        intensities : array
            强度数组 (任意单位，将自动归一化)
        pulse_energy : float
            目标脉冲能量 (J)
        GDD : float
            群延迟色散 (s²)
        TOD : float
            三阶色散 (s³)
        use_jacobian : bool
            是否应用雅可比变换 (波长到频率转换)
        interp_kind : str
            插值方法：'linear', 'cubic', 'quadratic'
        exact_fourier : bool
            是否使用精确傅里叶变换（较慢但更精确）
        time_window_auto : bool
            是否自动计算时间窗口（仅当exact_fourier=True时有效）
        N_time_points : int
            时间点数（仅当exact_fourier=True且time_window_auto=True时有效）
            
        返回：
        -------
        A_initial : array
            重建的时域场
        """
        print("从光谱重建脉冲 (增强版)...")
        print(f"  使用雅可比变换: {use_jacobian}")
        print(f"  插值方法: {interp_kind}")
        print(f"  精确傅里叶变换: {exact_fourier}")
        print(f"  自动时间窗口: {time_window_auto}")
        
        # 如果t为None，自动计算时间网格
        if t is None and exact_fourier and time_window_auto:
            # 使用EXACT FOURIER TRANSFORM ANALYSIS中的方法计算时间窗口
            # 首先计算频率范围
            freqs_input = self.c / (wavelengths_nm * 1e-9)  # Hz
            freq_min = freqs_input.min()
            freq_max = freqs_input.max()
            freq_span = freq_max - freq_min
            
            # 计算最大时间窗口
            if len(freqs_input) > 1:
                freq_spacing = np.abs(freqs_input[1] - freqs_input[0])
                t_max = 1 / freq_spacing
            else:
                t_max = 1 / freq_span if freq_span > 0 else 1e-12
                
            t = np.linspace(-t_max/2, t_max/2, N_time_points)
            print(f"  自动计算时间窗口: ±{t_max*1e12/2:.3f} ps, {N_time_points} 点")
        
        N = len(t)
        dt = t[1] - t[0]
        
        if exact_fourier:
            # 使用精确傅里叶变换（数值积分）
            print("  使用精确傅里叶变换（数值积分）...")
            A_initial, center_freq, delta_nu = self._exact_fourier_transform(
                t, wavelengths_nm, intensities, use_jacobian, interp_kind, GDD, TOD
            )
            
            if delta_nu:
                print(f"  光谱FWHM: {delta_nu/1e12:.6f} THz @ {center_freq/1e12:.2f} THz")
        else:
            # 使用快速傅里叶变换（FFT）
            print("  使用快速傅里叶变换（FFT）...")
            
            # 1. 准备输入数据
            idx = np.argsort(wavelengths_nm)
            wl_sorted = wavelengths_nm[idx]
            int_sorted = intensities[idx]
            
            # 归一化输入强度
            if np.max(int_sorted) > 0:
                int_sorted = int_sorted / np.max(int_sorted)
            
            # 2. 转换为频率并应用雅可比变换
            freqs_input = self.c / (wl_sorted * 1e-9)  # Hz
            
            if use_jacobian:
                # 雅可比变换: I(ν) = I(λ) × |dλ/dν|, 其中 dλ/dν = -λ²/c
                jacobian = (wl_sorted * 1e-9)**2 / self.c
                int_transformed = int_sorted * jacobian
                # 重新归一化
                if np.max(int_transformed) > 0:
                    int_transformed = int_transformed / np.max(int_transformed)
            else:
                int_transformed = int_sorted
            
            # 3. 移除重复频率值
            unique_freqs, unique_indices = np.unique(freqs_input, return_index=True)
            if len(unique_freqs) < len(freqs_input):
                freqs_input = unique_freqs
                int_transformed = int_transformed[unique_indices]
            
            # 4. 频率插值
            if freqs_input[0] < freqs_input[-1]:
                freqs_input = freqs_input[::-1]
                int_transformed = int_transformed[::-1]
            
            # 对于cubic插值，需要至少4个点
            if interp_kind in ['cubic', 'quadratic'] and len(freqs_input) < 4:
                interp_kind_actual = 'linear'
            else:
                interp_kind_actual = interp_kind
            
            f_interp = interp1d(freqs_input, int_transformed, 
                                kind=interp_kind_actual, bounds_error=False, fill_value=0.0)
            
            # 5. 准备仿真频率网格
            omega_rel = 2 * np.pi * fftfreq(N, dt)
            freqs_sim_abs = (omega_rel / (2*np.pi)) + (self.c / self.lambda0)
            
            # 6. 插值到仿真频率网格
            spec_density = f_interp(freqs_sim_abs)
            spec_density[spec_density < 0] = 0
            spec_density[np.isnan(spec_density)] = 0
            
            # 7. 计算频域振幅（考虑相位）
            A_freq_mag = np.sqrt(spec_density)
            
            # 应用色散相位
            dispersion_phase = 0.5 * GDD * omega_rel**2 + (1.0/6.0) * TOD * omega_rel**3
            A_freq_complex = A_freq_mag * np.exp(1j * dispersion_phase)
            
            # 8. 傅里叶变换到时域
            A_temp = ifft(A_freq_complex)
            A_initial = fftshift(A_temp)
        
        # 9. 能量归一化
        current_E = np.trapz(np.abs(A_initial)**2, t)
        if current_E > 0:
            scale_factor = np.sqrt(pulse_energy / current_E)
            A_initial *= scale_factor
            print(f"  能量归一化: 缩放因子 = {scale_factor:.6e}")
        else:
            print("  警告: 重建脉冲能量为零")
        
        # 10. 计算并显示脉冲参数
        intensity = np.abs(A_initial)**2
        max_intensity = np.max(intensity)
        
        # 计算FWHM
        t_left, t_right, fwhm = self._calculate_fwhm_precise(t, intensity)
        if fwhm:
            fwhm_ps = fwhm * 1e12
            print(f"  重建脉冲参数:")
            print(f"    • 脉冲宽度 (FWHM): {fwhm_ps:.3f} ps")
            print(f"    • 峰值功率: {max_intensity/1e6:.2f} MW")
            
            # 计算实际能量
            actual_energy = np.trapz(intensity, t)
            print(f"    • 实际能量: {actual_energy*1e6:.2f} uJ (目标: {pulse_energy*1e6:.2f} uJ)")
            print(f"    • 能量误差: {abs(actual_energy - pulse_energy)/pulse_energy*100:.2f}%")
        
        return A_initial

    def analyze_spectral_phase(self, A, t, fit_order=4, plot_threshold=0.01):
        # 保持你的原代码逻辑不变
        N = len(t)
        dt = t[1] - t[0]
        freq = fftfreq(N, dt)
        omega_rel = fftshift(2 * np.pi * freq)
        spectrum = fftshift(fft(A))
        spectral_intensity = np.abs(spectrum)**2
        
        max_val = np.max(spectral_intensity)
        if max_val > 0: spectral_intensity_norm = spectral_intensity / max_val
        else: spectral_intensity_norm = spectral_intensity

        phi = np.unwrap(np.angle(spectrum))
        mask = spectral_intensity_norm > plot_threshold
        
        if np.sum(mask) < 5: return None

        w_fit = omega_rel[mask]
        phi_fit = phi[mask]
        
        try:
            p = np.polyfit(w_fit, phi_fit, fit_order)
            phi_polynomial = np.polyval(p, omega_rel)
            phi_residual = phi - phi_polynomial
            center_index = np.argmax(spectral_intensity_norm)
            phi_residual = phi_residual - phi_residual[center_index]
            
            omega_abs = omega_rel + self.omega0
            omega_abs[omega_abs < 1e-10] = 1e-10 
            wavelengths_nm = 2 * np.pi * self.c / omega_abs * 1e9
            
            # 对波长排序以方便绘图 (优化)
            sort_idx = np.argsort(wavelengths_nm)
            
            return {
                "wavelength_nm": wavelengths_nm[sort_idx],
                "spectral_intensity": spectral_intensity_norm[sort_idx],
                "residual_phase": phi_residual[sort_idx],
                "mask": mask[sort_idx]
            }
        except:
            return None

    def get_beta2(self):
        # 辅助函数：计算中心 beta2
        w0 = self.omega0
        dw = w0 * 0.001
        
        # 内部调用 self._compute_refractive_index
        n0 = self._compute_refractive_index(np.array([2*np.pi*self.c/w0]))[0]
        np_ = self._compute_refractive_index(np.array([2*np.pi*self.c/(w0+dw)]))[0]
        nm = self._compute_refractive_index(np.array([2*np.pi*self.c/(w0-dw)]))[0]
        
        k0 = n0 * w0 / self.c
        kp = np_ * (w0+dw) / self.c
        km = nm * (w0-dw) / self.c
        
        beta2 = (kp - 2*k0 + km) / (dw**2)
        return beta2

class SimResultAdapter:
    """适配器类 (已修复 Bug)"""
    def __init__(self, solver, z_array, t, A_evolution):
        self.material = solver.material
        self.lambda0 = solver.lambda0
        self.omega0 = solver.omega0
        self.c = solver.c
        
        # [修复] 处理可能为 None 的属性
        br = getattr(solver, 'beam_radius', None)
        self.beam_radius = br if br is not None else 0.0
        
        n2 = getattr(solver, 'n2', None)
        self.n2 = n2 if n2 is not None else 0.0
        
        self.gamma = solver.gamma
        
        self.t = t
        self.Nt = len(t)
        self.dt = t[1] - t[0]
        self.T = self.Nt * self.dt
        
        self.z = z_array
        self.L = z_array[-1]
        self.Nz = len(z_array)
        self.dz = z_array[1] - z_array[0] if len(z_array) > 1 else 0
        
        try:
            self.beta2 = solver.get_beta2()
        except:
            self.beta2 = 0.0
        
        # [修复] 强制转换为 numpy 数组，防止可视化时切片报错
        self.A_evolution = np.array(A_evolution)

# 为向后兼容性添加别名
ERK43IP_FullDispersion = ERK43IP_UPPE
