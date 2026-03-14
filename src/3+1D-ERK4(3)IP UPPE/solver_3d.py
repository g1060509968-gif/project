import numpy as np
import cupy as cp
from cupyx.scipy.fft import fft, ifft, fft2, ifft2, fftfreq, fftshift
import time

class ERK43IP_UPPE_3D_Optimized:
    """
    3+1D UPPE 求解器 (ERK43-IP 算法) - 精度重构版
    
    优化亮点:
        - 引入内部电场动态归一化，解决 float32 下非线性项平方导致的浮点数溢出与步长死锁。
        - 强制双精度预计算色散算子，彻底根除大数相减 (k_z - k_ref) 导致的灾难性相消。
        - 外部接口完全保持 SI 单位制不变。
    """
    
    def __init__(self, x_grid, y_grid, t_grid, 
                 material='fused_silica', 
                 n2=3.0e-20,          
                 center_wavelength=800e-9, 
                 use_raman=True, 
                 use_shock=True,
                 f_R=0.18, 
                 tau1=12.2e-15, tau2=32e-15,
                 precision='single'):
        
        # --- 1. 精度配置 (运行时精度) ---
        if precision == 'double':
            self.float_dtype = cp.float64
            self.complex_dtype = cp.complex128
        else:
            self.float_dtype = cp.float32
            self.complex_dtype = cp.complex64

        # --- 2. 物理常数 ---
        self.c = 299792458.0
        self.lambda0 = center_wavelength
        self.omega0 = 2 * np.pi * self.c / self.lambda0
        self.n2 = n2
        
        # 基础非线性系数前缀 (未乘以强度的版本)
        self.gamma_base = 1j * (self.omega0 / self.c) * self.n2
        
        # --- 3. 网格初始化 ---
        self.x = cp.asarray(x_grid, dtype=self.float_dtype)
        self.y = cp.asarray(y_grid, dtype=self.float_dtype)
        self.t = cp.asarray(t_grid, dtype=self.float_dtype)
        
        self.Nx, self.Ny, self.Nt = len(x_grid), len(y_grid), len(t_grid)
        self.dx, self.dy, self.dt = float(x_grid[1]-x_grid[0]), float(y_grid[1]-y_grid[0]), float(t_grid[1]-t_grid[0])

        # --- 4. 频域坐标 ---
        kx = 2 * np.pi * fftfreq(self.Nx, self.dx)
        ky = 2 * np.pi * fftfreq(self.Ny, self.dy)
        KY, KX = cp.meshgrid(ky, kx, indexing='ij')
        
        # 横向波数平方 (此处为了算子精度，暂时存为 float64)
        self.K2_perp_64 = (KX**2 + KY**2).astype(cp.float64)
        
        omega = 2 * np.pi * fftfreq(self.Nt, self.dt)
        self.omega = cp.asarray(omega, dtype=self.float_dtype)
        self.freq_axis_hz = fftshift(omega) / (2 * np.pi)

        # --- 5. 算子预计算 ---
        self.material = material
        self.sellmeier_coeffs = self._get_sellmeier_coeffs(material)
        
        self._precompute_dispersion_operator()
        self._precompute_nonlinear_factors()
        self._init_absorber()
        
        # --- 6. 效应开关与 Raman ---
        self.use_raman = use_raman
        self.use_shock = use_shock
        self.f_R, self.tau1, self.tau2 = f_R, tau1, tau2
        
        if self.use_raman:
            self._precompute_raman_response()
            
        # 动态运行时的缩放系数 (将在 propagate 中初始化)
        self.current_gamma = None
        self.current_shock = None

        # --- 7. 频域抗混叠滤波器 (Anti-aliasing Filter) ---
        # 【关键修复】：必须使用未 shift 的 self.omega，与 fft 输出的原生频率顺序保持对齐
        freq_unshifted = self.omega / (2 * np.pi) 
        f_max = float(cp.max(cp.abs(freq_unshifted))) + 1e-10
        # 构造一个 16 阶超高斯吸收器，只在频谱最外侧 15% 区域生效
        filter_w = cp.exp(- (freq_unshifted / (0.85 * f_max))**16 ).astype(self.float_dtype)
        # 广播为 3D 形状 (1, 1, Nt)
        self.anti_alias_filter = filter_w[cp.newaxis, cp.newaxis, :].astype(self.complex_dtype)

    # =========================================================================
    # 内部辅助函数
    # =========================================================================

    def _get_sellmeier_coeffs(self, material):
        coeffs = {
            'fused_silica': ([0.6961663, 0.4079426, 0.8974794], 
                             [0.0684043**2, 0.1162414**2, 9.896161**2], 
                             (0.2e-6, 6.0e-6)),
            'sapphire': ([1.4313493, 0.65054713, 5.3414021], 
                         [0.0726631**2, 0.1193242**2, 18.028251**2], 
                         (0.2e-6, 5.5e-6)),
            'air': (None, None, None)
        }
        return coeffs.get(material, coeffs['fused_silica'])

    def _compute_refractive_index(self, omega_array):
        # 继承输入数组的精度，确保预计算时使用 float64
        dtype = omega_array.dtype 
        if self.material == 'air':
            return cp.ones_like(omega_array) * 1.00027

        B, C, valid_range = self.sellmeier_coeffs
        omega_safe = cp.where(cp.abs(omega_array) < 1e-12, 1e-12, omega_array)
        wavelengths = 2 * np.pi * self.c / cp.abs(omega_safe)
        
        if valid_range:
            wavelengths = cp.clip(wavelengths, valid_range[0], valid_range[1])
        
        wl_um = wavelengths * 1e6
        wl_sq = wl_um**2
        
        n_sq = cp.ones_like(wavelengths)
        for i in range(3):
            n_sq += (B[i] * wl_sq) / (wl_sq - C[i])
            
        return cp.sqrt(n_sq)

    def _precompute_dispersion_operator(self):
        """
        【关键修复】强制在 float64 空间内进行物理色散差值的计算
        彻底解决 1e14 级别波数相减导致的 float32 灾难性精度丢失
        """
        omega_64 = self.omega.astype(cp.float64)
        omega0_64 = float(self.omega0)
        c_64 = float(self.c)
        
        # 1. 计算宽带 K 向量
        n_omega = self._compute_refractive_index(omega_64 + omega0_64)
        k_val = n_omega * (omega_64 + omega0_64) / c_64
        
        # 2. 计算参考坐标系 K_ref = k0 + k1*dw
        w_center = cp.array([omega0_64], dtype=cp.float64)
        n_center = float(self._compute_refractive_index(w_center)[0])
        
        dw = omega0_64 * 1e-3
        w_plus = cp.array([omega0_64 + dw], dtype=cp.float64)
        n_plus = float(self._compute_refractive_index(w_plus)[0])
        
        k0 = n_center * omega0_64 / c_64
        k_plus = n_plus * (omega0_64 + dw) / c_64
        k1 = (k_plus - k0) / dw 
        k_ref = k0 + k1 * omega_64
        
        # 3. 三维扩展与代数变换 (全部使用 float64)
        k_sq = (k_val**2)[cp.newaxis, cp.newaxis, :]
        k_perp_sq = self.K2_perp_64[:, :, cp.newaxis]
        k_ref_3d = k_ref[cp.newaxis, cp.newaxis, :]
        
        kz = cp.sqrt((k_sq - k_perp_sq).astype(cp.complex128))
        
        # 运用平方差公式消除大数相减带来的误差放大
        numerator = (k_sq - k_ref_3d**2) - k_perp_sq
        denominator = kz + k_ref_3d
        D_op_64 = 1j * (numerator / denominator)
        
        # 计算完成后，安全地降级到指定的运行精度，完美兼顾精度与显存
        self.D_operator = D_op_64.astype(self.complex_dtype)

    def _precompute_nonlinear_factors(self):
        factor = self.gamma_base * ((self.omega + self.omega0) / self.omega0)
        self.shock_base = factor[cp.newaxis, cp.newaxis, :].astype(self.complex_dtype)

    def _init_absorber(self):
        # 空间边界吸收 (原逻辑保持不变)
        X, Y = cp.meshgrid(self.x, self.y, indexing='ij')
        R = cp.sqrt(X**2 + Y**2)
        half_w = float(cp.max(cp.abs(self.x)))
        r_edge = 0.98 * half_w
        r_start = 0.85 * half_w
        p_space = 10
        s = (R - r_start) / (r_edge - r_start + 1e-30)
        s = cp.clip(s, 0.0, 1.0)
        self.damp_space = (120.0 * (s ** p_space))[:, :, cp.newaxis].astype(self.float_dtype)

    def _precompute_raman_response(self):
        dt = self.dt
        t = cp.arange(self.Nt, dtype=self.float_dtype) * dt
        h_t = (self.tau1**2 + self.tau2**2)/(self.tau1 * self.tau2**2) * \
              cp.exp(-t/self.tau2) * cp.sin(t/self.tau1)
        h_t[0] = 0.0 
        H_w = fft(h_t) * dt
        norm = float(cp.abs(H_w[0]))
        if norm > 1e-20: H_w /= norm
        self.H_raman = H_w[cp.newaxis, cp.newaxis, :].astype(self.complex_dtype)

    # =========================================================================
    # 核心算子
    # =========================================================================

    def nonlinear_operator(self, A_t):
        """非线性项 N(A)，使用动态归一化后的系数"""
        Intensity = cp.abs(A_t)**2
        
        if self.use_raman:
            I_w = fft(Intensity, axis=-1)
            I_raman_t = ifft(I_w * self.H_raman, axis=-1)
            I_eff = (1 - self.f_R) * Intensity + self.f_R * I_raman_t
            P_nl_t = A_t * I_eff
        else:
            P_nl_t = A_t * Intensity
            
        P_nl_w = fft(P_nl_t, axis=-1)
        
        if self.use_shock:
            return self.current_shock * P_nl_w
        else:
            return self.current_gamma * P_nl_w

    def erk43_step(self, A_t, h, tol):
        A_w = fft(A_t, axis=-1)
        A_k_w = fft2(A_w, axes=(0,1))
        
        def L_step(field_kw, step):
            return field_kw * cp.exp(step * self.D_operator)
        
        def N_func(field_kw):
            f_w = ifft2(field_kw, axes=(0,1))
            f_t = ifft(f_w, axis=-1)
            nl_w = self.nonlinear_operator(f_t)
            return fft2(nl_w, axes=(0,1))

        # RK4 级联计算
        v_ip = L_step(A_k_w, h/2)
        k1 = N_func(A_k_w)
        alpha1_term = L_step(k1, h/2)
        alpha2 = N_func(v_ip + (h/2)*alpha1_term)
        alpha3 = N_func(v_ip + (h/2)*alpha2)
        
        temp_field = L_step(v_ip + h*alpha3, h/2)
        alpha4_prime = N_func(temp_field)
        
        term_sum = alpha1_term + 2*alpha2 + 2*alpha3
        beta = L_step(v_ip + (h/6)*term_sum, h/2)
        A4_kw = beta + (h/6)*alpha4_prime
        
        alpha5_prime = N_func(A4_kw)
        A3_kw = beta + (h/30)*(2*alpha4_prime + 3*alpha5_prime)
        
        # 误差计算 (由于 A 已经被归一化，此处绝对不会溢出)
        diff_sq = cp.abs(A4_kw - A3_kw)**2
        norm_sq = cp.abs(A4_kw)**2
        err = cp.sqrt(cp.sum(diff_sq)) / (cp.sqrt(cp.sum(norm_sq)) + 1e-20)
        
        A4_w = ifft2(A4_kw, axes=(0,1))
        A4_w = A4_w * self.anti_alias_filter
        A4_t = ifft(A4_w, axis=-1)
        A4_t *= cp.exp(-self.damp_space * h)
        
        return A4_t, float(err)

    # =========================================================================
    # 主传播函数
    # =========================================================================

    def propagate(self, A0, L, tol=1e-4, max_step=1e-2, min_step=1e-9, return_gpu=False):
        """
        执行 3D 传播 (带内部电场自适应归一化)
        """
        if not isinstance(A0, cp.ndarray):
            A0 = cp.asarray(A0, dtype=self.complex_dtype)
        
        # 【关键修复】动态电场归一化，将量级从 1e8 强行拉回 1 附近
        A_scale = float(cp.max(cp.abs(A0)))
        if A_scale < 1e-30: 
            A_scale = 1.0
            
        A = A0.astype(self.complex_dtype) / A_scale
        
        # 同步缩放非线性效应系数 (P_nl 正比于 |A|^2)
        self.current_gamma = self.gamma_base * (A_scale**2)
        self.current_shock = self.shock_base * (A_scale**2)

        z = 0.0
        h = min(max_step, L/1000.0) 
        print_interval = L / 10.0
        next_print = print_interval
        fail_count = 0
        
        print(f"[Solver] Start Prop: L={L*1000:.2f}mm, Shock={self.use_shock}, Raman={self.use_raman}, Scale={A_scale:.2e}")
        
        try:
            while z < L:
                dist_remaining = L - z
                if h > dist_remaining:
                    h = dist_remaining
                    if h < 1e-15: break
                
                A_new, err = self.erk43_step(A, h, tol)
                
                # 步长自适应逻辑
                if err < tol or h <= min_step * 1.01:
                    fail_count = fail_count + 1 if (h <= min_step * 1.01 and err > tol) else 0
                        
                    z += h
                    A = A_new
                    
                    if z >= next_print - 1e-9:
                        print(f"[Solver] Progress: {z/L*100:.1f}% (z={z*1000:.3f} mm)")
                        next_print += print_interval
                    
                    err = max(err, 1e-30)
                    factor = 0.95 * (tol / err)**0.2
                    if fail_count > 0: factor = min(factor, 1.2)
                    h = min(h * min(factor, 2.0), max_step)
                    
                else:
                    err = max(err, 1e-30)
                    factor = 0.95 * (tol / err)**0.2
                    h = max(h * factor, min_step)
                
        except KeyboardInterrupt:
            print(f"\n[Solver] Interrupted at z={z*1000:.3f} mm")
        
        print("[Solver] Finalizing... Reverting scale and computing spectrum.")
        
        # 解除归一化，恢复真实的物理单位电场
        A = A * A_scale
        
        A_w_raw = fft(A, axis=-1)
        if return_gpu:
            A_w_final = fftshift(A_w_raw, axes=-1)
            return z, self.freq_axis_hz, A_w_final
        else:
            A_w_cpu = cp.asnumpy(A_w_raw)
            freq_axis_cpu = self.freq_axis_hz.get() if hasattr(self.freq_axis_hz, 'get') else self.freq_axis_hz
            A_w_final = fftshift(A_w_cpu, axes=-1)
            
            del A, A_new, A_w_raw
            cp.get_default_memory_pool().free_all_blocks()
            
            return z, freq_axis_cpu, A_w_final