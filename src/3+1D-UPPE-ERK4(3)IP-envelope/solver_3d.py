import numpy as np
import cupy as cp
from cupyx.scipy.fft import fft, ifft, fft2, ifft2, ifftn
from cupy.fft import fftfreq, fftshift
import time

class ERK43IP_UPPE_3D_Optimized:
    def __init__(self, x_grid, y_grid, t_grid, medium, precision='single'):
        self.medium = medium
        self.c = medium.c
        self.lambda0 = medium.lambda0
        self.omega0 = medium.omega0
        
        if precision == 'double':
            self.float_dtype = cp.float64
            self.complex_dtype = cp.complex128
        else:
            self.float_dtype = cp.float32
            self.complex_dtype = cp.complex64

        self.x = cp.asarray(x_grid, dtype=self.float_dtype)
        self.y = cp.asarray(y_grid, dtype=self.float_dtype)
        self.t = cp.asarray(t_grid, dtype=self.float_dtype)
        
        self.Nx, self.Ny, self.Nt = len(x_grid), len(y_grid), len(t_grid)
        self.dx, self.dy, self.dt = float(x_grid[1]-x_grid[0]), float(y_grid[1]-y_grid[0]), float(t_grid[1]-t_grid[0])

        kx = 2 * np.pi * fftfreq(self.Nx, self.dx)
        ky = 2 * np.pi * fftfreq(self.Ny, self.dy)
        KY, KX = cp.meshgrid(ky, kx, indexing='ij')
        
        self.K2_perp_64 = (KX**2 + KY**2).astype(cp.float64)
        
        omega = 2 * np.pi * fftfreq(self.Nt, self.dt)
        self.omega = cp.asarray(omega, dtype=self.float_dtype)
        self.freq_axis_hz = fftshift(omega) / (2 * np.pi)

        if hasattr(self.medium, 'init_raman'):
            self.medium.init_raman(self.Nt, self.dt)
            
        self._precompute_dispersion_operator()
        self._init_absorber()
        
        self.shock_operator = self.medium.get_shock_operator(self.omega)
        self.A_scale = 1.0

        freq_unshifted = self.omega / (2 * np.pi) 
        f_max = float(cp.max(cp.abs(freq_unshifted))) + 1e-10
        filter_w = cp.exp(- (freq_unshifted / (0.85 * f_max))**16 ).astype(self.float_dtype)
        self.anti_alias_filter = filter_w[cp.newaxis, cp.newaxis, :].astype(self.complex_dtype)
        
        # 预计算绝对频率掩码，用于抹除负频率
        self.omega_abs = self.omega + self.omega0
        self.valid_freq_mask = (self.omega_abs > 0)[cp.newaxis, cp.newaxis, :].astype(self.float_dtype)

    def _precompute_dispersion_operator(self):
        omega_64 = self.omega.astype(cp.float64)
        omega0_64 = float(self.omega0)
        c_64 = float(self.c)
        
        n_omega = self.medium.refractive_index(omega_64 + omega0_64)
        k_val = n_omega * (omega_64 + omega0_64) / c_64
        
        w_center = cp.array([omega0_64], dtype=cp.float64)
        n_center = float(self.medium.refractive_index(w_center)[0])
        
        dw = omega0_64 * 1e-3
        w_plus = cp.array([omega0_64 + dw], dtype=cp.float64)
        n_plus = float(self.medium.refractive_index(w_plus)[0])
        
        k0 = n_center * omega0_64 / c_64
        k_plus = n_plus * (omega0_64 + dw) / c_64
        k1 = (k_plus - k0) / dw 
        k_ref = k0 + k1 * omega_64
        
        k_sq = (k_val**2)[cp.newaxis, cp.newaxis, :]
        k_perp_sq = self.K2_perp_64[:, :, cp.newaxis]
        k_ref_3d = k_ref[cp.newaxis, cp.newaxis, :]
        
        kz = cp.sqrt((k_sq - k_perp_sq).astype(cp.complex128))
        
        numerator = (k_sq - k_ref_3d**2) - k_perp_sq
        denominator = kz + k_ref_3d
        D_op_64 = 1j * (numerator / denominator)
        
        valid_mask = (omega_64 + omega0_64 > 0)[cp.newaxis, cp.newaxis, :]
        self.D_operator = (D_op_64 * valid_mask).astype(self.complex_dtype)

    def _init_absorber(self):
        X, Y = cp.meshgrid(self.x, self.y, indexing='ij')
        R = cp.sqrt(X**2 + Y**2)
        half_w = float(cp.max(cp.abs(self.x)))
        r_edge = 0.98 * half_w
        r_start = 0.85 * half_w
        p_space = 10
        s = (R - r_start) / (r_edge - r_start + 1e-30)
        s = cp.clip(s, 0.0, 1.0)
        self.damp_space = (120.0 * (s ** p_space))[:, :, cp.newaxis].astype(self.float_dtype)

    def erk43_step(self, A_t, h, tol):
        A_w = fft(A_t, axis=-1)
        A_k_w = fft2(A_w, axes=(0,1))
        
        def L_step(field_kw, step):
            return field_kw * cp.exp(step * self.D_operator)
        
        def N_func(field_kw):
            # 一次性完成完整的 3D 逆傅里叶变换
            A_t_local = ifftn(field_kw, axes=(0, 1, 2))
            
            # 还原物理量级
            A_physical = A_t_local * self.A_scale
            
            # 获取区分开的 Kerr/Raman 极化与等离子体项
            Kerr_phys, Plasma_phys = self.medium.calc_nonlinear_response(A_physical, self.dt)
            
            # 压回数值量级
            Kerr_t = Kerr_phys / self.A_scale
            Plasma_t = Plasma_phys / self.A_scale

            # 转到 (x, y, \omega) 域
            Kerr_w = fft(Kerr_t, axis=-1)
            Plasma_w = fft(Plasma_t, axis=-1)

            # 仅对极化项施加自陡峭算子
            N_w_total = self.shock_operator * Kerr_w + Plasma_w

            # 【修复 1】：UPPE 方程核心要求，强行抹除负物理频率（绝对频率 < 0），防止严重混叠和高频震荡
            N_w_total *= self.valid_freq_mask

            # 转回 (k_x, k_y, \omega) 域
            return fft2(N_w_total, axes=(0, 1))

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
        
        # 【修复 2】：改用 L无穷大范数（最大绝对误差）计算截断误差，防止成丝极强局域非线性被全空间低能量背景“稀释”
        max_diff = cp.max(cp.abs(A4_kw - A3_kw))
        max_norm = cp.max(cp.abs(A4_kw))
        err = float(max_diff / (max_norm + 1e-20))
        
        A4_w = ifft2(A4_kw, axes=(0,1))
        A4_w = A4_w * self.anti_alias_filter
        A4_t = ifft(A4_w, axis=-1)
        A4_t *= cp.exp(-self.damp_space * h)
        
        return A4_t, err

    def propagate(self, A0, L, tol=1e-4, max_step=1e-2, min_step=1e-9, return_gpu=False):
        if not isinstance(A0, cp.ndarray):
            A0 = cp.asarray(A0, dtype=self.complex_dtype)
        
        self.A_scale = float(cp.max(cp.abs(A0)))
        if self.A_scale < 1e-30:
            self.A_scale = 1.0

        A = (A0 / self.complex_dtype(self.A_scale * 1.0 + 0.0j)).astype(self.complex_dtype)

        z = 0.0
        h = min(max_step, L/1000.0) 
        print_interval = L / 10.0
        next_print = print_interval
        fail_count = 0
        
        med_name = type(self.medium).__name__
        print(f"[Solver] Start Prop in {med_name}: L={L*1000:.2f}mm, Scale={self.A_scale:.2e}")
        
        try:
            while z < L:
                dist_remaining = L - z
                if h > dist_remaining:
                    h = dist_remaining
                    if h < 1e-15: break
                
                A_new, err = self.erk43_step(A, h, tol)
                
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
        
        print(f"[Solver] Finalizing {med_name}... Reverting scale and computing spectrum.")
        
        A = A * self.A_scale
        A_w_raw = fft(A, axis=-1)
        if return_gpu:
            A_w_final = fftshift(A_w_raw, axes=-1)
            return z, self.freq_axis_hz, A_w_final
        else:
            # 在 GPU 上使用 cupyx 的 fftshift 处理数据
            A_w_final_gpu = fftshift(A_w_raw, axes=-1)
            
            # 将处理好的最终结果拷贝回 CPU
            A_w_final = cp.asnumpy(A_w_final_gpu)
            
            # 处理频率轴
            freq_axis_cpu = self.freq_axis_hz.get() if hasattr(self.freq_axis_hz, 'get') else self.freq_axis_hz
            
            # 显存清理
            del A, A_new, A_w_raw, A_w_final_gpu 
            cp.get_default_memory_pool().free_all_blocks()
            
            return z, freq_axis_cpu, A_w_final