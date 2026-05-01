import numpy as np
import cupy as cp
from cupyx.scipy.fft import rfft, irfft, fft2, ifft2
from cupy.fft import fftfreq, rfftfreq
import time

class ERK43_RealField_UPPE_3D:
    def __init__(self, x_grid, y_grid, t_grid, medium, precision='single'):
        self.medium = medium
        self.c = medium.c
        self.lambda0 = medium.lambda0
        self.omega0 = medium.omega0 # 仅用作参考坐标系的群速度计算基准
        self.eps0 = 8.85418781e-12
        
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

        # 空间频域
        kx = 2 * np.pi * fftfreq(self.Nx, self.dx)
        ky = 2 * np.pi * fftfreq(self.Ny, self.dy)
        KX, KY = cp.meshgrid(kx, ky, indexing='ij')
        self.K2_perp_64 = (KX**2 + KY**2).astype(cp.float64)
        
        # 时间频域改为 RFFT，只有 >= 0 的绝对物理频率
        omega = 2 * np.pi * rfftfreq(self.Nt, self.dt)
        self.omega_64 = cp.asarray(omega, dtype=cp.float64) 
        self.freq_axis_hz = self.omega_64 / (2 * np.pi)

        if hasattr(self.medium, 'init_raman'):
            self.medium.init_raman(self.Nt, self.dt)
            
        # 预计算线性色散算子和非线性耦合系数
        self._precompute_operators()
        self._init_absorber()
        
        self.E_scale = 1.0

        # 抗混叠滤波器 (仅对正频率滤波)
        f_max = float(cp.max(self.freq_axis_hz)) + 1e-10
        filter_w = cp.exp(- (self.freq_axis_hz / (0.85 * f_max))**16 ).astype(self.float_dtype)
        self.anti_alias_filter = filter_w[cp.newaxis, cp.newaxis, :].astype(self.complex_dtype)

    def _precompute_operators(self):
        omega_64 = self.omega_64
        c_64 = float(self.c)
        
        # 避免直流分量(omega=0)除以0的奇点
        omega_safe = cp.where(omega_64 == 0, 1e-12, omega_64)
        n_omega = self.medium.refractive_index(omega_safe)
        k_val = n_omega * omega_safe / c_64
        
        # 计算参考坐标系的移动速度 (以 omega0 的群速度为准，保持脉冲在时间窗口内)
        w_center = cp.array([self.omega0], dtype=cp.float64)
        n_center = float(self.medium.refractive_index(w_center)[0])
        k0 = n_center * self.omega0 / c_64
        
        dw = self.omega0 * 1e-3
        w_plus = cp.array([self.omega0 + dw], dtype=cp.float64)
        n_plus = float(self.medium.refractive_index(w_plus)[0])
        k_plus = n_plus * (self.omega0 + dw) / c_64
        k1 = (k_plus - k0) / dw 
        
        # 【核心修正 1】：全场模型中，参考系的移动必须是纯粹的时间延迟 tau = t - z/v_g
        # 绝对不能有常数截距，否则会导致严重的 CEP 滑移！
        k_ref = k1 * omega_64
        
        k_sq = (k_val**2)[cp.newaxis, cp.newaxis, :]
        k_perp_sq = self.K2_perp_64[:, :, cp.newaxis]
        
        # 严格的纵向波矢
        kz_sq = k_sq - k_perp_sq
        # 传播波取正常实数开方，凋落波强制赋予负虚部以保证物理衰减
        kz = cp.where(kz_sq >= 0,
                    cp.sqrt(kz_sq.astype(cp.complex128)),
                    -1j * cp.sqrt(cp.abs(kz_sq)).astype(cp.complex128))
        
        # 极简的线性色散推进算子
        self.D_operator = (-1j * (kz - k_ref[cp.newaxis, cp.newaxis, :])).astype(self.complex_dtype)
        
        # --- 预计算非线性源项的耦合系数 ---
        # 真实的 UPPE 方程： dE/dz = ... + i*(w^2)/(2*eps0*c^2*kz)*P_NL - w/(2*eps0*c^2*kz)*J
        kz_safe = cp.where(cp.abs(kz) < 1e-10, 1e-10, kz)
        
        self.UPPE_P_coeff = (-1j * (omega_64**2) / (2 * self.eps0 * c_64**2 * kz_safe)).astype(self.complex_dtype)
        self.UPPE_J_coeff = (- omega_64 / (2 * self.eps0 * c_64**2 * kz_safe)).astype(self.complex_dtype)
        
        # 抹除 DC(直流) 分量的非线性驱动，防止数值爆炸
        self.UPPE_P_coeff[:, :, 0] = 0.0
        self.UPPE_J_coeff[:, :, 0] = 0.0

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

    def erk43_step(self, E_t, h, tol):
        # 时域到频域改用 rfft
        E_w = rfft(E_t, axis=-1)
        E_kw = fft2(E_w, axes=(0,1))
        
        def L_step(field_kw, step):
            return field_kw * cp.exp(step * self.D_operator)
        
        def N_func(field_kw):
            # 频域转回时域（实数电场）
            E_w_local = ifft2(field_kw, axes=(0, 1))
            E_t_local = irfft(E_w_local, n=self.Nt, axis=-1)
            
            # 还原真实的电场物理量级 (V/m)
            E_physical = E_t_local * self.E_scale
            
            # 获取真实的极化强度 P (C/m^2) 和电流密度 J (A/m^2)
            P_phys, J_phys = self.medium.calc_nonlinear_response(E_physical, self.dt)
            
            # 压回数值量级
            P_t = P_phys / self.E_scale
            J_t = J_phys / self.E_scale

            # 转回绝对正频域
            P_w = rfft(P_t, axis=-1)
            J_w = rfft(J_t, axis=-1)

            # 严格的电磁波源项组装
            N_w_total = self.UPPE_P_coeff * P_w + self.UPPE_J_coeff * J_w

            return fft2(N_w_total, axes=(0, 1))

        # 4阶相互作用绘景 (IP-RK4)
        v_ip = L_step(E_kw, h/2)
        k1 = N_func(E_kw)
        alpha1_term = L_step(k1, h/2)
        alpha2 = N_func(v_ip + (h/2)*alpha1_term)
        alpha3 = N_func(v_ip + (h/2)*alpha2)
        
        temp_field = L_step(v_ip + h*alpha3, h/2)
        alpha4_prime = N_func(temp_field)
        
        term_sum = alpha1_term + 2*alpha2 + 2*alpha3
        beta = L_step(v_ip + (h/6)*term_sum, h/2)
        E4_kw = beta + (h/6)*alpha4_prime  # 完美的 4 阶结果
        
        # 【核心修正 2】：极简的降阶误差估计，省去 2 次昂贵的 N_func (FFT) 调用
        E_low_kw = L_step(v_ip + (h/2)*alpha1_term + (h/2)*alpha2, h/2)
        
        max_diff = cp.max(cp.abs(E4_kw - E_low_kw))
        max_norm = cp.max(cp.abs(E4_kw))
        err = float(max_diff / (max_norm + 1e-20))
        
        # 频域滤波并转回时域实数场
        E4_w = ifft2(E4_kw, axes=(0,1))
        E4_w = E4_w * self.anti_alias_filter
        E4_t = irfft(E4_w, n=self.Nt, axis=-1)
        E4_t *= cp.exp(-self.damp_space * h)
        
        return E4_t, err

    def propagate(self, E0, L, tol=1e-4, max_step=1e-2, min_step=1e-9, return_gpu=False):
        # 确保输入是实数场
        if not isinstance(E0, cp.ndarray):
            E0 = cp.asarray(E0, dtype=self.float_dtype)
        
        self.E_scale = float(cp.max(cp.abs(E0)))
        if self.E_scale < 1e-30:
            self.E_scale = 1.0

        E = (E0 / self.float_dtype(self.E_scale)).astype(self.float_dtype)

        z = 0.0
        h = min(max_step, L/1000.0) 
        print_interval = L / 10.0
        next_print = print_interval
        fail_count = 0
        
        med_name = type(self.medium).__name__
        print(f"[Solver] Start Real-Field UPPE in {med_name}: L={L*1000:.2f}mm, Scale={self.E_scale:.2e} V/m")
        
        try:
            while z < L:
                dist_remaining = L - z
                if h > dist_remaining:
                    h = dist_remaining
                    if h < 1e-15: break
                
                E_new, err = self.erk43_step(E, h, tol)
                
                if err < tol or h <= min_step * 1.01:
                    fail_count = fail_count + 1 if (h <= min_step * 1.01 and err > tol) else 0
                        
                    z += h
                    E = E_new
                    
                    if z >= next_print - 1e-9:
                        print(f"[Solver] Progress: {z/L*100:.1f}% (z={z*1000:.3f} mm)")
                        next_print += print_interval
                    
                    err = max(err, 1e-30)
                    # 【核心修正 3】：适配新的降阶误差估算，缩放因子指数改为 0.333
                    factor = 0.9 * (tol / err)**0.333
                    if fail_count > 0: factor = min(factor, 1.2)
                    h = min(h * min(factor, 2.0), max_step)
                    h = max(h, min_step) 
                else:
                    err = max(err, 1e-30)
                    # 失败步，退回并减小步长
                    factor = 0.9 * (tol / err)**0.333
                    h = max(h * factor, min_step)
                
        except KeyboardInterrupt:
            print(f"\n[Solver] Interrupted at z={z*1000:.3f} mm")
        
        print(f"[Solver] Finalizing {med_name}...")
        
        E = E * self.E_scale
        E_w_final_raw = rfft(E, axis=-1)
        
        if return_gpu:
            return z, self.freq_axis_hz, E_w_final_raw
        else:
            E_w_final = cp.asnumpy(E_w_final_raw)
            freq_axis_cpu = self.freq_axis_hz.get() if hasattr(self.freq_axis_hz, 'get') else self.freq_axis_hz
            
            del E, E_new, E_w_final_raw 
            cp.get_default_memory_pool().free_all_blocks()
            
            return z, freq_axis_cpu, E_w_final