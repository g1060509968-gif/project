"""
超快激光非线性晶体传播仿真器 - 核心模块 (2+1)D 版本
"""

import numpy as np
from scipy.fftpack import fftfreq # 仅用于生成网格，FFT本身使用numpy
from scipy.interpolate import interp1d

class NonlinearCrystalSimulator:
    """(2+1)D 超快激光非线性晶体传播仿真器"""
    
    def __init__(self, crystal_length, num_z_steps, time_window, num_t_points,
                 x_window, num_x_points,
                 center_wavelength=1064e-9,
                 material='fused_silica', dispersion_mode='sellmeier'):
        """初始化仿真参数"""
        # 输入验证
        if crystal_length <= 0:
            raise ValueError("crystal_length 必须为正数")
        if num_z_steps < 10:
            raise ValueError("num_z_steps 必须 >= 10")
        if time_window <= 0:
            raise ValueError("time_window 必须为正数")
        if num_t_points < 64:
            raise ValueError("num_t_points 必须 >= 64")
        if x_window <= 0:
            raise ValueError("x_window 必须为正数")
        if num_x_points < 32:
            raise ValueError("num_x_points 必须 >= 32")
        # 检查是否为2的幂次（FFT效率优化）
        if (num_t_points & (num_t_points - 1)) != 0:
            print(f"⚠️  警告: num_t_points={num_t_points} 不是2的幂次，FFT效率可能降低")
        if (num_x_points & (num_x_points - 1)) != 0:
            print(f"⚠️  警告: num_x_points={num_x_points} 不是2的幂次，FFT效率可能降低")
        if dispersion_mode not in ['sellmeier', 'taylor']:
            raise ValueError("dispersion_mode 必须为 'sellmeier' 或 'taylor'")
        
        self.L = crystal_length
        self.Nz = num_z_steps
        self.dz = crystal_length / num_z_steps
        self.z = np.linspace(0, crystal_length, num_z_steps)
        
        # 时间和频率网格
        self.T = time_window
        self.Nt = num_t_points
        self.dt = time_window / num_t_points
        self.t = np.linspace(-time_window/2, time_window/2, num_t_points)
        self.omega = 2 * np.pi * fftfreq(num_t_points, self.dt)
        
        # 横向空间和空间频率网格
        self.x_window = x_window
        self.Nx = num_x_points
        self.dx = x_window / num_x_points
        self.x = np.linspace(-x_window/2, x_window/2, num_x_points)
        self.kx = 2 * np.pi * fftfreq(num_x_points, self.dx)
        
        # 物理常数
        self.c = 299792458
        self.lambda0 = center_wavelength
        self.omega0 = 2 * np.pi * self.c / center_wavelength
        
        # 材料和色散
        self.material = material
        self.dispersion_mode = dispersion_mode
        self.sellmeier_coeffs = self._get_sellmeier_coeffs(material)
        self._compute_dispersion_coefficients() # 计算beta2, beta3等
        
        # 初始化非线性和损耗参数
        self.n2 = 0
        self.alpha = 0
        self.self_steepening = False

        print(f"✓ (2+1)D 仿真器初始化完成")
        print(f"  晶体长度: {crystal_length*1e3:.2f} mm, 步数: {num_z_steps}")
        print(f"  时间窗口: {time_window*1e12:.1f} ps, 采样点: {num_t_points}")
        print(f"  空间窗口: {x_window*1e3:.2f} mm, 采样点: {num_x_points}")
        print(f"  中心波长: {center_wavelength*1e9:.1f} nm, 材料: {material}")
        print(f"  色散模式: {dispersion_mode}")

    def set_parameters(self, n2=0, alpha=0, self_steepening=False):
        """设置非线性系数和损耗"""
        self.n2 = n2
        self.alpha = alpha
        self.self_steepening = self_steepening
        print("\n参数设置:")
        print(f"  n2 (非线性折射率): {self.n2:.2e} m²/W")
        print(f"  alpha (线性损耗): {self.alpha:.2f} /m")
        print(f"  Self-steepening: {'启用' if self.self_steepening else '禁用'}")

    def _get_sellmeier_coeffs(self, material):
        """获取材料的Sellmeier系数"""
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
        """使用Sellmeier方程计算折射率"""
        coeffs = self.sellmeier_coeffs
        B, C = coeffs['B'], coeffs['C']
        wavelength = np.atleast_1d(wavelength)
        is_scalar = wavelength.size == 1
        wavelength_um = wavelength * 1e6
        n_squared = np.ones_like(wavelength_um, dtype=float)
        for i in range(3):
            n_squared += (B[i] * wavelength_um**2) / (wavelength_um**2 - C[i])
        n = np.sqrt(n_squared)
        return float(n[0]) if is_scalar else n

    def _compute_dispersion_coefficients(self):
        """计算色散相关的系数 beta0, beta1, beta2, beta3"""
        n0 = self._compute_refractive_index(self.lambda0)
        domega = self.omega0 * 1e-5
        
        def get_beta(omega):
            if omega == 0: return 0
            lmbd = 2 * np.pi * self.c / omega
            n = self._compute_refractive_index(lmbd)
            return n * omega / self.c

        # Beta0, Beta1 (GVD)
        self.beta0 = get_beta(self.omega0)
        self.beta1 = (get_beta(self.omega0 + domega) - get_beta(self.omega0 - domega)) / (2 * domega)
        # Beta2, Beta3
        beta_p1 = get_beta(self.omega0 + domega)
        beta_m1 = get_beta(self.omega0 - domega)
        self.beta2 = (beta_p1 - 2*self.beta0 + beta_m1) / domega**2
        
        beta_p2 = get_beta(self.omega0 + 2*domega)
        beta_m2 = get_beta(self.omega0 - 2*domega)
        self.beta3 = (beta_p2 - 2*beta_p1 + 2*beta_m1 - beta_m2) / (2 * domega**3)

    def _compute_full_dispersion_curve(self):
        """计算全频率范围的传播常数 k(ω)"""
        omega_total = self.omega0 + self.omega
        # 避免 omega_total 为 0
        omega_total[omega_total == 0] = 1e-10
        
        wavelength = 2 * np.pi * self.c / np.abs(omega_total)
        
        valid_range = self.sellmeier_coeffs['valid_range']
        wavelength = np.clip(wavelength, valid_range[0], valid_range[1])
        
        n_omega = self._compute_refractive_index(wavelength)
        k_omega = n_omega * omega_total / self.c
        return k_omega

    def _compute_linear_operator(self):
        """计算 (2+1)D 线性算符 D(kx, ω)"""
        # 1. 色散部分 (沿 t 轴, axis=1)
        if self.dispersion_mode == 'sellmeier':
            k_omega = self._compute_full_dispersion_curve()
            # 移动参考系变换
            D_dispersion = 1j * (k_omega - self.beta0 - self.beta1 * self.omega)
        else:
            D_dispersion = 0.5j * self.beta2 * self.omega**2 + (1j * self.beta3 / 6) * self.omega**3
        
        D_dispersion -= self.alpha / 2

        # 2. 衍射部分 (沿 x 轴, axis=0)
        n0 = self._compute_refractive_index(self.lambda0)
        k0 = n0 * self.omega0 / self.c
        D_diffraction = -1j * self.kx**2 / (2 * k0)

        # 3. 使用广播合并成二维算符
        D = D_diffraction[:, np.newaxis] + D_dispersion[np.newaxis, :]
        
        print("  ✓ 使用 (2+1)D 线性算符 (包含衍射和色散)")
        return D

    def _nonlinear_operator(self, A):
        """计算点对点的非线性效应"""
        gamma_2d = self.n2 * self.omega0 / self.c
        intensity = np.abs(A)**2
        N = 1j * gamma_2d * intensity * A
        
        if self.self_steepening:
            # 注意: FFT作用在时间轴 (axis=1)
            T_shock = 1 / self.omega0
            term = A + 1j * T_shock * np.fft.ifft(1j * self.omega * np.fft.fft(A, axis=1), axis=1)
            N = 1j * gamma_2d * np.abs(term)**2 * term
        
        return N

    def gaussian_pulse(self, pulse_energy, pulse_width_fwhm, beam_waist_x, C=0):
        """生成 (2+1)D 高斯初始脉冲"""
        P0_temporal = pulse_energy / (pulse_width_fwhm * np.sqrt(np.pi/(4*np.log(2))))
        T0 = pulse_width_fwhm / (2*np.sqrt(np.log(2)))
        temporal_profile = np.sqrt(P0_temporal) * np.exp(-(1 + 1j*C) * (self.t/T0)**2 / 2)

        spatial_profile = np.exp(-(self.x / beam_waist_x)**2)

        A = spatial_profile[:, np.newaxis] * temporal_profile[np.newaxis, :]
        print("\n✓ 生成 (2+1)D 高斯初始脉冲")
        return A

    def propagate(self, A_initial, verbose=True):
        """执行分步傅里叶传播"""
        print("\n>>> 开始 (2+1)D 传播...")
        A = A_initial.copy()
        A_evolution = np.zeros((self.Nz, self.Nx, self.Nt), dtype=complex)
        A_evolution[0, :, :] = A
        
        D = self._compute_linear_operator()
        exp_D_half = np.exp(D * self.dz / 2)

        for i in range(1, self.Nz):
            if verbose and i % (self.Nz // 10) == 0:
                print(f"  传播进度: {i / self.Nz * 100:.0f}%")

            # 线性步 1
            A_freq = np.fft.fft2(A)
            A_freq *= exp_D_half
            A = np.fft.ifft2(A_freq)
            
            # 非线性步 (RK4)
            k1 = self._nonlinear_operator(A)
            k2 = self._nonlinear_operator(A + 0.5 * self.dz * k1)
            k3 = self._nonlinear_operator(A + 0.5 * self.dz * k2)
            k4 = self._nonlinear_operator(A + self.dz * k3)
            A += (self.dz / 6) * (k1 + 2*k2 + 2*k3 + k4)
            
            # 线性步 2
            A_freq = np.fft.fft2(A)
            A_freq *= exp_D_half
            A = np.fft.ifft2(A_freq)
            
            A_evolution[i, :, :] = A
            
        print("<<< 传播完成 <<<\n")
        return A_evolution

    def _compute_fwhm(self, power, time_axis):
        """计算一维功率分布的FWHM"""
        power = np.abs(power)
        half_max = np.max(power) / 2.0
        indices = np.where(power > half_max)[0]
        if len(indices) < 2: return 0
        return np.abs(time_axis[indices[-1]] - time_axis[indices[0]])

    def _compute_energy(self, A_1d):
        """计算一维脉冲的能量"""
        return np.trapz(np.abs(A_1d)**2, self.t)

    def _compute_energy_2d(self, A_2d):
        """
        计算 (2+1)D 场 A(x, t) 的总能量。
        能量是强度在时间和空间上的二重积分。
        
        Args:
            A_2d: 一个 (Nx, Nt) 的二维复数场数组。
            
        Returns:
            总能量 (J)。
        """
        # 强度 |A(x,t)|²
        intensity = np.abs(A_2d)**2
        
        # 首先沿时间轴 (axis=1) 积分，得到每个x位置的线性能量密度 (J/m)
        line_energy_density = np.trapz(intensity, self.t, axis=1)
        
        # 然后沿空间轴 (axis=0) 积分，得到总能量 (J)
        total_energy = np.trapz(line_energy_density, self.x, axis=0)
        
        return total_energy
