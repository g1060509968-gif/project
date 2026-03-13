"""
超快激光非线性晶体传播仿真器 - 核心模块
"""

import numpy as np
from scipy.fftpack import fft, ifft, fftfreq, fftshift, ifftshift
from scipy.interpolate import interp1d


class NonlinearCrystalSimulator:
    """超快激光非线性晶体传播仿真器"""
    
    def __init__(self, crystal_length, num_z_steps, time_window, num_t_points, 
                 center_wavelength=1064e-9, beam_radius=0.4e-3,
                 material='fused_silica', dispersion_mode='sellmeier'):
        """初始化仿真参数
        
        Args:
            crystal_length: 晶体长度 (m)
            num_z_steps: 传播步数
            time_window: 时间窗口 (s)
            num_t_points: 时间采样点数
            center_wavelength: 中心波长 (m)
            beam_radius: 光束半径 (m)
            material: 材料类型 ('fused_silica', 'sapphire', 'yag', 'custom')
            dispersion_mode: 色散模式 ('sellmeier' 真实曲线或 'taylor' β₂β₃近似)
        """
        # 输入验证
        if crystal_length <= 0:
            raise ValueError("crystal_length 必须为正数")
        if num_z_steps < 10:
            raise ValueError("num_z_steps 必须 >= 10")
        if time_window <= 0:
            raise ValueError("time_window 必须为正数")
        if num_t_points < 64:
            raise ValueError("num_t_points 必须 >= 64")
        # 检查是否为2的幂次（FFT效率优化）
        if (num_t_points & (num_t_points - 1)) != 0:
            print(f"⚠️  警告: num_t_points={num_t_points} 不是2的幂次，FFT效率可能降低")
        if beam_radius <= 0:
            raise ValueError("beam_radius 必须为正数")
        if dispersion_mode not in ['sellmeier', 'taylor']:
            raise ValueError("dispersion_mode 必须为 'sellmeier' 或 'taylor'")
        
        self.L = crystal_length
        self.Nz = num_z_steps
        self.dz = crystal_length / num_z_steps
        self.z = np.linspace(0, crystal_length, num_z_steps)  # 传播距离数组
        
        # 时间和频率网格
        self.T = time_window
        self.Nt = num_t_points
        self.dt = time_window / num_t_points
        self.t = np.linspace(-time_window/2, time_window/2, num_t_points)
        self.omega = 2 * np.pi * fftfreq(num_t_points, self.dt)
        
        # 物理常数
        self.c = 299792458  # m/s
        self.lambda0 = center_wavelength
        self.omega0 = 2 * np.pi * self.c / center_wavelength
        
        # 光束参数
        self.beam_radius = beam_radius
        self.beam_area = np.pi * beam_radius**2
        
        # 色散模式设置
        self.material = material
        self.dispersion_mode = dispersion_mode
        self.sellmeier_coeffs = self._get_sellmeier_coeffs(material)
        
        # 初始化损耗参数
        self.alpha = 0
        
        print(f"✓ 仿真器初始化完成")
        print(f"  晶体长度: {crystal_length*1e3:.2f} mm, 步数: {num_z_steps}")
        print(f"  时间窗口: {time_window*1e12:.1f} ps, 采样点: {num_t_points}")
        print(f"  中心波长: {center_wavelength*1e9:.1f} nm, 材料: {material}")
        print(f"  色散模式: {dispersion_mode}")
    
    def _get_sellmeier_coeffs(self, material):
        """获取材料的Sellmeier系数"""
        coeffs = {
            'fused_silica': {
                'B': [0.6961663, 0.4079426, 0.8974794],
                'C': [0.0684043**2, 0.1162414**2, 9.896161**2],
                'valid_range': (0.21e-6, 6.7e-6)  # m
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
        """使用Sellmeier方程计算折射率
        
        支持标量和数组输入，向量化实现以提高性能
        
        Args:
            wavelength: 波长 (m)，可以是标量或数组
            
        Returns:
            折射率，保持输入的形状
        """
        coeffs = self.sellmeier_coeffs
        B, C = coeffs['B'], coeffs['C']
        
        # 确保wavelength是数组以支持向量化操作
        wavelength = np.atleast_1d(wavelength)
        is_scalar = wavelength.size == 1
        
        # 将波长转换为μm，因为Sellmeier系数C的单位是μm²
        wavelength_um = wavelength * 1e6
        
        # Sellmeier方程: n²(λ) = 1 + Σ(Bᵢ·λ²)/(λ² - Cᵢ)
        # 注意：C[i]的单位是μm²，所以wavelength_um的平方是μm²
        n_squared = np.ones_like(wavelength_um, dtype=float)
        for i in range(3):
            n_squared += (B[i] * wavelength_um**2) / (wavelength_um**2 - C[i])
        
        n = np.sqrt(n_squared)
        
        # 如果输入是标量，返回标量
        if is_scalar:
            return float(n[0])
        return n
    
    def _compute_dispersion_from_sellmeier(self):
        """从Sellmeier方程计算色散系数β₂, β₃
        
        使用正确的物理公式：
        β(ω) = n(ω)·ω/c
        β₂ = d²β/dω² 在 ω₀ 处
        β₃ = d³β/dω³ 在 ω₀ 处
        
        使用更精确的五点有限差分公式
        """
        lambda0 = self.lambda0
        omega0 = self.omega0
        
        # 计算中心波长处的折射率
        n0 = self._compute_refractive_index(lambda0)
        
        # 使用适当的频率步长以获得更好的数值精度
        domega = omega0 * 1e-4  # 调整频率步长
        
        # 计算 ω₀ ± δω, ω₀ ± 2δω 处的波长和折射率
        lambda_plus = 2 * np.pi * self.c / (omega0 + domega)
        lambda_minus = 2 * np.pi * self.c / (omega0 - domega)
        lambda_plus2 = 2 * np.pi * self.c / (omega0 + 2*domega)
        lambda_minus2 = 2 * np.pi * self.c / (omega0 - 2*domega)
        
        n_plus = self._compute_refractive_index(lambda_plus)
        n_minus = self._compute_refractive_index(lambda_minus)
        n_plus2 = self._compute_refractive_index(lambda_plus2)
        n_minus2 = self._compute_refractive_index(lambda_minus2)
        
        # 计算 β(ω) = n(ω)·ω/c
        beta_0 = n0 * omega0 / self.c
        beta_plus = n_plus * (omega0 + domega) / self.c
        beta_minus = n_minus * (omega0 - domega) / self.c
        beta_plus2 = n_plus2 * (omega0 + 2*domega) / self.c
        beta_minus2 = n_minus2 * (omega0 - 2*domega) / self.c
        
        # 二阶导数：β₂ = d²β/dω²（中心差分）
        self.beta2 = (beta_plus - 2*beta_0 + beta_minus) / (domega**2)
        
        # 三阶导数：β₃ = d³β/dω³（五点公式）- 修正公式
        # 正确的五点公式：f'''(x) ≈ [f(x+2h) - 2f(x+h) + 2f(x-h) - f(x-2h)] / (2h³)
        self.beta3 = (beta_plus2 - 2*beta_plus + 2*beta_minus - beta_minus2) / (2 * domega**3)
    
    def _compute_full_dispersion_curve(self):
        """计算全波长范围的色散曲线 - 修正版本
        
        使用Sellmeier方程计算每个频率分量的传播常数β(ω)
        修正了频率计算、负频率处理和导数计算的问题
        
        Returns:
            beta_omega: 传播常数数组
            beta0: 中心频率处的传播常数
            dk_domega: 群速度相关的导数
        """
        # 计算总角频率 - 修正：确保频率网格正确
        # self.omega 是相对于中心频率的偏移，所以总频率为 omega0 + omega
        omega_total = self.omega0 + self.omega
        
        # 修正负频率处理：只处理物理上有意义的频率
        # 对于负频率，我们仍然需要计算，但要注意物理意义
        # 使用原始频率计算波长，因为折射率是频率的偶函数
        wavelength = 2 * np.pi * self.c / np.abs(omega_total)
        
        # 检查波长范围并裁剪到有效范围，但避免在边界处产生不连续性
        valid_range = self.sellmeier_coeffs['valid_range']
        # 使用更温和的裁剪方式，避免边界不连续
        wavelength = np.where(wavelength < valid_range[0], valid_range[0], wavelength)
        wavelength = np.where(wavelength > valid_range[1], valid_range[1], wavelength)
        
        # 使用向量化的折射率计算（一次性处理所有波长）
        n_omega = self._compute_refractive_index(wavelength)
        
        # 计算波数 k(ω) = n(ω)·ω/c - 使用原始频率（包含符号）
        k_omega = n_omega * omega_total / self.c
        
        # 中心频率的波数
        n0 = self._compute_refractive_index(self.lambda0)
        k0 = n0 * self.omega0 / self.c
        
        # 修正群速度导数计算：使用正确的频率差
        dk_domega = np.zeros_like(k_omega)
        for i in range(1, len(k_omega)-1):
            # 使用实际的频率差，而不是固定的步长
            domega = self.omega[i+1] - self.omega[i-1]
            dk_domega[i] = (k_omega[i+1] - k_omega[i-1]) / domega
        # 边界使用单侧差分
        dk_domega[0] = (k_omega[1] - k_omega[0]) / (self.omega[1] - self.omega[0])
        dk_domega[-1] = (k_omega[-1] - k_omega[-2]) / (self.omega[-1] - self.omega[-2])
        
        # 存储结果供其他方法使用
        self.beta_full = k_omega
        self.wavelengths_full = wavelength
        self.beta0 = k0
        self.dk_domega = dk_domega
        
        print(f"✓ 全色散曲线计算完成（修正版）: 波长范围 {wavelength.min()*1e9:.1f} - {wavelength.max()*1e9:.1f} nm")
        return k_omega, k0, dk_domega
    
    def _compute_full_dispersion_operator(self):
        """计算全波长色散算符
        
        使用完整的Sellmeier色散曲线构建色散算符
        D(ω) = i[k(ω) - k₀ - k'₀·Δω] - α/2
        
        Returns:
            色散算符数组
        """
        # 确保已计算全色散曲线
        if not hasattr(self, 'beta_full'):
            k_omega, k0, dk_domega = self._compute_full_dispersion_curve()
        else:
            k_omega = self.beta_full
            k0 = self.beta0
            dk_domega = self.dk_domega
        
        # 中心频率处的dk/dω（群速度的倒数）
        k_prime_0 = dk_domega[self.Nt // 2]
        
        # 完整色散算符（相对于移动参考系）
        # D(ω) = i[k(ω) - k₀ - k'₀·Δω] - α/2
        D = 1j * (k_omega - k0 - k_prime_0 * self.omega)
        
        # 添加损耗
        D -= self.alpha / 2
        
        return D
    
    def _compute_group_velocity(self):
        """计算群速度 β₁ = dβ/dω 在中心频率处"""
        # 使用有限差分计算导数
        domega = self.omega0 * 1e-6
        
        # 计算中心频率附近的β值
        lambda_plus = 2 * np.pi * self.c / (self.omega0 + domega)
        lambda_minus = 2 * np.pi * self.c / (self.omega0 - domega)
        
        n_plus = self._compute_refractive_index(lambda_plus)
        n_minus = self._compute_refractive_index(lambda_minus)
        
        beta_plus = n_plus * (self.omega0 + domega) / self.c
        beta_minus = n_minus * (self.omega0 - domega) / self.c
        
        beta1 = (beta_plus - beta_minus) / (2 * domega)
        return beta1
    
    def _precompute_raman_response(self):
        """预计算拉曼响应函数 (频域)"""
        # 石英光纤的标准拉曼参数
        self.fr = 0.18         # 拉曼分数 (Fractional Raman contribution)
        tau1 = 12.2e-15        # 声子振荡周期参数 (s)
        tau2 = 32.0e-15        # 声子衰减寿命参数 (s)
        
        # 1. 在时域构建拉曼响应函数 hR(t)
        # 注意：必须确保时间轴对齐 FFT 的逻辑 (t=0 在数组开头)
        # 我们重新生成一个从 0 开始的时间轴用于计算响应函数
        t_response = np.linspace(0, self.T, self.Nt, endpoint=False)
        
        # hR(t) = (tau1^2 + tau2^2)/(tau1*tau2^2) * exp(-t/tau2) * sin(t/tau1)
        hR = (tau1**2 + tau2**2)/(tau1 * tau2**2) * np.exp(-t_response/tau2) * np.sin(t_response/tau1)
        
        # 2. 归一化 (保证积分面积为1，能量守恒)
        # 离散积分：sum(hR) * dt = 1
        norm_factor = np.sum(hR) * self.dt
        if norm_factor > 0:
            hR /= norm_factor
            
        # 3. 转换到频域并存储
        # 这里的 FFT 结果将用于循环中的卷积计算
        # 根据卷积定理：FFT(f * g) = FFT(f) · FFT(g) · dt (视具体FFT实现定义的归一化系数而定)
        # Scipy 的 FFT 没有包含 dt，所以我们需要在卷积时乘以 dt
        self.Hr_freq = fft(hR) * self.dt
        
        print(f"  ⚡ 拉曼响应函数已预计算 (f_R={self.fr})")
        
    def set_parameters(self, n2, n0=1.45, beta2=36e-27, beta3=0.0, alpha=0, 
                     self_steepening=False, raman_response=False):
        """设置晶体物理参数"""
        self.n2 = n2
        self.n0 = n0
        self.alpha = alpha
        
        # 根据色散模式设置色散系数
        if self.dispersion_mode == 'sellmeier':
            # 使用真实色散曲线计算β₂, β₃
            self._compute_dispersion_from_sellmeier()
            print(f"使用真实色散曲线: β₂ = {self.beta2:.2e} s²/m, β₃ = {self.beta3:.2e} s³/m")
        else:
            # 使用用户提供的β₂, β₃
            self.beta2 = beta2
            self.beta3 = beta3
            print(f"使用泰勒展开: β₂ = {self.beta2:.2e} s²/m, β₃ = {self.beta3:.2e} s³/m")
        
        # 非线性系数（物理正确版本）
        # 在非线性薴定谔方程 (NLSE) 中：
        # ∂A/∂z = iγ|A|²A + 色散项
        # 
        # 其中 A 的单位是 sqrt(W)，γ 的单位是 W⁻¹m⁻¹
        # 
        # 标准定义：γ = 2π·n₂/(λ₀·A_eff) 或等价地 γ = n₂·ω₀/(c·A_eff)
        # 单位验证：
        #   [n₂] = m²/W (非线性折射率系数)
        #   [2π/λ₀] = m⁻¹ (波数)
        #   [A_eff] = m² (有效面积)
        #   ∴ [γ] = [m²/W] / ([m]·[m²]) = W⁻¹m⁻¹ ✓
        if n2 != 0:
            # 使用标准公式：γ = 2πn₂/(λ₀·A_eff)
            self.gamma = (2 * np.pi / self.lambda0) * (n2 / self.beam_area)
        else:
            self.gamma = 0
        
        # 高阶效应
        self.self_steepening = self_steepening
        self.raman_response = raman_response
        
        # 【新增】如果启用了拉曼，预计算响应函数
        if self.raman_response:
            self._precompute_raman_response()
            
        print(f"\n{'='*60}")
        print(f"参数设置: λ={self.lambda0*1e9:.1f}nm, n₂={n2:.2e}, γ={self.gamma:.6e}")
        print(f"  光束面积: {self.beam_area:.6e} m²")
        print(f"  非线性长度 L_NL = 1/(γ·P0) ~ {1/(self.gamma*1e3) if self.gamma > 0 else float('inf'):.3f} mm @ 1kW")
        print(f"{'='*60}")
    
    def _compute_dispersion_operator(self):
        """计算色散算符
        
        根据色散模式选择不同的实现方式：
        - 'sellmeier': 使用完整的Sellmeier色散曲线
        - 'taylor': 使用β₂、β₃泰勒展开近似
        
        Returns:
            色散算符数组
        """
        if self.dispersion_mode == 'sellmeier':
            # 使用完整的Sellmeier色散曲线（修复了属性名错误）
            D = self._compute_full_dispersion_operator()
            print("  ✓ 使用完整Sellmeier色散曲线")
        else:
            # 标准NLSE色散算符（泰勒展开）
            # 物理正确的形式：D(ω) = i·β₂/2·ω² + i·β₃/6·ω³ - α/2
            # 其中 β₂ > 0 对应正常色散，β₂ < 0 对应反常色散
            D = 0.5j * self.beta2 * self.omega**2 + (1j * self.beta3 / 6) * self.omega**3
            D -= self.alpha / 2
            print("  ✓ 使用泰勒展开色散近似")
        
        return D
    
    def _nonlinear_operator(self, A):
        """计算非线性算符 (含拉曼效应修正版)
        
        Args:
            A: 复数场幅 (sqrt(W))
        Returns:
            非线性算符 N
        """
        # 检查输入有效性
        if np.any(np.isnan(A)) or np.any(np.isinf(A)):
            raise ValueError("输入场包含NaN或Inf值，请检查输入参数")
        
        # 1. 计算瞬时功率 I(t)
        power = np.abs(A)**2  # W
        
        # 2. 计算有效非线性响应项 (Effective Nonlinear Response)
        # 如果启用了拉曼，非线性项不再只是 |A|^2
        # 而是 (1-fr)*|A|^2 + fr*(hR ⊗ |A|^2)
        
        if self.raman_response and hasattr(self, 'Hr_freq'):
            # --- 拉曼卷积计算 (利用卷积定理加速) ---
            # 步骤: I(t) -> FFT -> 乘Hr(ω) -> IFFT -> I_raman(t)
            power_freq = fft(power)
            
            # 频域乘法 (对应时域卷积)
            # 注意：self.Hr_freq 在预计算时已经包含了 dt 因子
            conv_result = ifft(power_freq * self.Hr_freq)
            
            # 组合瞬时项和延迟项
            # effective_power = (1 - f_R) * I(t) + f_R * (h_R ⊗ I(t))
            nonlinear_term = (1 - self.fr) * power + self.fr * conv_result
        else:
            # 无拉曼效应，全是瞬时响应
            nonlinear_term = power

        # 3. 构造基本非线性算符 N
        # N = i * gamma * (有效响应项) * A
        N = 1j * self.gamma * nonlinear_term * A
        
        # 4. 自陡峭效应 (Self-Steepening)
        # GNLSE 中的自陡峭项是对整个非线性极化做微分: (1 + i/w0 * d/dt) (P_NL)
        if self.self_steepening:
            try:
                # P_NL = nonlinear_term * A
                # 我们需要计算 d/dt (P_NL)
                # 在频域做微分: d/dt -> i*omega
                
                P_nl_freq = fft(nonlinear_term * A)
                d_Pnl_dt = ifft(1j * self.omega * P_nl_freq)
                
                # 叠加到 N 上: i*gamma * (i/w0 * dP/dt)
                # 注意前面的 1j*gamma 已经在 N 里算了一部分，这里补上导数项
                # 公式: i*gamma * (1 + i/w0*dt) * P_nl = i*gamma*P_nl - gamma/w0 * dP_nl/dt
                # N 已经是 i*gamma*P_nl 了，所以这里加的是第二项
                
                N += (1j / self.omega0) * self.gamma * d_Pnl_dt
                
            except Exception as e:
                print(f"⚠️  自陡峭效应计算失败: {e}")
        
        # 最终数值检查
        if np.any(np.isnan(N)) or np.any(np.isinf(N)):
            raise ValueError("非线性算符计算产生NaN或Inf值，请减小步长或功率")
        
        return N
    
    def propagate(self, A_initial, verbose=True, track_phase_evolution=False):
        """使用RK4IP方法传播脉冲
        
        Args:
            A_initial: 初始脉冲
            verbose: 是否显示进度信息
            track_phase_evolution: 是否跟踪相位积累演化
        """
        A = A_initial.copy()
        A_evolution = np.zeros((self.Nz, self.Nt), dtype=complex)
        A_evolution[0, :] = A
        
        # 初始化相位积累演化数组（如果启用跟踪）
        if track_phase_evolution:
            self.phase_evolution = np.zeros((self.Nz, self.Nt), dtype=float)
            self.phase_evolution[0, :] = 0  # 初始相位积累为0
        
        D = self._compute_dispersion_operator()
        exp_D_half = np.exp(D * self.dz / 2)
        
        if verbose:
            print(f"\n开始RK4IP仿真...")
            if track_phase_evolution:
                print("启用相位积累演化跟踪")
        
        # 计算初始能量（在循环外）
        E0 = self._compute_energy(A)
        
        for i in range(1, self.Nz):
            # RK4IP步骤
            # 第一步：前半色散步
            A_freq = fft(A)
            A_freq *= exp_D_half
            A = ifft(A_freq)
            
            # 第二步：纯时域RK4非线性步（无色散）
            k1 = self._nonlinear_operator(A)
            k2 = self._nonlinear_operator(A + 0.5 * self.dz * k1)
            k3 = self._nonlinear_operator(A + 0.5 * self.dz * k2)
            k4 = self._nonlinear_operator(A + self.dz * k3)
            
            # 组合非线性项（时域相加）
            A += (self.dz / 6) * (k1 + 2*k2 + 2*k3 + k4)
            
            # 第三步：后半色散步
            A_freq = fft(A)
            A_freq *= exp_D_half
            A = ifft(A_freq)
            
            A_evolution[i, :] = A
            
            # 计算当前步的相位积累（如果启用跟踪）
            if track_phase_evolution:
                self._compute_step_phase_accumulation(A_evolution, i)
            
            if verbose and (i % (self.Nz // 10) == 0 or i == self.Nz - 1):
                progress = i / (self.Nz - 1) * 100
                E_ratio = self._compute_energy(A) / E0 * 100
                print(f"  进度: {progress:5.1f}% | 能量: {E_ratio:6.2f}%")
        
        # 计算最终相位积累
        self._compute_phase_accumulation(A_evolution)
        
        if verbose:
            print("仿真完成！\n")
        
        return A_evolution
    
    def _compute_step_phase_accumulation(self, A_evolution, step_index):
        """计算每一步的相位积累（相对于输入脉冲）"""
        A_freq_in = fft(A_evolution[0, :])
        A_freq_current = fft(A_evolution[step_index, :])
        
        phase_in = np.angle(A_freq_in)
        phase_current = np.angle(A_freq_current)
        phase_diff = np.unwrap(phase_current - phase_in)
        
        self.phase_evolution[step_index, :] = phase_diff
    
    def _compute_phase_accumulation(self, A_evolution):
        """计算相位积累"""
        A_freq_in = fft(A_evolution[0, :])
        A_freq_out = fft(A_evolution[-1, :])
        
        phase_in = np.angle(A_freq_in)
        phase_out = np.angle(A_freq_out)
        phase_diff = np.unwrap(phase_out - phase_in)
        
        self.phase_accumulation = phase_diff
    
    def gaussian_pulse(self, pulse_energy, pulse_width_fwhm, GDD=0, TOD=0, target_chirped_width=None):
        """生成高斯脉冲 (支持直接指定目标展宽宽度)
        
        Args:
            pulse_energy: 脉冲能量 (J)
            pulse_width_fwhm: 变换受限脉冲宽度 (s) -> 对应光谱宽度 (此处填 200e-15)
            GDD: 额外的群延迟色散 (s^2)
            TOD: 三阶色散 (s^3)
            target_chirped_width: [新增] 目标时域宽度 (s) -> (此处填 10e-12)
                                  如果设置此值，函数会自动计算所需的额外 GDD
            
        Returns:
            A: 复数场幅 sqrt(W)
        """
        # --- 新增逻辑：根据目标宽度自动计算 GDD ---
        if target_chirped_width is not None:
            if target_chirped_width < pulse_width_fwhm:
                print("⚠️ 警告: 目标宽度小于变换极限宽度，忽略 target_chirped_width")
            else:
                # 基于高斯脉冲展宽公式反推所需的 GDD
                # tau_out = tau_in * sqrt(1 + (4*ln2*GDD / tau_in^2)^2)
                # 推导: GDD = (tau_in^2 / (4*ln2)) * sqrt((tau_out/tau_in)^2 - 1)
                
                tau_in = pulse_width_fwhm
                tau_out = target_chirped_width
                factor = 4 * np.log(2)
                
                # 计算所需的总 GDD 绝对值
                required_GDD = (tau_in**2 / factor) * np.sqrt((tau_out/tau_in)**2 - 1)
                
                # 默认施加正色散 (类似展宽器)，叠加到用户输入的 GDD 上
                print(f"  ⚡ 自动计算展宽 GDD: {required_GDD*1e30:.1f} fs² (由 {tau_in*1e15:.0f}fs -> {tau_out*1e12:.1f}ps)")
                GDD = GDD + required_GDD

        # 1. 计算变换受限脉冲的参数 (由 pulse_width_fwhm 决定光谱宽度)
        # 峰值功率 P0 (由变换受限宽度决定)
        P0 = pulse_energy / (pulse_width_fwhm * np.sqrt(np.pi/(4*np.log(2))))
        
        # 1/e 宽度 T0
        T0 = pulse_width_fwhm / (2*np.sqrt(np.log(2)))
        
        # 2. 生成初始时域场 (纯实数，无相位，变换受限)
        # 此时拥有 200fs 的光谱和 200fs 的脉宽
        A_tl = np.sqrt(P0) * np.exp(-(self.t/T0)**2 / 2)
        
        # 如果没有色散，直接返回变换受限脉冲
        if GDD == 0 and TOD == 0:
            return A_tl
            
        # 3. 转换到频域应用色散
        A_freq = fft(A_tl)
        
        # 4. 计算色散相位因子
        # spectral phase Φ(ω) = (1/2)*GDD*ω^2 + (1/6)*TOD*ω^3
        dispersion_phase = 0.5 * GDD * self.omega**2 + (1.0/6.0) * TOD * self.omega**3
        
        # 应用相位
        phase_factor = np.exp(1j * dispersion_phase)
        A_dispersed_freq = A_freq * phase_factor
        
        # 5. 转换回时域 (此时脉宽变为 10ps，但光谱仍对应 200fs)
        A_final = ifft(A_dispersed_freq)
        
        # 6. 打印展宽信息
        current_fwhm = self._compute_fwhm(np.abs(A_final)**2, self.t)
        stretch_ratio = current_fwhm / pulse_width_fwhm
        
        print(f"  ℹ️  [色散加载] 已应用频域色散:")
        print(f"      GDD = {GDD*1e30:.0f} fs²")
        if abs(TOD) > 0:
            print(f"      TOD = {TOD*1e45:.0f} fs³")
        print(f"      光谱对应宽度 (TL): {pulse_width_fwhm*1e12:.3f} ps")
        print(f"      实际输出宽度 (Chirped): {current_fwhm*1e12:.3f} ps (展宽因子: {stretch_ratio:.2f})")
        print(f"      峰值功率降低因子: {np.max(np.abs(A_tl)**2) / np.max(np.abs(A_final)**2):.2f}")
        
        return A_final
    
    def _compute_energy(self, A):
        """计算脉冲能量
        
        Args:
            A: 复数场幅 (sqrt(W))
            
        Returns:
            总能量 (J)
        """
        power = np.abs(A)**2  # W
        energy = np.trapz(power, self.t)  # J
        return energy
    
    def pulse_from_spectrum_data(self, wavelengths_nm, intensities, pulse_energy):
        """
        从光谱数据直接重建初始脉冲（修正版：频域直接构建法）。
        
        Args:
            wavelengths_nm (np.array): 波长数据 (nm)
            intensities (np.array): 光谱强度数据
            pulse_energy (float): 目标脉冲能量 (J)
            
        Returns:
            np.array: 时域电场 A(t)
        """
        print("  从光谱数据重建初始脉冲 (频域直接构建法)...")
        
        # 1. 数据预处理
        # 确保数据按波长升序排列
        sort_idx = np.argsort(wavelengths_nm)
        wl_sorted = wavelengths_nm[sort_idx]
        int_sorted = intensities[sort_idx]
        
        # 转换为频率 (Hz)
        # 注意：波长从小到大 -> 频率从大到小，需要再次翻转以配合 interp1d
        input_freqs = self.c / (wl_sorted * 1e-9)
        
        # 2. 构建仿真器的频率网格
        # self.omega 是角频率偏移 (rad/s)，我们需要绝对频率 (Hz)
        # freq_grid_hz = (omega0 + omega) / 2pi
        sim_abs_freqs = (self.omega0 + self.omega) / (2 * np.pi)
        
        # 3. 插值
        # 将输入光谱插值到仿真的频率网格上
        # 注意：input_freqs 目前是从大到小，interp1d 需要 x 单调递增
        f_interp = interp1d(input_freqs[::-1], int_sorted[::-1], 
                           bounds_error=False, fill_value=0, kind='linear')
        
        target_spectrum_density = f_interp(sim_abs_freqs)
        
        # 消除可能的负值
        target_spectrum_density[target_spectrum_density < 0] = 0
        
        # 4. 构建频域电场 A(ω)
        # 强度 I(ω) ~ |A(ω)|²  =>  |A(ω)| = sqrt(I(ω))
        # 假设为变换受限脉冲（平坦相位），即相位为 0
        A_freq_magnitude = np.sqrt(target_spectrum_density)
        
        # 5. 逆傅里叶变换得到时域场
        # 【关键修改】这里必须加 fftshift，将 t=0 移到数组中心
        # 注意：scipy.fftpack.ifft 期望的输入顺序与 self.omega 的顺序一致
        # (0, positive freqs, negative freqs)，我们上面的计算已经保持了这个顺序
        A_initial = fftshift(ifft(A_freq_magnitude))
        
        # 6. 能量归一化
        current_energy = self._compute_energy(A_initial)
        if current_energy > 0:
            scaling_factor = np.sqrt(pulse_energy / current_energy)
            A_initial = A_initial * scaling_factor
        else:
            raise ValueError("重建的脉冲能量为0，请检查输入光谱波长范围是否覆盖了中心波长")
            
        # 计算并打印结果参数
        bw_nm = wavelengths_nm.max() - wavelengths_nm.min()
        pulse_width = self._compute_fwhm(np.abs(A_initial)**2, self.t)
        
        print(f"    输入数据范围: {wavelengths_nm.min():.2f} - {wavelengths_nm.max():.2f} nm")
        print(f"    中心波长设定: {self.lambda0*1e9:.2f} nm")
        print(f"    重建脉冲宽度: {pulse_width*1e12:.2f} ps")
        print(f"  ✅ 脉冲重建完成")
        
        return A_initial
    
    @staticmethod
    def _compute_fwhm(power, time):
        """计算半高全宽"""
        max_p = np.max(power)
        indices = np.where(power >= max_p/2)[0]
        if len(indices) > 1:
            return time[indices[-1]] - time[indices[0]]
        return 0
