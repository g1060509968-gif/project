import numpy as np
import cupy as cp
from cupyx.scipy.fft import fft, ifft

# =============================================================================
# CUDA C Kernel: 溢出安全的全场等离子体 + Drude 电流引擎
# 输入 float32 (E_field, W_ion, W_ava 各自不溢出)
# 输出 float64 (J_out, rho_out — 避免 w_pi * rho_nt 和 ρ*E 溢出 float32)
# 内部计算全部 double
# =============================================================================
PLASMA_KERNEL_CODE = '''
extern "C" __global__
void plasma_integration_kernel(
    const float* E_field,
    const float* W_ion,
    const float* W_ava,
    double* J_out,
    double* rho_out,
    const float rate_rec,
    const float dt,
    const float rho_nt,
    const float Ui,
    const float e_charge,
    const float m_e,
    const float tau_c,
    const int Nt,
    const int num_spatial_pts
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= num_spatial_pts) return;

    int base_offset = idx * Nt;

    double current_rho = 0.0;
    double current_J_free = 0.0;

    double rho_nt_d = (double)rho_nt;
    double rate_rec_d = (double)rate_rec;
    double dt_d = (double)dt;
    double Ui_d = (double)Ui;
    double e_d = (double)e_charge;
    double m_d = (double)m_e;
    double tau_c_d = (double)tau_c;
    double nu_c = 1.0 / tau_c_d;
    double e2_over_m = e_d * e_d / m_d;

    for (int t = 0; t < Nt; ++t) {
        int flat_idx = base_offset + t;

        // 读取 float32 输入
        double E = (double)E_field[flat_idx];
        double w_pi = (double)W_ion[flat_idx];
        double w_av = (double)W_ava[flat_idx];

        // 中性分子耗尽
        double depletion = (rho_nt_d - current_rho) / rho_nt_d;
        if (depletion < 0.0) depletion = 0.0;

        // -----------------------------------------------------
        // 1. 等离子体密度演化 (精确指数积分)
        // -----------------------------------------------------
        double S_eff = w_pi * rho_nt_d * depletion;
        double Gamma_eff = w_av * depletion - rate_rec_d;

        double gdt = Gamma_eff * dt_d;
        if (gdt > 10.0) gdt = 10.0;
        if (gdt < -10.0) gdt = -10.0;

        double exp_gdt = exp(gdt);
        double step_factor;
        if (fabs(gdt) < 1e-5) {
            step_factor = dt_d * (1.0 + 0.5 * gdt);
        } else {
            step_factor = (exp_gdt - 1.0) / Gamma_eff;
        }

        double next_rho = current_rho * exp_gdt + S_eff * step_factor;
        if (next_rho > rho_nt_d) next_rho = rho_nt_d;
        if (next_rho < 0.0) next_rho = 0.0;

        // 总电子生成率 (光电离 + 雪崩电离)
        double true_gen_rate = S_eff + w_av * depletion * current_rho;

        // -----------------------------------------------------
        // 2. 自由电子 Drude 电流 J_free 演化
        //    dJ/dt = -ν_c J + (e²/m_e) ρ E
        // -----------------------------------------------------
        double J_drive = e2_over_m * current_rho * E;
        double nu_dt = nu_c * dt_d;
        double exp_nu = exp(-nu_dt);
        double step_J;
        if (fabs(nu_dt) < 1e-5) {
            step_J = dt_d * (1.0 - 0.5 * nu_dt);
        } else {
            step_J = (1.0 - exp_nu) / nu_c;
        }
        double next_J_free = current_J_free * exp_nu + J_drive * step_J;

        // -----------------------------------------------------
        // 3. 电离损耗电流 J_loss (包含光电离 + 雪崩电离)
        //    能量守恒: J_loss · E = true_gen_rate · Ui
        // -----------------------------------------------------
        double J_loss = 0.0;
        double E_mag = fabs(E);
        if (E_mag > 1e-10) {
            J_loss = (true_gen_rate * Ui_d * E) / (E * E + 1e-20);
        }

        // 写入 float64 输出
        J_out[flat_idx] = current_J_free + J_loss;
        rho_out[flat_idx] = current_rho;

        // 状态更新
        current_rho = next_rho;
        current_J_free = next_J_free;
    }
}
'''


class OpticalMedium:
    """全场(Real-field)光学介质基类 (PPT 电离 + Drude 等离子体)"""
    def __init__(self, center_wavelength, precision='single'):
        self.lambda0 = center_wavelength
        self.c = 299792458.0
        self.omega0 = 2 * np.pi * self.c / self.lambda0

        # 物理常数
        self.e_charge = 1.60217663e-19
        self.eps0 = 8.85418781e-12
        self.m_e = 9.1093837e-31
        self.hbar = 1.054571817e-34

        # 介质特性 (由子类覆盖)
        self.n0 = 1.0
        self.n2 = 0.0
        self.n4 = 0.0    # 高阶克尔 HOKE (m⁴/W²), 0 = 禁用
        self.chi3 = 0.0
        self.chi5 = 0.0  # 五阶极化率 (m⁴/V⁴), 从 n₄ 导出
        self.Ui = 0.0
        self.rho_nt = 0.0
        self.tau_c = 1e-15
        self.tau_rec = 1e-9

        # PPT 量子参数 (Z: 剩余电荷, l: 轨道角量子数, m: 磁量子数)
        self.Z = 1.0
        self.l = 0
        self.m = 0

        if precision == 'double':
            self.float_dtype = cp.float64
            self.complex_dtype = cp.complex128
        else:
            self.float_dtype = cp.float32
            self.complex_dtype = cp.complex64

        # 编译溢出安全 CUDA Kernel (统一 float32 入 → double 出)
        self._compile_plasma_kernel()

    def _compile_plasma_kernel(self):
        """编译 CUDA Kernel (固定签名: float ins, double outs)"""
        self.plasma_kernel = cp.RawKernel(PLASMA_KERNEL_CODE, 'plasma_integration_kernel')

    def refractive_index(self, omega_array):
        raise NotImplementedError

    def _init_ionization_lut(self, I_min_W_cm2=1e8, I_max_W_cm2=1e16, points=2000):
        """初始化 PPT 电离率查表"""
        I_min = I_min_W_cm2 * 1e4
        I_max = I_max_W_cm2 * 1e4
        self.log10_I_cpu = np.linspace(np.log10(I_min), np.log10(I_max), points)
        self.I_grid_cpu = 10**self.log10_I_cpu
        W_cpu = self.calculate_ppt_rate_cpu(self.I_grid_cpu)
        self.log10_I_gpu = cp.asarray(self.log10_I_cpu, dtype=self.float_dtype)
        self.W_gpu = cp.asarray(W_cpu, dtype=self.float_dtype)

    def get_ionization_rate(self, I_equiv):
        """GPU 上的电离率查表插值 (输入为等效光强 W/m^2)"""
        log_I = cp.log10(I_equiv + 1e-30)
        log_I = cp.clip(log_I, float(self.log10_I_gpu[0]), float(self.log10_I_gpu[-1]))
        log_I_flat = log_I.ravel()
        W_flat = cp.interp(log_I_flat, self.log10_I_gpu, self.W_gpu)
        return W_flat.reshape(log_I.shape)

    def calculate_ppt_rate_cpu(self, I_grid):
        """
        PPT (Perelomov-Popov-Terent'ev) 电离率模型
        输入: I_grid — 等效光强 (W/m^2)
        输出: 电离率 W_PPT (s^-1)
        """
        from scipy.special import gamma

        I_safe = np.maximum(I_grid, 1e4)

        # 1. SI → 原子单位
        E_SI = np.sqrt(2 * I_safe / (self.c * self.eps0 * self.n0))
        E_au = E_SI / 5.142206538e11
        omega_au = self.omega0 * 2.418884e-17
        Ui_au = self.Ui / (27.21138 * self.e_charge)

        Z = self.Z
        l = self.l
        m = self.m

        # 2. Keldysh 参数
        gamma_k = omega_au * np.sqrt(2 * Ui_au) / E_au

        # 3. 有效主量子数 n* 与渐近系数 C_nl
        n_star = Z / np.sqrt(2 * Ui_au)
        C_nl_sq = (2 ** (2 * n_star)) / (n_star * gamma(n_star + l + 1) * gamma(n_star - l))

        # 4. 角因子 f(l, m)
        f_lm = (2 * l + 1) * 0.5 * gamma(l + np.abs(m) + 1) / \
               ((2 ** np.abs(m)) * gamma(np.abs(m) + 1) * gamma(l - np.abs(m) + 1))

        # 5. α 参数
        alpha = 2 * (gamma_k ** 2) / (1 + gamma_k ** 2)

        # 6. g(γ) 指数修正函数
        g_gamma = 3 / (2 * gamma_k) * ((1 + 1 / (2 * gamma_k ** 2)) * np.arcsinh(gamma_k)
                                       - np.sqrt(1 + gamma_k ** 2) / (2 * gamma_k))

        # 7. 指数项
        exponent = -(2 * (2 * Ui_au) ** 1.5) / (3 * E_au) * g_gamma

        # 8. 前置因子
        prefactor = Ui_au * C_nl_sq * f_lm * np.sqrt(6 / np.pi) * \
                    ((2 * (2 * Ui_au) ** 1.5) / E_au) ** (2 * n_star - np.abs(m) - 1.5) * \
                    (1 + gamma_k ** 2) ** (np.abs(m) / 2 + 0.75 - n_star)

        # 9. A(ω, γ) ATI 多通道求和 (15 通道)
        nu = (Ui_au / omega_au) * (1 + 1 / (2 * gamma_k ** 2))
        K_min = np.ceil(nu)

        A_sum = np.zeros_like(E_au)
        for k in range(15):
            K_index = K_min + k
            x = 2 * (K_index - nu) * np.sqrt(1 + gamma_k ** 2) / (gamma_k ** 2 + 1e-20)
            A_sum += np.exp(-alpha * x)

        # 10. a.u. → SI
        W_au = prefactor * A_sum * np.exp(exponent)
        W_SI = W_au / 2.418884e-17

        return W_SI

    def get_avalanche_rate(self, I_equiv):
        """雪崩电离速率 (1/s)"""
        sigma_drude_env = (self.e_charge ** 2 * self.tau_c) / \
                          (self.c * self.eps0 * self.n0 * self.m_e
                           * (1.0 + self.omega0 ** 2 * self.tau_c ** 2))
        return (sigma_drude_env / self.Ui) * I_equiv

    def get_raman_intensity(self, E_t):
        """拉曼卷积激发项 (由子类实现)"""
        return E_t ** 2

    def calc_nonlinear_response(self, E_t, dt):
        """
        全场非线性响应计算。

        溢出安全设计:
        - J_out, rho_out 强制 float64
        - CUDA Kernel 内部全 double 计算
        - 最终输出转回 solver 精度
        """
        # 1. 等效 CW 光强 (用于电离模型输入)
        I_equiv = 0.5 * self.eps0 * self.c * self.n0 * E_t ** 2

        # 2. 瞬态物理速率 (float32, 单值不溢出)
        E_t = cp.ascontiguousarray(E_t)
        W_ion = cp.ascontiguousarray(
            self.get_ionization_rate(I_equiv).astype(cp.float32))
        W_ava = cp.ascontiguousarray(
            self.get_avalanche_rate(I_equiv).astype(cp.float32))

        # 3. 输出数组强制 float64
        J_out = cp.zeros_like(E_t, dtype=cp.float64)
        rho_out = cp.zeros_like(E_t, dtype=cp.float64)

        # 4. 调用溢出安全 CUDA Kernel
        Nt = E_t.shape[-1]
        num_spatial_pts = E_t.size // Nt
        threads_per_block = 256
        blocks_per_grid = (num_spatial_pts + threads_per_block - 1) // threads_per_block

        self.plasma_kernel(
            (blocks_per_grid,), (threads_per_block,),
            (E_t, W_ion, W_ava, J_out, rho_out,
             cp.float32(1.0 / self.tau_rec),
             cp.float32(dt),
             cp.float32(self.rho_nt),
             cp.float32(self.Ui),
             cp.float32(self.e_charge),
             cp.float32(self.m_e),
             cp.float32(self.tau_c),
             cp.int32(Nt), cp.int32(num_spatial_pts))
        )

        # 5. 非线性极化 (Kerr + HOKE + 拉曼)
        # 分步乘法避免 E³/E⁴ 中间值溢出 float32
        chi3_coef = self.eps0 * self.chi3
        if hasattr(self, 'f_R') and self.use_raman:
            E_eff_sq = (1 - self.f_R) * (E_t ** 2) + self.f_R * self.get_raman_intensity(E_t)
            P_NL = chi3_coef * E_t * E_eff_sq
        else:
            P_NL = ((chi3_coef * E_t) * E_t) * E_t
        # 高阶克尔 HOKE (分步避免 E⁴ 溢出)
        if self.chi5 != 0.0:
            chi5_coef = self.eps0 * self.chi5
            E_sq = E_t * E_t
            P_NL = P_NL + ((chi5_coef * E_t) * E_sq) * E_sq

        # 6. 转回 solver 精度类型
        J_out = J_out.astype(self.float_dtype)
        P_NL = P_NL.astype(self.float_dtype)

        return P_NL, J_out


class FusedSilica(OpticalMedium):
    def __init__(self, center_wavelength, precision='single', use_raman=False):
        super().__init__(center_wavelength, precision)

        self.n0 = 1.45
        self.n2 = 3.0e-20
        self.chi3 = (4.0 / 3.0) * self.c * self.eps0 * (self.n0 ** 2) * self.n2

        self.use_raman = use_raman
        self.B = [0.6961663, 0.4079426, 0.8974794]
        self.C = [0.0684043**2, 0.1162414**2, 9.896161**2]
        self.valid_range = (0.2e-6, 6.0e-6)
        self.f_R = 0.18
        self.tau1 = 12.2e-15
        self.tau2 = 32.0e-15
        self.H_raman = None

        self.Ui = 9.0 * self.e_charge
        self.rho_nt = 2.1e28
        self.tau_rec = 150e-15
        self.tau_c = 3.0e-15

        # PPT 等效量子参数
        self.Z = 1.0
        self.l = 0
        self.m = 0

        self._init_ionization_lut()

    def refractive_index(self, omega_array):
        omega_64 = omega_array.astype(cp.float64)
        omega_safe = cp.where(cp.abs(omega_64) < 1e-12, 1e-12, omega_64)
        wavelengths = 2 * np.pi * self.c / cp.abs(omega_safe)
        wavelengths = cp.clip(wavelengths, self.valid_range[0], self.valid_range[1])
        wl_um = wavelengths * 1e6
        wl_sq = wl_um**2
        n_sq = cp.ones_like(wavelengths)
        for B_i, C_i in zip(self.B, self.C):
            n_sq += (B_i * wl_sq) / (wl_sq - C_i)
        return cp.sqrt(n_sq)

    def init_raman(self, Nt, dt):
        if not self.use_raman:
            return
        pad_Nt = 2 * Nt
        t = cp.arange(pad_Nt, dtype=self.float_dtype) * dt
        h_t = (self.tau1**2 + self.tau2**2) / (self.tau1 * self.tau2**2) * \
              cp.exp(-t / self.tau2) * cp.sin(t / self.tau1)
        h_t[0] = 0.0
        H_w = fft(h_t) * dt
        norm = float(cp.abs(H_w[0]))
        if norm > 1e-20:
            H_w /= norm
        self.H_raman = H_w[cp.newaxis, cp.newaxis, :].astype(self.complex_dtype)

    def get_raman_intensity(self, E_t):
        if self.use_raman and self.H_raman is not None:
            Nt = E_t.shape[-1]
            pad_width = [(0, 0)] * (E_t.ndim - 1) + [(0, Nt)]
            E_sq_pad = cp.pad(E_t ** 2, pad_width, mode='constant')

            I_w = fft(E_sq_pad, axis=-1)

            reshape_target = [1] * (E_t.ndim - 1) + [-1]
            H_r = cp.reshape(self.H_raman, reshape_target)

            I_raman_pad = cp.real(ifft(I_w * H_r, axis=-1))
            return I_raman_pad[..., :Nt]
        return E_t ** 2


class Air(OpticalMedium):
    def __init__(self, center_wavelength, pressure_atm=1.0, precision='single'):
        super().__init__(center_wavelength, precision)
        self.pressure = pressure_atm

        self.n0 = 1.0 + 0.00027 * self.pressure
        self.n2 = 3.0e-23 * pressure_atm
        self.chi3 = (4.0 / 3.0) * self.c * self.eps0 * (self.n0 ** 2) * self.n2

        # 电离参数 (以 O₂ 分子为主)
        self.Ui = 12.1 * self.e_charge
        self.rho_nt = 2.7e25 * pressure_atm
        self.tau_rec = 1.0e-9
        self.tau_c = 350e-15 / pressure_atm

        # PPT 等效量子参数 (O₂ 单原子近似)
        self.Z = 1.0
        self.l = 0
        self.m = 0

        self._init_ionization_lut()

    def refractive_index(self, omega_array):
        n_val = self.n0
        return cp.ones_like(omega_array, dtype=cp.float64) * n_val

    def get_avalanche_rate(self, I_equiv):
        """空气在飞秒/阿秒尺度内忽略雪崩电离"""
        return cp.zeros_like(I_equiv, dtype=self.float_dtype)


# =============================================================================
# 稀有气体介质 (实场版, PPT 电离 + Sellmeier/Cauchy 色散)
# =============================================================================
class Helium(OpticalMedium):
    """氦气 He — 最高电离势，非线性最弱"""

    def __init__(self, center_wavelength, pressure_atm=1.0, precision='single'):
        super().__init__(center_wavelength, precision)
        self.pressure = pressure_atm

        # Cauchy 色散系数 (1 atm 基准)
        self.cauchy_A = 3.48e-5
        self.cauchy_B = 5.4e-8
        self.cauchy_C = 1.2e-10

        # 线性/非线性折射率
        self.n0 = 1.0 + self.cauchy_A * pressure_atm  # 可见光中心近似
        self.n2 = 4.0e-25 * pressure_atm
        self.n4 = -1.0e-38 * (pressure_atm ** 2)
        self.chi3 = (4.0 / 3.0) * self.c * self.eps0 * (self.n0 ** 2) * self.n2
        self.chi5 = (4.0 / 5.0) * (self.c ** 2) * (self.eps0 ** 2) * (self.n0 ** 3) * self.n4

        # 电离参数
        self.Ui = 24.59 * self.e_charge
        self.rho_nt = 2.45e25 * pressure_atm
        self.tau_rec = 1.0e-9
        self.tau_c = 500e-15 / pressure_atm

        self.Z = 1.0
        self.l = 0
        self.m = 0

        self._init_ionization_lut()

    def refractive_index(self, omega_array):
        omega_64 = omega_array.astype(cp.float64)
        omega_safe = cp.where(cp.abs(omega_64) < 1e-12, 1e-12, omega_64)
        wl_um = (2 * np.pi * self.c / cp.abs(omega_safe)) * 1e6
        wl_sq = wl_um ** 2
        K = self.cauchy_A + self.cauchy_B / wl_sq + self.cauchy_C / wl_sq ** 2
        return 1.0 + K * (self.pressure / 1.0)

    def get_avalanche_rate(self, I_equiv):
        return cp.zeros_like(I_equiv, dtype=self.float_dtype)


class Neon(OpticalMedium):
    """氖气 Ne"""

    def __init__(self, center_wavelength, pressure_atm=1.0, precision='single'):
        super().__init__(center_wavelength, precision)
        self.pressure = pressure_atm

        self.cauchy_A = 6.66e-5
        self.cauchy_B = 2.4e-8
        self.cauchy_C = 1.7e-10

        self.n0 = 1.0 + self.cauchy_A * pressure_atm
        self.n2 = 2.0e-24 * pressure_atm
        self.n4 = -5.0e-38 * (pressure_atm ** 2)
        self.chi3 = (4.0 / 3.0) * self.c * self.eps0 * (self.n0 ** 2) * self.n2
        self.chi5 = (4.0 / 5.0) * (self.c ** 2) * (self.eps0 ** 2) * (self.n0 ** 3) * self.n4

        self.Ui = 21.56 * self.e_charge
        self.rho_nt = 2.45e25 * pressure_atm
        self.tau_rec = 1.0e-9
        self.tau_c = 350e-15 / pressure_atm

        self.Z = 1.0
        self.l = 0
        self.m = 0

        self._init_ionization_lut()

    def refractive_index(self, omega_array):
        omega_64 = omega_array.astype(cp.float64)
        omega_safe = cp.where(cp.abs(omega_64) < 1e-12, 1e-12, omega_64)
        wl_um = (2 * np.pi * self.c / cp.abs(omega_safe)) * 1e6
        wl_sq = wl_um ** 2
        K = self.cauchy_A + self.cauchy_B / wl_sq + self.cauchy_C / wl_sq ** 2
        return 1.0 + K * (self.pressure / 1.0)

    def get_avalanche_rate(self, I_equiv):
        return cp.zeros_like(I_equiv, dtype=self.float_dtype)


class Argon(OpticalMedium):
    """氩气 Ar"""

    def __init__(self, center_wavelength, pressure_atm=1.0, precision='single'):
        super().__init__(center_wavelength, precision)
        self.pressure = pressure_atm

        self.cauchy_A = 27.9e-5
        self.cauchy_B = 12.8e-8
        self.cauchy_C = 5.2e-10

        self.n0 = 1.0 + self.cauchy_A * pressure_atm
        self.n2 = 1.0e-23 * pressure_atm
        self.n4 = -1.0e-36 * (pressure_atm ** 2)
        self.chi3 = (4.0 / 3.0) * self.c * self.eps0 * (self.n0 ** 2) * self.n2
        self.chi5 = (4.0 / 5.0) * (self.c ** 2) * (self.eps0 ** 2) * (self.n0 ** 3) * self.n4

        self.Ui = 15.76 * self.e_charge
        self.rho_nt = 2.45e25 * pressure_atm
        self.tau_rec = 1.0e-9
        self.tau_c = 190e-15 / pressure_atm

        self.Z = 1.0
        self.l = 0
        self.m = 0

        self._init_ionization_lut()

    def refractive_index(self, omega_array):
        omega_64 = omega_array.astype(cp.float64)
        omega_safe = cp.where(cp.abs(omega_64) < 1e-12, 1e-12, omega_64)
        wl_um = (2 * np.pi * self.c / cp.abs(omega_safe)) * 1e6
        wl_sq = wl_um ** 2
        K = self.cauchy_A + self.cauchy_B / wl_sq + self.cauchy_C / wl_sq ** 2
        return 1.0 + K * (self.pressure / 1.0)

    def get_avalanche_rate(self, I_equiv):
        return cp.zeros_like(I_equiv, dtype=self.float_dtype)


class Krypton(OpticalMedium):
    """氪气 Kr"""

    def __init__(self, center_wavelength, pressure_atm=1.0, precision='single'):
        super().__init__(center_wavelength, precision)
        self.pressure = pressure_atm

        self.cauchy_A = 41.9e-5
        self.cauchy_B = 21.3e-8
        self.cauchy_C = 8.4e-10

        self.n0 = 1.0 + self.cauchy_A * pressure_atm
        self.n2 = 2.8e-23 * pressure_atm
        self.n4 = -2.5e-36 * (pressure_atm ** 2)
        self.chi3 = (4.0 / 3.0) * self.c * self.eps0 * (self.n0 ** 2) * self.n2
        self.chi5 = (4.0 / 5.0) * (self.c ** 2) * (self.eps0 ** 2) * (self.n0 ** 3) * self.n4

        self.Ui = 14.00 * self.e_charge
        self.rho_nt = 2.45e25 * pressure_atm
        self.tau_rec = 1.0e-9
        self.tau_c = 150e-15 / pressure_atm

        self.Z = 1.0
        self.l = 0
        self.m = 0

        self._init_ionization_lut()

    def refractive_index(self, omega_array):
        omega_64 = omega_array.astype(cp.float64)
        omega_safe = cp.where(cp.abs(omega_64) < 1e-12, 1e-12, omega_64)
        wl_um = (2 * np.pi * self.c / cp.abs(omega_safe)) * 1e6
        wl_sq = wl_um ** 2
        K = self.cauchy_A + self.cauchy_B / wl_sq + self.cauchy_C / wl_sq ** 2
        return 1.0 + K * (self.pressure / 1.0)

    def get_avalanche_rate(self, I_equiv):
        return cp.zeros_like(I_equiv, dtype=self.float_dtype)


class Xenon(OpticalMedium):
    """氙气 Xe — 最低电离势，非线性最强"""

    def __init__(self, center_wavelength, pressure_atm=1.0, precision='single'):
        super().__init__(center_wavelength, precision)
        self.pressure = pressure_atm

        self.cauchy_A = 68.7e-5
        self.cauchy_B = 40.6e-8
        self.cauchy_C = 16.3e-10

        self.n0 = 1.0 + self.cauchy_A * pressure_atm
        self.n2 = 6.5e-23 * pressure_atm
        self.n4 = -8.0e-36 * (pressure_atm ** 2)
        self.chi3 = (4.0 / 3.0) * self.c * self.eps0 * (self.n0 ** 2) * self.n2
        self.chi5 = (4.0 / 5.0) * (self.c ** 2) * (self.eps0 ** 2) * (self.n0 ** 3) * self.n4

        self.Ui = 12.13 * self.e_charge
        self.rho_nt = 2.45e25 * pressure_atm
        self.tau_rec = 1.0e-9
        self.tau_c = 120e-15 / pressure_atm

        self.Z = 1.0
        self.l = 0
        self.m = 0

        self._init_ionization_lut()

    def refractive_index(self, omega_array):
        omega_64 = omega_array.astype(cp.float64)
        omega_safe = cp.where(cp.abs(omega_64) < 1e-12, 1e-12, omega_64)
        wl_um = (2 * np.pi * self.c / cp.abs(omega_safe)) * 1e6
        wl_sq = wl_um ** 2
        K = self.cauchy_A + self.cauchy_B / wl_sq + self.cauchy_C / wl_sq ** 2
        return 1.0 + K * (self.pressure / 1.0)

    def get_avalanche_rate(self, I_equiv):
        return cp.zeros_like(I_equiv, dtype=self.float_dtype)
