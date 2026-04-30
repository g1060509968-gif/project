import numpy as np
import cupy as cp
from cupyx.scipy.fft import fft, ifft

# =============================================================================
# CUDA C Kernel: 溢出安全的等离子体引擎
# 输入 float32 (W_ion, W_ava 自身不溢出)
# 输出 float64 (rho, gen_rate_out — 避免 w_pi * rho_nt 溢出 float32)
# 内部计算全部 double
# =============================================================================
PLASMA_KERNEL_CODE = '''
extern "C" __global__
void plasma_integration_kernel(
    const float* W_ion,
    const float* W_ava,
    double* rho,
    double* gen_rate_out,
    const float rate_rec,
    const float dt,
    const float rho_nt,
    const int Nt,
    const int num_spatial_pts
) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx >= num_spatial_pts) return;

    int base_offset = idx * Nt;
    double current_rho = 0.0;
    double rho_nt_d = (double)rho_nt;
    double rate_rec_d = (double)rate_rec;
    double dt_d = (double)dt;

    for (int t = 0; t < Nt; ++t) {
        int flat_idx = base_offset + t;

        rho[flat_idx] = current_rho;

        double w_pi = (double)W_ion[flat_idx];
        double w_av = (double)W_ava[flat_idx];

        double depletion = (rho_nt_d - current_rho) / rho_nt_d;
        if (depletion < 0.0) depletion = 0.0;

        double true_gen_rate = (w_pi * rho_nt_d + w_av * current_rho) * depletion;
        gen_rate_out[flat_idx] = true_gen_rate;

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

        current_rho = current_rho * exp_gdt + S_eff * step_factor;
        if (current_rho > rho_nt_d) current_rho = rho_nt_d;
        if (current_rho < 0.0) current_rho = 0.0;
    }
}
'''


class OpticalMedium:
    """所有光学介质的基类 (Keldysh 电离模型 + 等离子体演化引擎)"""
    def __init__(self, center_wavelength, precision='single', use_shock=False):
        self.lambda0 = center_wavelength
        self.c = 299792458.0
        self.omega0 = 2 * np.pi * self.c / self.lambda0
        self.use_shock = use_shock

        # 物理常数
        self.e_charge = 1.60217663e-19
        self.eps0 = 8.85418781e-12
        self.m_e = 9.1093837e-31
        self.hbar = 1.054571817e-34

        # 介质特性 (由子类覆盖)
        self.n0 = 1.0
        self.n2 = 0.0
        self.n4 = 0.0   # 高阶克尔 HOKE (m⁴/W²), 0 = 禁用
        self.Ui = 0.0
        self.rho_nt = 0.0
        self.tau_c = 0.0
        self.tau_rec = 1e-9
        self.sigma_drude = 0.0
        self.sigma_K = 0.0
        self.K = 1

        if precision == 'double':
            self.float_dtype = cp.float64
            self.complex_dtype = cp.complex128
        else:
            self.float_dtype = cp.float32
            self.complex_dtype = cp.complex64

        # 编译 CUDA 核心 (统一 float32 输入 → double 内部 → float64 输出)
        self._compile_plasma_kernel()

    def _compile_plasma_kernel(self):
        """编译溢出安全的 CUDA Kernel"""
        self.plasma_kernel = cp.RawKernel(PLASMA_KERNEL_CODE, 'plasma_integration_kernel')

    def refractive_index(self, omega_array):
        raise NotImplementedError

    def get_shock_operator(self, omega_array):
        """纯粹的光学自陡峭算子 (1 + w/w0)"""
        if self.use_shock:
            factor = (omega_array + self.omega0) / self.omega0
        else:
            factor = cp.ones_like(omega_array)
        return factor[cp.newaxis, cp.newaxis, :].astype(self.complex_dtype)

    def _init_ionization_lut(self, I_min_W_cm2=1e8, I_max_W_cm2=1e16, points=2000):
        """初始化 Keldysh 电离率查表"""
        I_min = I_min_W_cm2 * 1e4
        I_max = I_max_W_cm2 * 1e4
        self.log10_I_cpu = np.linspace(np.log10(I_min), np.log10(I_max), points)
        self.I_grid_cpu = 10**self.log10_I_cpu
        W_cpu = self.calculate_keldysh_rate_cpu(self.I_grid_cpu)
        self.log10_I_gpu = cp.asarray(self.log10_I_cpu, dtype=self.float_dtype)
        self.W_gpu = cp.asarray(W_cpu, dtype=self.float_dtype)

    def get_ionization_rate(self, Intensity):
        """GPU 上的电离率查表插值"""
        log_I = cp.log10(Intensity + 1e-30)
        log_I = cp.clip(log_I, float(self.log10_I_gpu[0]), float(self.log10_I_gpu[-1]))
        log_I_flat = log_I.ravel()
        W_flat = cp.interp(log_I_flat, self.log10_I_gpu, self.W_gpu)
        return W_flat.reshape(log_I.shape)

    def calculate_keldysh_rate_cpu(self, I_grid):
        """Keldysh-MPI 解析公式 (使用介质的线性折射率 n0)"""
        W_mpi = self.sigma_K * (I_grid ** self.K)
        E_field = np.sqrt(2 * I_grid / (self.c * self.eps0 * self.n0))
        exponent = - (4.0/3.0) * np.sqrt(2 * self.m_e) * (self.Ui**1.5) / \
                   (self.e_charge * self.hbar * E_field + 1e-20)
        W_tunnel = 1e15 * np.exp(exponent)
        return W_mpi + W_tunnel

    def get_effective_intensity(self, Intensity):
        """获取等效光强（子类若有拉曼效应则重写此方法）"""
        return Intensity

    def get_avalanche_rate(self, Intensity):
        """默认的雪崩电离速率 (1/s)"""
        return (self.sigma_drude / self.Ui) * Intensity

    def calc_nonlinear_response(self, A_t, dt):
        """
        核心非线性响应计算。

        溢出安全设计:
        - rho_t, gen_rate_out 强制 float64，避免 w_pi * rho_nt 超出 float32
        - CUDA Kernel 内部以 double 计算
        - 最终输出转回 solver 的精度类型
        """
        Intensity = cp.abs(A_t)**2
        I_eff = self.get_effective_intensity(Intensity)

        # 1. 电离/雪崩速率 (float32 自身安全，单值不溢出)
        W_ion = self.get_ionization_rate(Intensity).astype(cp.float32)
        W_ava = self.get_avalanche_rate(Intensity).astype(cp.float32)

        # 2. 输出数组强制 float64，避免 gen_rate_out ~ 10^47 溢出 float32
        rho_t = cp.zeros_like(Intensity, dtype=cp.float64)
        gen_rate_out = cp.zeros_like(Intensity, dtype=cp.float64)

        # 3. Kernel 启动
        Nt = Intensity.shape[-1]
        num_spatial_pts = Intensity.size // Nt
        threads_per_block = 256
        blocks_per_grid = (num_spatial_pts + threads_per_block - 1) // threads_per_block

        self.plasma_kernel(
            (blocks_per_grid,), (threads_per_block,),
            (W_ion, W_ava, rho_t, gen_rate_out,
             cp.float32(1.0 / self.tau_rec),
             cp.float32(dt),
             cp.float32(self.rho_nt),
             cp.int32(Nt), cp.int32(num_spatial_pts))
        )

        # 4. Kerr + HOKE 项 (保持 solver 精度)
        gamma_kerr = 1j * (self.omega0 / self.c) * self.n2
        Kerr_term = (gamma_kerr * I_eff * A_t).astype(self.complex_dtype)
        if self.n4 != 0.0:
            gamma_hoke = 1j * (self.omega0 / self.c) * self.n4
            Kerr_term = Kerr_term + (gamma_hoke * (I_eff ** 2) * A_t).astype(self.complex_dtype)

        # 5. 等离子体项 (内部 float64 保护，输出转回 solver 精度)
        # gen_rate_out 和 rho_t 均为 float64，避免中间乘积溢出
        Plasma_loss = - (gen_rate_out * self.Ui) / (2.0 * Intensity + 1e-30) * A_t
        Plasma_defocus = - (self.sigma_drude / 2.0) * \
                         (1.0 + 1j * self.omega0 * self.tau_c) * rho_t * A_t
        Plasma_term = (Plasma_loss + Plasma_defocus).astype(self.complex_dtype)

        return Kerr_term, Plasma_term


class FusedSilica(OpticalMedium):
    def __init__(self, center_wavelength, precision='single', use_shock=False, use_raman=False):
        super().__init__(center_wavelength, precision, use_shock)

        self.n0 = 1.45
        self.n2 = 3.0e-20
        self.use_raman = use_raman

        # Sellmeier 色散系数
        self.B = [0.6961663, 0.4079426, 0.8974794]
        self.C = [0.0684043**2, 0.1162414**2, 9.896161**2]
        self.valid_range = (0.2e-6, 6.0e-6)

        # 拉曼参数
        self.f_R = 0.18
        self.tau1 = 12.2e-15
        self.tau2 = 32.0e-15
        self.H_raman = None

        # 电离参数
        self.Ui = 9.0 * self.e_charge
        self.rho_nt = 2.1e28
        self.K = 8
        self.tau_rec = 150e-15
        self.tau_c = 3.0e-15

        self.sigma_drude = (self.e_charge**2 * self.tau_c) / \
                           (self.c * self.eps0 * self.n0 * self.m_e * \
                            (1.0 + self.omega0**2 * self.tau_c**2))
        self.sigma_K = 2.5e-125
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

    def get_effective_intensity(self, Intensity):
        """Zero-padding 拉曼卷积，消除周期边界引入的非因果响应"""
        if self.use_raman and self.H_raman is not None:
            Nt = Intensity.shape[-1]
            pad_width = [(0, 0)] * (Intensity.ndim - 1) + [(0, Nt)]
            I_pad = cp.pad(Intensity, pad_width, mode='constant')

            I_w = fft(I_pad, axis=-1)
            I_raman_pad = cp.real(ifft(I_w * self.H_raman, axis=-1))

            I_raman_t = I_raman_pad[..., :Nt]
            return (1 - self.f_R) * Intensity + self.f_R * I_raman_t
        return Intensity


class Air(OpticalMedium):
    def __init__(self, center_wavelength, pressure_atm=1.0, precision='single', use_shock=False):
        super().__init__(center_wavelength, precision, use_shock)
        self.pressure = pressure_atm

        self.n0 = 1.0 + 0.00027 * self.pressure
        self.n2 = 3.0e-23 * pressure_atm

        # 电离参数 (以 O₂ 分子为主)
        self.Ui = 12.1 * self.e_charge
        self.rho_nt = 2.7e25 * pressure_atm
        self.K = 11
        self.tau_rec = 1.0e-9
        self.tau_c = 350e-15 / pressure_atm

        self.sigma_drude = (self.e_charge**2 * self.tau_c) / \
                           (self.c * self.eps0 * self.n0 * self.m_e * \
                            (1.0 + self.omega0**2 * self.tau_c**2))
        self.sigma_K = 2.0e-183 * pressure_atm
        self._init_ionization_lut()

    def refractive_index(self, omega_array):
        n_val = self.n0
        return cp.ones_like(omega_array, dtype=cp.float64) * n_val

    def get_avalanche_rate(self, Intensity):
        """空气在飞秒尺度内忽略雪崩电离"""
        return cp.zeros_like(Intensity, dtype=self.float_dtype)


# =============================================================================
# 稀有气体介质 (包络版)
# =============================================================================
class Helium(OpticalMedium):
    """氦气 He — 最高电离势，非线性最弱"""

    def __init__(self, center_wavelength, pressure_atm=1.0, precision='single',
                 use_shock=False):
        super().__init__(center_wavelength, precision, use_shock)
        self.pressure = pressure_atm

        # 折射率 Cauchy 色散系数 (1 atm 基准)
        self.cauchy_A = 3.48e-5
        self.cauchy_B = 5.4e-8   # μm²
        self.cauchy_C = 1.2e-10  # μm⁴

        # 非线性折射率 @ 1 atm (P 线性定标)
        self.n2 = 4.0e-25 * pressure_atm
        # 高阶克尔 HOKE @ 1 atm (P² 定标, 负值 = 自散焦)
        self.n4 = -1.0e-38 * (pressure_atm ** 2)

        # 电离参数
        self.Ui = 24.59 * self.e_charge
        self.rho_nt = 2.45e25 * pressure_atm
        self.K = 17
        self.tau_rec = 1.0e-9
        self.tau_c = 500e-15 / pressure_atm

        self.sigma_drude = (self.e_charge ** 2 * self.tau_c) / \
                           (self.c * self.eps0 * 1.0 * self.m_e
                            * (1.0 + self.omega0 ** 2 * self.tau_c ** 2))
        self.sigma_K = 0.0  # 高 K 气体，Keldysh 隧穿项主导
        self._init_ionization_lut()

    def refractive_index(self, omega_array):
        omega_64 = omega_array.astype(cp.float64)
        omega_safe = cp.where(cp.abs(omega_64) < 1e-12, 1e-12, omega_64)
        wl_um = (2 * np.pi * self.c / cp.abs(omega_safe)) * 1e6
        wl_sq = wl_um ** 2
        K = self.cauchy_A + self.cauchy_B / wl_sq + self.cauchy_C / wl_sq ** 2
        return 1.0 + K * (self.pressure / 1.0)

    def get_avalanche_rate(self, Intensity):
        return cp.zeros_like(Intensity, dtype=self.float_dtype)


class Neon(OpticalMedium):
    """氖气 Ne"""

    def __init__(self, center_wavelength, pressure_atm=1.0, precision='single',
                 use_shock=False):
        super().__init__(center_wavelength, precision, use_shock)
        self.pressure = pressure_atm

        self.cauchy_A = 6.66e-5
        self.cauchy_B = 2.4e-8
        self.cauchy_C = 1.7e-10

        self.n2 = 2.0e-24 * pressure_atm
        self.n4 = -5.0e-38 * (pressure_atm ** 2)

        self.Ui = 21.56 * self.e_charge
        self.rho_nt = 2.45e25 * pressure_atm
        self.K = 15
        self.tau_rec = 1.0e-9
        self.tau_c = 350e-15 / pressure_atm

        self.sigma_drude = (self.e_charge ** 2 * self.tau_c) / \
                           (self.c * self.eps0 * 1.0 * self.m_e
                            * (1.0 + self.omega0 ** 2 * self.tau_c ** 2))
        self.sigma_K = 0.0
        self._init_ionization_lut()

    def refractive_index(self, omega_array):
        omega_64 = omega_array.astype(cp.float64)
        omega_safe = cp.where(cp.abs(omega_64) < 1e-12, 1e-12, omega_64)
        wl_um = (2 * np.pi * self.c / cp.abs(omega_safe)) * 1e6
        wl_sq = wl_um ** 2
        K = self.cauchy_A + self.cauchy_B / wl_sq + self.cauchy_C / wl_sq ** 2
        return 1.0 + K * (self.pressure / 1.0)

    def get_avalanche_rate(self, Intensity):
        return cp.zeros_like(Intensity, dtype=self.float_dtype)


class Argon(OpticalMedium):
    """氩气 Ar"""

    def __init__(self, center_wavelength, pressure_atm=1.0, precision='single',
                 use_shock=False):
        super().__init__(center_wavelength, precision, use_shock)
        self.pressure = pressure_atm

        self.cauchy_A = 27.9e-5
        self.cauchy_B = 12.8e-8
        self.cauchy_C = 5.2e-10

        self.n2 = 1.0e-23 * pressure_atm
        self.n4 = -1.0e-36 * (pressure_atm ** 2)

        self.Ui = 15.76 * self.e_charge
        self.rho_nt = 2.45e25 * pressure_atm
        self.K = 11
        self.tau_rec = 1.0e-9
        self.tau_c = 190e-15 / pressure_atm

        self.sigma_drude = (self.e_charge ** 2 * self.tau_c) / \
                           (self.c * self.eps0 * 1.0 * self.m_e
                            * (1.0 + self.omega0 ** 2 * self.tau_c ** 2))
        self.sigma_K = 1.0e-188 * pressure_atm
        self._init_ionization_lut()

    def refractive_index(self, omega_array):
        omega_64 = omega_array.astype(cp.float64)
        omega_safe = cp.where(cp.abs(omega_64) < 1e-12, 1e-12, omega_64)
        wl_um = (2 * np.pi * self.c / cp.abs(omega_safe)) * 1e6
        wl_sq = wl_um ** 2
        K = self.cauchy_A + self.cauchy_B / wl_sq + self.cauchy_C / wl_sq ** 2
        return 1.0 + K * (self.pressure / 1.0)

    def get_avalanche_rate(self, Intensity):
        return cp.zeros_like(Intensity, dtype=self.float_dtype)


class Krypton(OpticalMedium):
    """氪气 Kr"""

    def __init__(self, center_wavelength, pressure_atm=1.0, precision='single',
                 use_shock=False):
        super().__init__(center_wavelength, precision, use_shock)
        self.pressure = pressure_atm

        self.cauchy_A = 41.9e-5
        self.cauchy_B = 21.3e-8
        self.cauchy_C = 8.4e-10

        self.n2 = 2.8e-23 * pressure_atm
        self.n4 = -2.5e-36 * (pressure_atm ** 2)

        self.Ui = 14.00 * self.e_charge
        self.rho_nt = 2.45e25 * pressure_atm
        self.K = 10
        self.tau_rec = 1.0e-9
        self.tau_c = 150e-15 / pressure_atm

        self.sigma_drude = (self.e_charge ** 2 * self.tau_c) / \
                           (self.c * self.eps0 * 1.0 * self.m_e
                            * (1.0 + self.omega0 ** 2 * self.tau_c ** 2))
        self.sigma_K = 1.0e-168 * pressure_atm
        self._init_ionization_lut()

    def refractive_index(self, omega_array):
        omega_64 = omega_array.astype(cp.float64)
        omega_safe = cp.where(cp.abs(omega_64) < 1e-12, 1e-12, omega_64)
        wl_um = (2 * np.pi * self.c / cp.abs(omega_safe)) * 1e6
        wl_sq = wl_um ** 2
        K = self.cauchy_A + self.cauchy_B / wl_sq + self.cauchy_C / wl_sq ** 2
        return 1.0 + K * (self.pressure / 1.0)

    def get_avalanche_rate(self, Intensity):
        return cp.zeros_like(Intensity, dtype=self.float_dtype)


class Xenon(OpticalMedium):
    """氙气 Xe — 最低电离势，非线性最强"""

    def __init__(self, center_wavelength, pressure_atm=1.0, precision='single',
                 use_shock=False):
        super().__init__(center_wavelength, precision, use_shock)
        self.pressure = pressure_atm

        self.cauchy_A = 68.7e-5
        self.cauchy_B = 40.6e-8
        self.cauchy_C = 16.3e-10

        self.n2 = 6.5e-23 * pressure_atm
        self.n4 = -8.0e-36 * (pressure_atm ** 2)

        self.Ui = 12.13 * self.e_charge
        self.rho_nt = 2.45e25 * pressure_atm
        self.K = 8
        self.tau_rec = 1.0e-9
        self.tau_c = 120e-15 / pressure_atm

        self.sigma_drude = (self.e_charge ** 2 * self.tau_c) / \
                           (self.c * self.eps0 * 1.0 * self.m_e
                            * (1.0 + self.omega0 ** 2 * self.tau_c ** 2))
        self.sigma_K = 1.0e-130 * pressure_atm
        self._init_ionization_lut()

    def refractive_index(self, omega_array):
        omega_64 = omega_array.astype(cp.float64)
        omega_safe = cp.where(cp.abs(omega_64) < 1e-12, 1e-12, omega_64)
        wl_um = (2 * np.pi * self.c / cp.abs(omega_safe)) * 1e6
        wl_sq = wl_um ** 2
        K = self.cauchy_A + self.cauchy_B / wl_sq + self.cauchy_C / wl_sq ** 2
        return 1.0 + K * (self.pressure / 1.0)

    def get_avalanche_rate(self, Intensity):
        return cp.zeros_like(Intensity, dtype=self.float_dtype)
