"""
ERK4(3)-IP方法可视化模块
提供各种非线性光学现象的可视化函数
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftfreq, fftshift
import os

# 使用默认字体避免中文显示问题
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False

# 物理常数
c = 299792458.0  # 光速 (m/s)


def freq_to_wavelength(freq_val, center_wavelength=1064e-9, is_angular=False):
    """
    将频率数组转换为波长数组
    
    Parameters:
    -----------
    freq_val : array
        频率数组。如果是 is_angular=True，单位为 rad/s；否则默认为 THz
    center_wavelength : float
        中心波长 (m)
    is_angular : bool
        如果为 True，表示输入是角频率 (rad/s)；否则为频率 (THz)
    
    Returns:
    --------
    wavelength : array
        波长数组 (nm)
    """
    # 物理常数
    c = 299792458.0
    
    if is_angular:
        # 如果输入是 rad/s，转换为 Hz
        # 中心角频率 (rad/s)
        omega0 = 2 * np.pi * c / center_wavelength
        # 绝对角频率 (rad/s)
        omega_abs = omega0 + freq_val
        # 转换为频率 (Hz)
        abs_freq_Hz = omega_abs / (2 * np.pi)
    else:
        # 如果输入是 THz，转换为 Hz
        abs_freq_Hz = (c / center_wavelength) + (freq_val * 1e12)
    
    # 防止除零（处理数值极小的情况）
    abs_freq_Hz[abs_freq_Hz <= 0] = 1e-12
    
    return (c / abs_freq_Hz) * 1e9  # 返回 nm

def wavelength_from_frequency(sim, freq):
    """
    统一的波长-频率转换函数
    使用精确的转换公式而非线性近似
    
    Args:
        sim: 仿真器实例
        freq: 频率偏移数组 (Hz)
        
    Returns:
        wavelength: 波长数组 (nm)
    """
    # 物理公式：lambda = c / f_absolute
    # 绝对频率 = 中心频率 + 频率偏移
    # 这里使用角频率转换： omega_total = omega0 + 2*pi*freq_shift
    omega_total = sim.omega0 + 2 * np.pi * freq
    
    # 防止除以零
    omega_total[omega_total == 0] = 1e-15
    
    wavelength = 2 * np.pi * sim.c / omega_total * 1e9
    return wavelength


def plot_supercontinuum(z_array, A_array, t, freq_shift, spectrum_init, spectrum_final, 
                       save_path=None, dpi=300, center_wavelength=1064e-9):
    """
    绘制超连续谱产生可视化
    
    Parameters:
    -----------
    z_array : array
        传播距离数组
    A_array : array
        场振幅数组
    t : array
        时间数组
    freq_shift : array
        频率数组 (已 fftshift 过，单调递增)
    spectrum_init : array
        初始频谱 (已 fftshift 过)
    spectrum_final : array
        最终频谱 (已 fftshift 过)
    save_path : str, optional
        保存路径
    dpi : int
        图像分辨率
    center_wavelength : float
        中心波长 (m)
    """
    # 修复：假设输入数据已经正确 fftshift 过
    # 直接计算波长，不要再次 fftshift
    wavelength = freq_to_wavelength(freq_shift, center_wavelength)
    
    # 直接使用输入的频谱数据
    spec_init_plot = spectrum_init
    spec_final_plot = spectrum_final
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # 频谱展宽 - 使用波长坐标
    ax = axes[0]
    ax.semilogy(wavelength, spec_init_plot / np.max(spec_init_plot), 
                'b-', label='Initial', linewidth=2)
    ax.semilogy(wavelength, spec_final_plot / np.max(spec_final_plot), 
                'r-', label='Final', linewidth=2)
    ax.set_xlabel('Wavelength (nm)', fontsize=12)
    ax.set_ylabel('Normalized Spectrum (log)', fontsize=12)
    ax.set_title('Supercontinuum Generation', fontsize=14, fontweight='bold')
    # 设置合理的波长范围
    center_wl_nm = center_wavelength * 1e9
    ax.set_xlim([center_wl_nm - 200, center_wl_nm + 200])
    ax.set_ylim([1e-6, 1])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 频谱演化 - 使用波长坐标
    ax = axes[1]
    # 确保有足够的数据点用于 contourf (至少需要3个z位置和2个波长才能有意义的contour图)
    if len(z_array) >= 3 and len(wavelength) >= 2:
        # 检查数据形状
        print(f"Debug: z_array shape = {z_array.shape}, A_array shape = {A_array.shape}")
        print(f"Debug: wavelength shape = {wavelength.shape}")
        
        # 使用所有z位置，不进行下采样
        Z, W = np.meshgrid(z_array*100, wavelength)  # 转换为cm
        spectrogram = np.zeros((len(wavelength), len(z_array)))
        for i, A in enumerate(A_array):
            spec = np.fft.fftshift(np.abs(np.fft.fft(A))**2)
            spectrogram[:, i] = spec / np.max(spec)
        
        print(f"Debug: Z shape = {Z.shape}, W shape = {W.shape}, spectrogram shape = {spectrogram.shape}")
        
        im = ax.contourf(Z, W, spectrogram, levels=50, cmap='jet')
        ax.set_xlabel('Propagation Distance (cm)', fontsize=12)
        ax.set_ylabel('Wavelength (nm)', fontsize=12)
        ax.set_title('Spectral Evolution', fontsize=14, fontweight='bold')
        ax.set_ylim([center_wl_nm - 200, center_wl_nm + 200])
        plt.colorbar(im, ax=ax, label='Normalized Intensity')
    else:
        # 如果数据不足，使用简单的线图显示初始和最终频谱
        ax.semilogy(wavelength, spectrum_init / np.max(spectrum_init), 
                    'b-', label='Initial', linewidth=2)
        ax.semilogy(wavelength, spectrum_final / np.max(spectrum_final), 
                    'r-', label='Final', linewidth=2)
        ax.set_xlabel('Wavelength (nm)', fontsize=12)
        ax.set_ylabel('Normalized Spectrum (log)', fontsize=12)
        ax.set_title('Spectral Evolution (Line Plot)', fontsize=14, fontweight='bold')
        ax.set_xlim([center_wl_nm - 200, center_wl_nm + 200])
        ax.set_ylim([1e-6, 1])
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi)
    
    return fig, axes


def plot_pulse_compression(z_total, w_total, t, A0, A1_final, A2_final, 
                          freq_shift, spec0, spec1, spec2, save_path=None, dpi=300,
                          center_wavelength=1064e-9):
    """
    绘制脉冲压缩可视化
    
    Parameters:
    -----------
    z_total : array
        总传播距离
    w_total : array
        脉宽演化
    t : array
        时间数组
    A0 : array
        初始脉冲
    A1_final : array
        非线性传播后脉冲
    A2_final : array
        压缩后脉冲
    freq_shift : array
        频率数组 (已 fftshift 过，单调递增)
    spec0, spec1, spec2 : array
        不同阶段的频谱 (已 fftshift 过)
    save_path : str, optional
        保存路径
    dpi : int
        图像分辨率
    center_wavelength : float
        中心波长 (m)
    """
    # 修复：假设输入数据已经正确 fftshift 过
    # 直接计算波长，不要再次 fftshift
    wavelength = freq_to_wavelength(freq_shift, center_wavelength)
    
    # 直接使用输入的频谱数据
    s0 = spec0
    s1 = spec1
    s2 = spec2
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 脉冲形状
    ax = axes[0, 0]
    ax.plot(t, np.abs(A0)**2, 'b-', label='Initial', linewidth=2)
    ax.plot(t, np.abs(A1_final)**2, 'g-', label='After Nonlinear', linewidth=2)
    ax.plot(t, np.abs(A2_final)**2, 'r-', label='Compressed', linewidth=2)
    ax.set_xlabel('Time (ps)', fontsize=12)
    ax.set_ylabel('Power (W)', fontsize=12)
    ax.set_title('Pulse Shape Evolution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 脉宽演化
    ax = axes[0, 1]
    ax.plot(z_total, w_total, 'b-', linewidth=2)
    ax.axvline(z_total[len(z_total)//2], color='r', linestyle='--', label='Transition')
    ax.set_xlabel('Propagation Distance (m)', fontsize=12)
    ax.set_ylabel('Pulse Width FWHM (ps)', fontsize=12)
    ax.set_title('Pulse Width Evolution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 频谱 - 使用波长坐标
    ax = axes[1, 0]
    ax.plot(wavelength, s0/np.max(s0), 'b-', label='Initial', linewidth=2)
    ax.plot(wavelength, s1/np.max(s1), 'g-', label='After Nonlinear', linewidth=2)
    ax.plot(wavelength, s2/np.max(s2), 'r-', label='Compressed', linewidth=2)
    ax.set_xlabel('Wavelength (nm)', fontsize=12)
    ax.set_ylabel('Normalized Spectrum', fontsize=12)
    ax.set_title('Spectral Evolution', fontsize=14, fontweight='bold')
    # 设置合理的波长范围
    center_wl_nm = center_wavelength * 1e9
    ax.set_xlim([center_wl_nm - 50, center_wl_nm + 50])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 啁啾
    ax = axes[1, 1]
    def instantaneous_freq(A, t):
        phase = np.unwrap(np.angle(A))
        chirp = -np.gradient(phase, t) / (2*np.pi)
        return chirp
    
    chirp0 = instantaneous_freq(A0, t)
    chirp2 = instantaneous_freq(A2_final, t)
    
    ax.plot(t, chirp0, 'b-', label='Initial', linewidth=2)
    ax.plot(t, chirp2, 'r-', label='Compressed', linewidth=2)
    ax.set_xlabel('Time (ps)', fontsize=12)
    ax.set_ylabel('Instantaneous Frequency (THz)', fontsize=12)
    ax.set_title('Chirp Characteristics', fontsize=14, fontweight='bold')
    ax.set_xlim([-5, 5])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi)
    
    return fig, axes


def plot_modulation_instability(z_array, A_array, t, freq_shift, A0, 
                               gain_MI, omega_MI, save_path=None, dpi=300,
                               center_wavelength=1064e-9):
    """
    绘制调制不稳定性可视化
    
    Parameters:
    -----------
    z_array : array
        传播距离数组
    A_array : array
        场振幅数组
    t : array
        时间数组
    freq_shift : array
        频率数组 (已 fftshift 过，单调递增)
    A0 : array
        初始场
    gain_MI : array
        理论增益谱
    omega_MI : array
        理论增益频率
    save_path : str, optional
        保存路径
    dpi : int
        图像分辨率
    center_wavelength : float
        中心波长 (m)
    """
    # 修复：假设输入数据已经正确 fftshift 过
    # 直接计算波长，不要再次 fftshift
    wavelength = freq_to_wavelength(freq_shift, center_wavelength)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 时域演化
    ax = axes[0, 0]
    Z, T = np.meshgrid(z_array[::5], t)
    power = np.abs(A_array[::5].T)**2
    im = ax.contourf(Z, T, power, levels=50, cmap='hot')
    ax.set_xlabel('Propagation Distance (m)', fontsize=12)
    ax.set_ylabel('Time (ps)', fontsize=12)
    ax.set_title('Time Domain Evolution', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Power (W)')
    
    # 频域演化 - 使用波长坐标
    ax = axes[0, 1]
    spectrogram = np.zeros((len(wavelength), len(z_array[::5])))
    for i, A in enumerate(A_array[::5]):
        spec = np.fft.fftshift(np.abs(np.fft.fft(A))**2)
        spectrogram[:, i] = 10*np.log10(spec / np.max(spec) + 1e-10)
    
    W, Z = np.meshgrid(wavelength, z_array[::5])
    im = ax.contourf(Z, W, spectrogram.T, levels=50, cmap='jet', 
                     vmin=-50, vmax=0)
    ax.set_xlabel('Propagation Distance (m)', fontsize=12)
    ax.set_ylabel('Wavelength (nm)', fontsize=12)
    ax.set_title('Wavelength Domain Evolution (dB)', fontsize=14, fontweight='bold')
    # 设置合理的波长范围
    center_wl_nm = center_wavelength * 1e9
    ax.set_ylim([center_wl_nm - 10, center_wl_nm + 10])
    plt.colorbar(im, ax=ax, label='Relative Intensity (dB)')
    
    # 增益谱 - 保持频率坐标（理论增益通常用频率表示）
    ax = axes[1, 0]
    # 修正：将角频率 (rad/s) 转换为 THz
    ax.plot(omega_MI/(2*np.pi*1e12), gain_MI, 'b-', 
            label='Theoretical Gain', linewidth=2)
    
    # 数值结果 - 保持频率坐标以便与理论增益比较
    spec_init = np.fft.fftshift(np.abs(np.fft.fft(A0))**2)
    spec_final = np.fft.fftshift(np.abs(np.fft.fft(A_array[-1]))**2)
    numerical_gain = np.log(spec_final / (spec_init + 1e-10)) / z_array[-1]
    numerical_gain[numerical_gain < 0] = 0
    
    ax.plot(freq_shift, numerical_gain, 'r--', 
            label='Numerical Result', linewidth=2, alpha=0.7)
    ax.set_xlabel('Frequency (THz)', fontsize=12)
    ax.set_ylabel('Gain (m⁻¹)', fontsize=12)
    ax.set_title('Modulation Instability Gain Spectrum', fontsize=14, fontweight='bold')
    ax.set_xlim([-0.5, 0.5])
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # 峰值功率演化
    ax = axes[1, 1]
    peak_power = np.max(np.abs(A_array)**2, axis=1)
    mean_power = np.mean(np.abs(A_array)**2, axis=1)
    
    ax.plot(z_array, peak_power, 'r-', label='Peak Power', linewidth=2)
    ax.plot(z_array, mean_power, 'b-', label='Mean Power', linewidth=2)
    ax.set_xlabel('Propagation Distance (m)', fontsize=12)
    ax.set_ylabel('Power (W)', fontsize=12)
    ax.set_title('Power Evolution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi)
    
    return fig, axes


def plot_soliton_collision(z_array, A_array, t, save_path=None, dpi=300):
    """
    绘制孤子碰撞可视化
    
    Parameters:
    -----------
    z_array : array
        传播距离数组
    A_array : array
        场振幅数组
    t : array
        时间数组
    save_path : str, optional
        保存路径
    dpi : int
        图像分辨率
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 时空演化
    ax = axes[0, 0]
    Z, T = np.meshgrid(z_array[::5], t)
    power = np.abs(A_array[::5].T)**2
    im = ax.contourf(Z, T, power, levels=50, cmap='hot')
    ax.set_xlabel('Propagation Distance (m)', fontsize=12)
    ax.set_ylabel('Time (ps)', fontsize=12)
    ax.set_title('Soliton Collision Spacetime Diagram', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Power (W)')
    
    # 不同时刻的脉冲形状
    ax = axes[0, 1]
    indices = [0, len(z_array)//3, len(z_array)//2, 2*len(z_array)//3, -1]
    colors = ['b', 'g', 'orange', 'purple', 'r']
    for idx, color in zip(indices, colors):
        ax.plot(t, np.abs(A_array[idx])**2, color=color,
                label=f'z={z_array[idx]:.1f}m', linewidth=2)
    ax.set_xlabel('Time (ps)', fontsize=12)
    ax.set_ylabel('Power (W)', fontsize=12)
    ax.set_title('Pulse Shapes at Different Positions', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    # 相位演化
    ax = axes[1, 0]
    phase = np.unwrap(np.angle(A_array), axis=1)
    im = ax.contourf(Z, T, phase[::5].T, levels=50, cmap='twilight')
    ax.set_xlabel('Propagation Distance (m)', fontsize=12)
    ax.set_ylabel('Time (ps)', fontsize=12)
    ax.set_title('Phase Evolution', fontsize=14, fontweight='bold')
    plt.colorbar(im, ax=ax, label='Phase (rad)')
    
    # 能量分布
    ax = axes[1, 1]
    energy_left = []
    energy_right = []
    for A in A_array:
        power = np.abs(A)**2
        mid_idx = len(t) // 2
        energy_left.append(np.sum(power[:mid_idx]))
        energy_right.append(np.sum(power[mid_idx:]))
    
    ax.plot(z_array, energy_left, 'b-', label='Left Energy', linewidth=2)
    ax.plot(z_array, energy_right, 'r-', label='Right Energy', linewidth=2)
    ax.set_xlabel('Propagation Distance (m)', fontsize=12)
    ax.set_ylabel('Energy (arb. units)', fontsize=12)
    ax.set_title('Energy Distribution Evolution', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi)
    
    return fig, axes


# 辅助函数
def pulse_width(A_arr, t):
    """
    计算脉冲宽度 (FWHM)
    
    Parameters:
    -----------
    A_arr : array
        场振幅数组
    t : array
        时间数组
    
    Returns:
    --------
    widths : array
        脉宽数组
    """
    widths = []
    for A in A_arr:
        power = np.abs(A)**2
        power_norm = power / np.max(power)
        fwhm_indices = np.where(power_norm > 0.5)[0]
        if len(fwhm_indices) > 0:
            width = t[fwhm_indices[-1]] - t[fwhm_indices[0]]
            widths.append(width)
        else:
            widths.append(np.nan)
    return np.array(widths)


def instantaneous_freq(A, t):
    """
    计算瞬时频率
    
    Parameters:
    -----------
    A : array
        场振幅
    t : array
        时间数组
    
    Returns:
    --------
    chirp : array
        瞬时频率
    """
    phase = np.unwrap(np.angle(A))
    chirp = -np.gradient(phase, t) / (2*np.pi)
    return chirp


def show_all_plots():
    """
    显示所有当前打开的图形
    """
    plt.show()


def close_all_plots():
    """
    关闭所有当前打开的图形
    """
    plt.close('all')


def detect_spectral_range(wavelength, spectrum, threshold_ratio=1e-4, margin_nm=10):
    """自动检测频谱有效范围"""
    # 对波长进行排序（确保从小到大）
    sort_idx = np.argsort(wavelength)
    wl_sorted = wavelength[sort_idx]
    spec_sorted = spectrum[sort_idx]
    
    # 计算阈值
    max_intensity = np.max(spec_sorted)
    threshold = max_intensity * threshold_ratio
    
    # 找到超过阈值的点
    above_threshold = spec_sorted > threshold
    
    if not np.any(above_threshold):
        lambda_center = np.median(wl_sorted)
        return lambda_center - 50, lambda_center + 50
    
    indices = np.where(above_threshold)[0]
    first_idx = indices[0]
    last_idx = indices[-1]
    
    wl_min = wl_sorted[first_idx] - margin_nm
    wl_max = wl_sorted[last_idx] + margin_nm
    
    return wl_min, wl_max


def analyze_results(sim, A_evolution):
    """分析仿真结果，计算功率、能量和FWHM"""
    Power = np.abs(A_evolution)**2
    Spectrum = np.abs(fft(A_evolution, axis=1))**2
    
    # 尝试调用 sim 中的辅助函数，如果不存在则使用默认计算
    if hasattr(sim, '_compute_fwhm'):
        fwhm_in = sim._compute_fwhm(Power[0, :], sim.t)
        fwhm_out = sim._compute_fwhm(Power[-1, :], sim.t)
    else:
        fwhm_in, fwhm_out = 0.0, 0.0
        
    if hasattr(sim, '_compute_energy'):
        energy_in = sim._compute_energy(A_evolution[0, :])
        energy_out = sim._compute_energy(A_evolution[-1, :])
    else:
        energy_in = np.trapz(Power[0, :], sim.t)
        energy_out = np.trapz(Power[-1, :], sim.t)
    
    results = {
        'power': Power,
        'spectrum': Spectrum,
        'peak_power_in': np.max(Power[0, :]),
        'peak_power_out': np.max(Power[-1, :]),
        'energy_in': energy_in,
        'energy_out': energy_out,
        'fwhm_in': fwhm_in,
        'fwhm_out': fwhm_out,
    }
    return results


def plot_results(sim, A_evolution, save_path=None, reference_spectrum_data=None):
    """
    绘制主要仿真结果，包括时域、频域的演化和对比。
    
    Args:
        sim: NonlinearCrystalSimulator 实例
        A_evolution: 传播演化数组
        save_path: 保存路径
        reference_spectrum_data: (wavelengths, intensities) 元组，用于绘制参考光谱
    """
    results = analyze_results(sim, A_evolution)
    Power = results['power']
    Spectrum = results['spectrum']
    
    fig = plt.figure(figsize=(14, 10))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    # 1. 时域演化
    ax1 = fig.add_subplot(gs[0, 0])
    extent_tuple = (sim.t[0]*1e12, sim.t[-1]*1e12, 0, sim.z[-1]*1e3)
    im = ax1.imshow(Power, aspect='auto', extent=extent_tuple,
                  origin='lower', cmap='hot', interpolation='bilinear')
    ax1.set_xlabel('Time (ps)')
    ax1.set_ylabel('Distance (mm)')
    ax1.set_title('Temporal Evolution', fontweight='bold')
    plt.colorbar(im, ax=ax1, label='Power (W)')
    
    # 2. 输入输出对比
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(sim.t*1e12, Power[0, :], 'b-', linewidth=2, label='Input')
    ax2.plot(sim.t*1e12, Power[-1, :], 'r--', linewidth=2, label='Output')
    ax2.set_xlabel('Time (ps)')
    ax2.set_ylabel('Power (W)')
    ax2.set_title('Pulse Comparison', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # 3. 频谱演化
    ax3 = fig.add_subplot(gs[1, 0])
    freq = fftfreq(sim.Nt, sim.dt)
    
    # 计算波长 (兼容不同的波长转换函数名)
    if 'wavelength_from_frequency' in globals():
        wavelength = wavelength_from_frequency(sim, freq)
    elif 'freq_to_wavelength' in globals():
        # 兼容旧版本函数名
        wavelength = freq_to_wavelength(freq * 1e-12, sim.lambda0, is_angular=False)
    else:
        # 手动计算兜底
        omega_total = sim.omega0 + 2*np.pi*freq
        wavelength = 2 * np.pi * sim.c / omega_total * 1e9
    
    # 自动检测显示范围
    output_spectrum = Spectrum[-1, :]
    wl_min, wl_max = detect_spectral_range(wavelength, output_spectrum, 
                                           threshold_ratio=1e-4, margin_nm=10)
    
    mask = (wavelength > wl_min) & (wavelength < wl_max)
    wl_plot = wavelength[mask]
    spec_plot = Spectrum[:, mask]
    
    # 排序
    sort_idx = np.argsort(wl_plot)
    wl_plot = wl_plot[sort_idx]
    spec_plot = spec_plot[:, sort_idx]
    
    extent_wave_tuple = (wl_plot[0], wl_plot[-1], 0, sim.z[-1]*1e3)
    im = ax3.imshow(spec_plot, aspect='auto', extent=extent_wave_tuple,
                  origin='lower', cmap='viridis', interpolation='bilinear')
    ax3.set_xlabel('Wavelength (nm)')
    ax3.set_ylabel('Distance (mm)')
    ax3.set_title('Spectral Evolution', fontweight='bold')
    plt.colorbar(im, ax=ax3, label='Intensity')
    
    # 4. 频谱对比
    ax4 = fig.add_subplot(gs[1, 1])
    spec_in = Spectrum[0, :] / np.max(Spectrum[0, :])
    spec_out = Spectrum[-1, :] / np.max(Spectrum[-1, :])
    
    ax4.plot(wl_plot, spec_in[mask][sort_idx], 'b-', linewidth=2, label='Input (Sim)')
    ax4.plot(wl_plot, spec_out[mask][sort_idx], 'r--', linewidth=2, label='Output (Sim)')
    
    if reference_spectrum_data is not None:
        ref_wl, ref_intens = reference_spectrum_data
        ax4.plot(ref_wl, ref_intens / np.max(ref_intens), 'ko', markersize=3, alpha=0.6, label='Reference')
        
    ax4.set_xlabel('Wavelength (nm)')
    ax4.set_ylabel('Normalized Intensity')
    ax4.set_title('Spectral Comparison', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1.05)
    
    # 5. 相位积累 (如果有)
    ax5 = fig.add_subplot(gs[2, :])
    if hasattr(sim, 'phase_accumulation') and sim.phase_accumulation is not None:
        phase_plot = sim.phase_accumulation[mask][sort_idx]
        ax5.plot(wl_plot, phase_plot, 'g-', linewidth=2.5, label='Phase Accumulation')
        ax5.set_xlabel('Wavelength (nm)')
        ax5.set_ylabel('Phase (rad)')
        ax5.set_title('Phase Accumulation', fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.legend()
    
    # 使用手动布局调整，避免 tight_layout 警告
    plt.subplots_adjust(left=0.08, right=0.95, bottom=0.08, top=0.95, 
                      hspace=0.3, wspace=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"图像已保存: {save_path}")
    
    plt.show()
    
    print(f"\n{'='*30}")
    print("仿真结果摘要:")
    print(f"输入峰值功率: {results['peak_power_in']:.1f} W")
    print(f"输出峰值功率: {results['peak_power_out']:.1f} W")
    print(f"能量守恒率: {results['energy_out']/results['energy_in']*100:.2f}%")
    print(f"{'='*30}\n")


def draw_parameter_table(sim, pulse_params, save_path=None):
    """
    绘制详细的仿真参数表 (增强版，兼容 Adapter)
    """
    import matplotlib.pyplot as plt
    
    # 1. 准备数据
    # 安全获取 FWHM
    fwhm = pulse_params.get('fwhm', 1e-12)
    T0 = fwhm / 1.665
    
    # 安全获取 beta2
    beta2_val = getattr(sim, 'beta2', 0)
    
    # 计算特征长度 L_D
    if beta2_val != 0:
        L_D = T0**2 / np.abs(beta2_val)
    else:
        L_D = float('inf')
        
    # 计算特征长度 L_NL
    peak_power = pulse_params.get('peak_power', 0)
    gamma_val = getattr(sim, 'gamma', 0)
    
    if gamma_val > 0 and peak_power > 0:
        L_NL = 1 / (gamma_val * peak_power)
    else:
        L_NL = float('inf')

    # 定义孤子阶数 N
    if L_D != float('inf') and L_NL != float('inf') and L_NL > 0:
        N_soliton = np.sqrt(L_D / L_NL)
    else:
        N_soliton = 0

    # 构造表格数据列
    data = []
    
    # --- A. 基础设置 ---
    L_val = getattr(sim, 'L', 0)
    dz_val = getattr(sim, 'dz', 0)
    
    data.append(["--- Simulation Setup ---", "", ""])
    data.append(["Crystal Length (L)", f"{L_val*1000:.2f}", "mm"])
    data.append(["Step Number (Nz)", f"{getattr(sim, 'Nz', 0)}", ""])
    data.append(["Step Size (avg dz)", f"{dz_val*1e6:.2f}", "um"])
    data.append(["Time Window", f"{getattr(sim, 'T', 0)*1e12:.1f}", "ps"])
    
    # --- B. 脉冲参数 ---
    lambda0 = getattr(sim, 'lambda0', 1064e-9)
    data.append(["--- Pulse Parameters ---", "", ""])
    data.append(["Center Wavelength", f"{lambda0*1e9:.1f}", "nm"])
    data.append(["Pulse Energy", f"{pulse_params.get('energy', 0)*1e6:.2f}", "uJ"])
    data.append(["Pulse Width (FWHM)", f"{fwhm*1e12:.3f}", "ps"])
    data.append(["Peak Power (Input)", f"{peak_power/1e6:.2f}", "MW"])
    
    # --- C. 材料与物理 ---
    beam_radius = getattr(sim, 'beam_radius', 0)
    n2_val = getattr(sim, 'n2', 0)
    
    data.append(["--- Material & Physics ---", "", ""])
    data.append(["Material", f"{getattr(sim, 'material', 'Unknown')}", ""])
    data.append(["Beam Radius", f"{beam_radius*1e6:.1f}", "um"])
    data.append(["Nonlinear Index (n2)", f"{n2_val:.2e}", "m^2/W"])
    data.append(["Gamma (gamma)", f"{gamma_val:.2e}", "/(W*m)"])
    data.append(["GVD (beta2)", f"{beta2_val*1e27:.1f}", "fs^2/mm"])
    
    # --- D. 特征尺度分析 ---
    data.append(["--- Characteristic Lengths ---", "", ""])
    data.append(["Dispersion Length (L_D)", f"{L_D:.4f}", "m"])
    data.append(["Nonlinear Length (L_NL)", f"{L_NL*1000:.2f}", "mm"])
    
    if L_D != float('inf'):
        ratio_ld = L_val/L_D
    else:
        ratio_ld = 0
        
    if L_NL != float('inf') and L_NL > 0:
        ratio_lnl = L_val/L_NL
    else:
        ratio_lnl = 0
        
    data.append(["Ratio (L / L_D)", f"{ratio_ld:.4f}", ""])
    data.append(["Ratio (L / L_NL)", f"{ratio_lnl:.2f}", ""])
    data.append(["Soliton Order (N)", f"{N_soliton:.2f}", ""])

    # 绘图逻辑 (保持不变)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    
    table = plt.table(cellText=data, 
                      colLabels=["Parameter", "Value", "Unit"],
                      cellLoc='center', loc='center',
                      bbox=[0.1, 0.05, 0.8, 0.9])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)
    
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#40466e')
        elif data[row-1][0].startswith("---"):
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#e6e6e6')
            
    plt.title("Simulation Parameter Report", fontsize=14, weight='bold', pad=20)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✅ 参数表已保存: {save_path}")
    
    plt.show()
