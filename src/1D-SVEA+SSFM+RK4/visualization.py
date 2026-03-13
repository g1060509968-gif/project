"""
可视化模块
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fftpack import fft, fftfreq, fftshift
import matplotlib

# 字体设置
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


def wavelength_from_frequency(sim, freq):
    """统一的波长-频率转换函数
    
    使用精确的转换公式而非线性近似
    
    Args:
        sim: 仿真器实例
        freq: 频率偏移数组 (Hz)
        
    Returns:
        wavelength: 波长数组 (nm)
    """
    omega_total = sim.omega0 + 2*np.pi*freq
    wavelength = 2 * np.pi * sim.c / omega_total * 1e9
    return wavelength


def detect_spectral_range(wavelength, spectrum, threshold_ratio=1e-4, margin_nm=10):
    """自动检测频谱有效范围
    
    通过检测首个和最后一个非零强度点确定显示范围
    
    Args:
        wavelength: 波长数组 (nm)
        spectrum: 频谱强度数组
        threshold_ratio: 阈值比（相对于峰值），默认1e-4
        margin_nm: 向外扩展的波长范围 (nm)，默认10nm
        
    Returns:
        wl_min: 最小波长 (nm)
        wl_max: 最大波长 (nm)
    """
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
        # 如果没有超过阈值的点，返回默认范围
        lambda_center = np.median(wl_sorted)
        return lambda_center - 50, lambda_center + 50
    
    # 找到首个和最后一个非零点的索引
    indices = np.where(above_threshold)[0]
    first_idx = indices[0]
    last_idx = indices[-1]
    
    # 获取对应的波长
    wl_min = wl_sorted[first_idx] - margin_nm
    wl_max = wl_sorted[last_idx] + margin_nm
    
    return wl_min, wl_max


def analyze_results(sim, A_evolution):
    """分析仿真结果"""
    Power = np.abs(A_evolution)**2
    Spectrum = np.abs(fft(A_evolution, axis=1))**2
    
    fwhm_in = sim._compute_fwhm(Power[0, :], sim.t)
    fwhm_out = sim._compute_fwhm(Power[-1, :], sim.t)
    
    results = {
        'power': Power,
        'spectrum': Spectrum,
        'peak_power_in': np.max(Power[0, :]),
        'peak_power_out': np.max(Power[-1, :]),
        'energy_in': sim._compute_energy(A_evolution[0, :]),
        'energy_out': sim._compute_energy(A_evolution[-1, :]),
        'fwhm_in': fwhm_in,
        'fwhm_out': fwhm_out,
    }
    
    return results

def plot_results(sim, A_evolution, save_path=None, reference_spectrum_data=None):
    """
    绘制主要仿真结果，包括时域、频域的演化和对比。
    
    Args:
        sim: NonlinearCrystalSimulator 实例。
        A_evolution: 传播演化数组。
        save_path (str, optional): 保存图片的路径。
        reference_spectrum_data (tuple, optional): 一个元组 (wavelengths, intensities)，用于绘制参考光谱。
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
    # 精确波长-频率转换（使用统一函数）
    wavelength = wavelength_from_frequency(sim, freq)
    
    # 自动检测频谱范围（使用输出频谱）
    output_spectrum = Spectrum[-1, :]
    wl_min, wl_max = detect_spectral_range(wavelength, output_spectrum, 
                                           threshold_ratio=1e-4, margin_nm=10)
    
    lambda_center = sim.lambda0 * 1e9
    mask = (wavelength > wl_min) & (wavelength < wl_max)
    wl_plot = wavelength[mask]
    spec_plot = Spectrum[:, mask]
    
    print(f"  频谱图波长范围: {wl_min:.1f} - {wl_max:.1f} nm (范围: {wl_max-wl_min:.1f} nm)")
    
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
    
    ax4.plot(wl_plot, spec_in[mask][sort_idx], 'b-', linewidth=2, label='Input Spectrum (Sim)')
    ax4.plot(wl_plot, spec_out[mask][sort_idx], 'r--', linewidth=2, label='Output Spectrum (Sim)')
    
    # --- 新增：绘制参考光谱数据 ---
    if reference_spectrum_data is not None:
        ref_wl, ref_intens = reference_spectrum_data
        # 归一化参考数据并绘制
        ax4.plot(ref_wl, ref_intens / ref_intens.max(), 'ko', markersize=3, alpha=0.6, label='Reference Data (from image)')
    # --- 结束新增部分 ---
        
    ax4.set_xlabel('Wavelength (nm)')
    ax4.set_ylabel('Normalized Intensity')
    ax4.set_title('Spectral Comparison', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    ax4.set_ylim(0, 1.05)
    
    # 5. 相位积累
    ax5 = fig.add_subplot(gs[2, :])
    if sim.phase_accumulation is not None:
        phase_plot = sim.phase_accumulation[mask][sort_idx]
        
        ax5.plot(wl_plot, phase_plot, 'g-', linewidth=2.5, label='Phase Accumulation')
        ax5.set_xlabel('Wavelength (nm)', fontsize=11)
        ax5.set_ylabel('Phase (rad)', fontsize=11)
        ax5.set_title('Phase Accumulation', fontweight='bold', fontsize=12)
        ax5.grid(True, alpha=0.3)
        ax5.legend(fontsize=10)
        
        ax5.axhline(y=0, color='k', linestyle='--', alpha=0.3, linewidth=1)
        ax5.axvline(x=lambda_center, color='r', linestyle='--', alpha=0.3, linewidth=1)
    
    if save_path:
        plt.savefig(save_path, dpi=150*2*2, bbox_inches='tight')
        print(f"图像已保存: {save_path}")
    
    plt.show()
    
    # 打印结果
    print(f"\n{'='*60}")
    print("仿真结果:")
    print(f"{'='*60}")
    print(f"  输入: {results['peak_power_in']:.1f} W | {results['energy_in']*1e6:.1f} uJ | {results['fwhm_in']*1e12:.2f} ps")
    print(f"  输出: {results['peak_power_out']:.1f} W | {results['energy_out']*1e6:.1f} uJ | {results['fwhm_out']*1e12:.2f} ps")
    print(f"  能量守恒: {results['energy_out']/results['energy_in']*100:.1f}%")
    print(f"{'='*60}\n")


def plot_compressibility(sim, A_final, threshold_db=-20):
    """
    绘制可压缩性分析图（增强版：包含GDD计算）
    
    Args:
        sim: 仿真器实例
        A_final: 最终输出的脉冲场 (复数数组)
        threshold_db: 分析相位的强度阈值(dB)
    """
    # 1. 准备频域数据
    spectrum_complex = fftshift(fft(A_final))
    freqs = fftshift(fftfreq(sim.Nt, sim.dt)) # Hz
    
    # 关键：转换为角频率偏移 (rad/s) 用于拟合，避免 2pi 换算错误
    # 物理公式: Phase(w) = 1/2 * GDD * w^2 + ...
    omega_shift = 2 * np.pi * freqs 
    
    # 转换频率为波长 (nm) 用于绘图 x 轴
    omega_total = sim.omega0 + omega_shift
    wavelengths = 2 * np.pi * sim.c / omega_total * 1e9
    
    # 2. 计算光谱强度
    spectral_intensity = np.abs(spectrum_complex)**2
    spectral_intensity /= np.max(spectral_intensity)
    
    # 3. 提取并解卷绕相位
    raw_phase = np.unwrap(np.angle(spectrum_complex))
    
    # 4. 确定有效分析范围 (Mask)
    threshold_linear = 10**(threshold_db/10)
    mask = spectral_intensity > threshold_linear
    
    if np.sum(mask) < 10:
        print("⚠️ 警告: 光谱能量太弱，无法分析")
        return

    valid_omega = omega_shift[mask]
    valid_phase = raw_phase[mask]
    valid_wavelengths = wavelengths[mask]
    
    # 5. 物理拟合：提取 GDD
    # 拟合公式: Phase = a * w^2 + b * w + c
    # 其中 a = 1/2 * GDD
    poly_coeffs = np.polyfit(valid_omega, valid_phase, 2)
    fitted_phase = np.polyval(poly_coeffs, valid_omega)
    
    # 计算 GDD (单位换算为 fs^2)
    # a = GDD/2  => GDD = 2a
    gdd_val_s2 = 2 * poly_coeffs[0]
    gdd_val_fs2 = gdd_val_s2 * 1e30 
    
    # 计算残余相位 (高阶色散 + 非线性相位噪声)
    residual_phase = valid_phase - fitted_phase
    
    # --- 开始绘图 ---
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    # 绘制光谱
    color_spec = 'tab:blue'
    ax1.set_xlabel('Wavelength (nm)', fontsize=12)
    ax1.set_ylabel('Spectral Intensity (norm.)', color=color_spec, fontsize=12)
    ax1.fill_between(valid_wavelengths, spectral_intensity[mask], color=color_spec, alpha=0.3)
    ax1.plot(valid_wavelengths, spectral_intensity[mask], color=color_spec, linewidth=1.5, label='Spectrum')
    ax1.tick_params(axis='y', labelcolor=color_spec)
    ax1.grid(True, alpha=0.2)
    
    # 绘制残余相位
    ax2 = ax1.twinx()
    color_phase = 'tab:red'
    ax2.set_ylabel('Residual Phase (rad)', color=color_phase, fontsize=12)
    ax2.plot(valid_wavelengths, residual_phase, color=color_phase, linewidth=2, label='Residual Phase (High Order)')
    ax2.tick_params(axis='y', labelcolor=color_phase)
    
    # 限制相位显示范围
    phase_range = np.percentile(residual_phase, 98) - np.percentile(residual_phase, 2)
    limit = max(1.0, phase_range * 1.5)
    ax2.set_ylim(-limit, limit)
    
    # 参考线
    ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    
    # --- 结论与数值显示 ---
    ptp_phase = np.max(residual_phase) - np.min(residual_phase)
    
    # 质量评判
    if ptp_phase < 1.0:
        quality = "Excellent"
        q_color = "green"
    elif ptp_phase < np.pi:
        quality = "Good"
        q_color = "orange"
    else:
        quality = "Poor"
        q_color = "red"

    plt.title(f'Compressibility Analysis\nPhase Quality: {quality}', fontsize=14, fontweight='bold')
    
    # 信息框：显示 GDD 和 残余相位范围
    info_text = (
        f"Compensatable GDD: {gdd_val_fs2:.0f} fs^2\n"
        f"Residual Phase (P-P): {ptp_phase:.2f} rad"
    )
    
    props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='gray')
    ax1.text(0.02, 0.95, info_text, transform=ax1.transAxes, fontsize=11,
            verticalalignment='top', bbox=props, fontfamily='monospace')

    plt.tight_layout()
    plt.show()

def draw_parameter_table(sim, pulse_params, save_path=None):
    """
    绘制详细的仿真参数表
    
    Args:
        sim: 仿真器实例
        pulse_params: 包含脉冲参数的字典 {'energy', 'fwhm', 'chirp', 'peak_power'}
        save_path: 保存路径
    """
    import matplotlib.pyplot as plt
    
    # 1. 准备数据
    # 计算特征长度
    # L_D = T0^2 / |beta2|, T0 = T_FWHM / 1.665
    T0 = pulse_params['fwhm'] / 1.665
    beta2_val = getattr(sim, 'beta2', 0)
    if beta2_val != 0:
        L_D = T0**2 / np.abs(beta2_val)
    else:
        L_D = float('inf')
        
    # L_NL = 1 / (gamma * P0)
    if sim.gamma > 0 and pulse_params.get('peak_power', 0) > 0:
        L_NL = 1 / (sim.gamma * pulse_params['peak_power'])
    else:
        L_NL = float('inf')

    # 定义孤子阶数 N = sqrt(L_D / L_NL)
    if L_D != float('inf') and L_NL != float('inf'):
        N_soliton = np.sqrt(L_D / L_NL)
    else:
        N_soliton = 0

    # 构造表格数据列: [参数名称, 值, 单位]
    data = []
    
    # --- A. 基础设置 ---
    data.append(["--- Simulation Setup ---", "", ""])
    data.append(["Crystal Length (L)", f"{sim.L*1000:.2f}", "mm"])
    data.append(["Step Number (Nz)", f"{sim.Nz}", ""])
    data.append(["Step Size (dz)", f"{sim.dz*1e6:.2f}", "um"])
    data.append(["Time Window", f"{sim.T*1e12:.1f}", "ps"])
    data.append(["Time Points", f"{sim.Nt}", ""])
    
    # --- B. 脉冲参数 ---
    data.append(["--- Pulse Parameters ---", "", ""])
    data.append(["Center Wavelength", f"{sim.lambda0*1e9:.1f}", "nm"])
    data.append(["Pulse Energy", f"{pulse_params['energy']*1e6:.2f}", "uJ"])
    data.append(["Pulse Width (FWHM)", f"{pulse_params['fwhm']*1e12:.3f}", "ps"])
    data.append(["Peak Power (Input)", f"{pulse_params['peak_power']/1e6:.2f}", "MW"])
    data.append(["Chirp Parameter (C)", f"{pulse_params.get('chirp', 0):.2f}", ""])
    
    # --- C. 材料与物理 ---
    data.append(["--- Material & Physics ---", "", ""])
    data.append(["Material", f"{sim.material}", ""])
    data.append(["Beam Radius", f"{sim.beam_radius*1e6:.1f}", "um"])
    data.append(["Nonlinear Index (n2)", f"{sim.n2:.2e}", "m^2/W"])
    data.append(["Gamma (γ)", f"{sim.gamma:.2e}", "/(W*m)"])
    data.append(["GVD (β2)", f"{beta2_val*1e27:.1f}", "fs^2/mm"])
    
    # --- D. 特征尺度分析 (关键) ---
    data.append(["--- Characteristic Lengths ---", "", ""])
    data.append(["Dispersion Length (L_D)", f"{L_D:.4f}", "m"])
    data.append(["Nonlinear Length (L_NL)", f"{L_NL*1000:.2f}", "mm"])
    data.append(["Ratio (L / L_D)", f"{sim.L/L_D:.4f}", ""])
    data.append(["Ratio (L / L_NL)", f"{sim.L/L_NL:.2f}", ""])
    data.append(["Soliton Order (N)", f"{N_soliton:.2f}", ""])

    # 2. 绘图
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    
    # 创建表格
    table = plt.table(cellText=data, 
                      colLabels=["Parameter", "Value", "Unit"],
                      cellLoc='center', loc='center',
                      bbox=[0.1, 0.05, 0.8, 0.9]) # [left, bottom, width, height]
    
    # 3. 样式美化
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.5)
    
    # 设置特定行的颜色（标题行）
    for (row, col), cell in table.get_celld().items():
        if row == 0: # Header
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#40466e')
        elif data[row-1][0].startswith("---"): # Section Headers
            cell.set_text_props(weight='bold')
            cell.set_facecolor('#e6e6e6')
            # 让标题跨列 (Matplotlib table跨列比较麻烦，这里简单处理为改背景色)
            
    plt.title("Simulation Parameter Report", fontsize=14, weight='bold', pad=20)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  ✅ 参数表已保存: {save_path}")
    
    plt.show()
