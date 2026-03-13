"""
精确傅里叶变换脉冲持续时间分析
====================================
本脚本使用数值积分而非FFT插值执行精确傅里叶变换，确保最高精度。

方法：直接数值积分傅里叶积分
日期：2025-10-20
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson, trapezoid
from scipy.interpolate import interp1d

# 设置matplotlib支持中文显示
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# 物理常数
c = 299792458  # 光速，单位：m/s

print("="*80)
print("精确傅里叶变换分析")
print("="*80)

# ============================================================================
# 1. 数据加载模块
# ============================================================================
print("\n[1/7] Loading spectrum data...")

def load_spectrum_data(filename):
    """
    加载光谱数据文件，支持多种格式：
    - .txt 文件：使用 np.loadtxt 读取
    - .csv 文件：使用 np.genfromtxt 读取，自动检测分隔符
    
    文件格式要求：
    第一列：波长（单位：nm）
    第二列：强度（任意单位，将自动归一化）
    
    参数：
        filename: 数据文件路径
    
    返回：
        wavelength: 波长数组（nm）
        intensity: 归一化强度数组
    """
    import os
    
    # 检查文件是否存在
    if not os.path.exists(filename):
        raise FileNotFoundError(f"数据文件 '{filename}' 不存在")
    
    # 根据文件扩展名选择加载方式
    file_ext = os.path.splitext(filename)[1].lower()
    
    if file_ext == '.txt':
        # 文本文件：假设有标题行，跳过第一行
        try:
            data = np.loadtxt(filename, skiprows=1)
        except:
            # 如果没有标题行，直接加载
            data = np.loadtxt(filename)
    elif file_ext == '.csv':
        # CSV文件：尝试自动检测分隔符
        try:
            # 首先尝试逗号分隔，跳过可能的标题行
            data = np.genfromtxt(filename, delimiter=',', skip_header=1)
            # 检查数据是否有效（可能因为末尾有逗号导致形状不对）
            if data.ndim != 2 or data.shape[1] < 2:
                # 尝试无标题行
                data = np.genfromtxt(filename, delimiter=',')
        except:
            # 如果逗号分隔失败，尝试通用分隔符
            try:
                data = np.genfromtxt(filename, delimiter=None, skip_header=1)
                if data.ndim != 2 or data.shape[1] < 2:
                    data = np.genfromtxt(filename, delimiter=None)
            except:
                # 最后尝试：读取原始数据并处理
                with open(filename, 'r') as f:
                    lines = f.readlines()
                # 处理每行数据，移除末尾的逗号
                processed_lines = []
                for line in lines:
                    line = line.strip()
                    if line.endswith(','):
                        line = line[:-1]
                    if line:  # 跳过空行
                        processed_lines.append(line)
                # 将处理后的数据转换为数组
                data = np.array([list(map(float, line.split(','))) for line in processed_lines])
    else:
        # 其他格式：尝试通用加载
        data = np.genfromtxt(filename, delimiter=None)
    
    # 检查数据形状
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"数据文件 '{filename}' 格式不正确，需要至少两列数据")
    
    # 提取波长和强度
    wavelength = data[:, 0]  # nm
    intensity = data[:, 1]   # a.u.
    
    # 移除可能的NaN值
    valid_mask = ~(np.isnan(wavelength) | np.isnan(intensity))
    wavelength = wavelength[valid_mask]
    intensity = intensity[valid_mask]
    
    # 归一化强度
    if np.max(intensity) > 0:
        intensity = intensity / np.max(intensity)
    else:
        print("  ⚠ 警告：强度数据全为零")
    
    return wavelength, intensity

# 加载数据
# 默认使用原来的文件路径，但用户可以修改为CSV文件
data_file = 'D:\project\Sheet1.csv'  # 可以修改为 'input_spectrum.csv' 或 'input.csv'
wavelength, intensity = load_spectrum_data(data_file)

print(f"  ✓ Loaded {len(wavelength)} data points from '{data_file}'")
print(f"  ✓ Wavelength range: {wavelength.min():.2f} - {wavelength.max():.2f} nm")

# ============================================================================
# 2. 频率域转换模块（包含正确的雅可比变换）
# ============================================================================
print("\n[2/7] Converting to frequency domain...")

# Convert wavelength to frequency: ν = c/λ
freq = c / (wavelength * 1e-9)  # Hz

# Sort by increasing frequency
sort_idx = np.argsort(freq)
freq = freq[sort_idx]
wavelength_sorted = wavelength[sort_idx]
intensity_sorted = intensity[sort_idx]

# Apply Jacobian transformation: I(ν) = I(λ) × |dλ/dν|
# Since λ = c/ν, we have dλ/dν = -c/ν² = -λ²/c
jacobian = (wavelength_sorted * 1e-9)**2 / c
intensity_freq = intensity_sorted * jacobian

# Normalize frequency-domain intensity
intensity_freq = intensity_freq / np.max(intensity_freq)

print(f"  ✓ Frequency range: {freq.min()/1e12:.2f} - {freq.max()/1e12:.2f} THz")

# ============================================================================
# 3. 光谱半高全宽（FWHM）计算模块
# ============================================================================
print("\n[3/7] Calculating spectral FWHM...")

def calculate_fwhm_precise(x, y):
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

# 计算波长域的半高全宽
lambda_left, lambda_right, delta_lambda = calculate_fwhm_precise(wavelength, intensity)
center_wavelength = (lambda_left + lambda_right) / 2

# 计算频率域的半高全宽
freq_left, freq_right, delta_nu = calculate_fwhm_precise(freq, intensity_freq)
center_freq = (freq_left + freq_right) / 2
delta_nu_THz = delta_nu / 1e12

print(f"  ✓ Wavelength FWHM: {delta_lambda:.3f} nm @ {center_wavelength:.2f} nm")
print(f"  ✓ Frequency FWHM: {delta_nu_THz:.3f} THz @ {center_freq/1e12:.2f} THz")

# ============================================================================
# 4. 精确傅里叶变换到时域模块
# ============================================================================
print("\n[4/7] Performing EXACT Fourier Transform...")
print("  Method: Direct numerical integration of Fourier integral")

# Create interpolation function for smooth integration
# Remove duplicate frequency values and corresponding intensities
unique_freq, unique_indices = np.unique(freq, return_index=True)
unique_intensity_freq = intensity_freq[unique_indices]

print(f"  ✓ Removed {len(freq) - len(unique_freq)} duplicate frequency points")
print(f"  ✓ Using {len(unique_freq)} unique frequency points for interpolation")

interp_amplitude = interp1d(unique_freq, np.sqrt(unique_intensity_freq), kind='cubic',
                           bounds_error=False, fill_value=0)

# Define time axis (centered around zero)
# Time resolution determined by frequency span
freq_span = freq.max() - freq.min()
dt_max = 1 / freq_span  # Maximum meaningful time resolution
t_max = 1 / (freq[1] - freq[0])  # Maximum time window

# Create dense time grid
N_time = 4096  # Number of time points
time = np.linspace(-t_max/2, t_max/2, N_time)
time_fs = time * 1e15  # Convert to femtoseconds

print(f"  ✓ Time grid: {N_time} points from {time_fs.min():.1f} to {time_fs.max():.1f} fs")

# Perform exact Fourier transform: E(t) = ∫ E(ω) exp(-iωt) dω
# Using E(ω) = √I(ω) for transform-limited pulse (zero phase)
print("  ✓ Computing Fourier integral (this may take a moment)...")

field_time = np.zeros(N_time, dtype=complex)

# Shift frequency to be centered around zero for proper FT
freq_centered = unique_freq - center_freq
omega = 2 * np.pi * freq_centered  # Angular frequency

for i, t in enumerate(time):
    # Integrand: E(ω) * exp(-iωt)
    integrand = np.sqrt(unique_intensity_freq) * np.exp(-1j * omega * t)
    # Use Simpson's rule for accurate integration
    field_time[i] = simpson(integrand, x=freq_centered)
    
    if (i+1) % 512 == 0:
        print(f"    Progress: {(i+1)/N_time*100:.1f}%")

print("  ✓ Fourier transform completed")

# Calculate intensity in time domain
intensity_time = np.abs(field_time)**2
intensity_time = intensity_time / np.max(intensity_time)

# ============================================================================
# 5. 高精度脉冲半高全宽计算模块
# ============================================================================
print("\n[5/7] Calculating pulse FWHM...")

t_left, t_right, fwhm = calculate_fwhm_precise(time_fs, intensity_time)

if fwhm is None:
    print("  ✗ ERROR: Could not determine FWHM")
    fwhm_fs = 0
else:
    fwhm_fs = fwhm
    print(f"  ✓ Pulse FWHM: {fwhm_fs:.3f} fs")
    print(f"  ✓ FWHM edges: {t_left:.3f} fs to {t_right:.3f} fs")

# ============================================================================
# 6. 时间带宽积计算模块
# ============================================================================
print("\n[6/7] Calculating Time-Bandwidth Product...")

TBP = delta_nu * fwhm_fs * 1e-15

print(f"  ✓ TBP = Δν × Δt = {delta_nu_THz:.3f} THz × {fwhm_fs:.3f} fs = {TBP:.4f}")
print(f"\n  Reference values:")
print(f"    • Gaussian pulse:     TBP = 0.4413")
print(f"    • Sech² pulse:        TBP = 0.3148")
print(f"    • Lorentzian pulse:   TBP = 0.2206")
print(f"\n  Your pulse: TBP = {TBP:.4f}")

if TBP < 0.44:
    print(f"    → Closer to Sech² shape")
elif TBP < 0.50:
    print(f"    → Close to Gaussian shape")
else:
    print(f"    → Non-standard spectral shape (structured/asymmetric)")

# ============================================================================
# 7. 数据可视化模块
# ============================================================================
print("\n[7/7] Generating visualization...")

# 创建大尺寸图形
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

# 子图(a)：波长域光谱
ax1 = fig.add_subplot(gs[0, 0])
ax1.plot(wavelength, intensity, 'b-', linewidth=2, label='Spectrum')
ax1.axhline(0.5, color='r', linestyle='--', alpha=0.5, linewidth=1)
if lambda_left and lambda_right:
    ax1.axvline(lambda_left, color='r', linestyle='--', alpha=0.5, linewidth=1)
    ax1.axvline(lambda_right, color='r', linestyle='--', alpha=0.5, linewidth=1)
    ax1.fill_between(wavelength, 0, intensity, where=(intensity >= 0.5), 
                    alpha=0.3, color='red', label=f'FWHM = {delta_lambda:.2f} nm')
ax1.set_xlabel('Wavelength (nm)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Normalized Intensity', fontsize=12, fontweight='bold')
ax1.set_title(f'(a) Input Spectrum (Wavelength Domain)', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)

# 子图(b)：频率域光谱
ax2 = fig.add_subplot(gs[0, 1])
ax2.plot(freq/1e12, intensity_freq, 'g-', linewidth=2, label='Spectrum')
ax2.axhline(0.5, color='r', linestyle='--', alpha=0.5, linewidth=1)
if freq_left and freq_right:
    ax2.axvline(freq_left/1e12, color='r', linestyle='--', alpha=0.5, linewidth=1)
    ax2.axvline(freq_right/1e12, color='r', linestyle='--', alpha=0.5, linewidth=1)
    ax2.fill_between(freq/1e12, 0, intensity_freq, where=(intensity_freq >= 0.5),
                    alpha=0.3, color='red', label=f'FWHM = {delta_nu_THz:.2f} THz')
ax2.set_xlabel('Frequency (THz)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Normalized Intensity', fontsize=12, fontweight='bold')
ax2.set_title(f'(b) Spectrum (Frequency Domain)', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)

# 子图(c)：时域全视图
ax3 = fig.add_subplot(gs[1, :])
ax3.plot(time_fs, intensity_time, 'purple', linewidth=2, label='Pulse intensity')
ax3.axhline(0.5, color='r', linestyle='--', alpha=0.5, linewidth=1, label='FWHM level')
ax3.set_xlabel('Time (fs)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Normalized Intensity', fontsize=12, fontweight='bold')
ax3.set_title(f'(c) Fourier-Limited Pulse (Full Time Window)', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3)
ax3.legend(fontsize=10)
ax3.set_xlim(time_fs.min(), time_fs.max())

# 子图(d)：时域放大视图
ax4 = fig.add_subplot(gs[2, :])
if fwhm_fs > 0:
    # 计算缩放范围（至少200 fs窗口）
    zoom_range = max(fwhm_fs * 4, 200)
    center_time = (t_left + t_right) / 2
    time_mask = (time_fs >= center_time - zoom_range) & (time_fs <= center_time + zoom_range)
    
    # 绘制脉冲强度曲线
    ax4.plot(time_fs[time_mask], intensity_time[time_mask], 'purple', linewidth=2.5, 
            label='Pulse intensity')
    ax4.axhline(0.5, color='r', linestyle='--', alpha=0.5, linewidth=1.5, label='FWHM level')
    ax4.axvline(t_left, color='r', linestyle='--', alpha=0.7, linewidth=1.5)
    ax4.axvline(t_right, color='r', linestyle='--', alpha=0.7, linewidth=1.5)
    ax4.fill_between(time_fs[time_mask], 0, intensity_time[time_mask], 
                    where=(intensity_time[time_mask] >= 0.5), 
                    alpha=0.3, color='red', label=f'FWHM = {fwhm_fs:.2f} fs')
    
    # 添加标注
    ax4.annotate(f'{t_left:.2f} fs', xy=(t_left, 0.5), xytext=(t_left-zoom_range*0.3, 0.65),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=11, color='red', fontweight='bold')
    ax4.annotate(f'{t_right:.2f} fs', xy=(t_right, 0.5), xytext=(t_right+zoom_range*0.15, 0.65),
                arrowprops=dict(arrowstyle='->', color='red', lw=1.5),
                fontsize=11, color='red', fontweight='bold')
    
    ax4.set_xlim(center_time - zoom_range, center_time + zoom_range)
else:
    ax4.plot(time_fs, intensity_time, 'purple', linewidth=2)

ax4.set_xlabel('Time (fs)', fontsize=12, fontweight='bold')
ax4.set_ylabel('Normalized Intensity', fontsize=12, fontweight='bold')
ax4.set_title(f'(d) Pulse FWHM Detail: τ = {fwhm_fs:.2f} fs (TBP = {TBP:.3f})', 
            fontsize=13, fontweight='bold')
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=10, loc='upper right')

# 保存高分辨率图像
plt.savefig('exact_fourier_transform.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Figure saved: exact_fourier_transform.png")

# ============================================================================
# 8. 结果保存模块
# ============================================================================
# 保存时域数据
np.savetxt('pulse_time_domain.txt', 
           np.column_stack([time_fs, intensity_time]),
           header='时间 (fs)\t强度 (归一化)',
           fmt='%.6e')
print(f"  ✓ Time-domain data saved: pulse_time_domain.txt")

# 保存频率域数据
np.savetxt('spectrum_frequency_domain.txt',
           np.column_stack([freq/1e12, intensity_freq]),
           header='Frequency (THz)\tIntensity (normalized)',
           fmt='%.6e')
print(f"  ✓ Frequency-domain data saved: spectrum_frequency_domain.txt")

# ============================================================================
# 9. 最终结果总结模块
# ============================================================================
print("\n" + "="*80)
print("EXACT FOURIER TRANSFORM ANALYSIS - FINAL RESULTS")
print("="*80)
print(f"\nINPUT SPECTRUM:")
print(f"  Center Wavelength:              {center_wavelength:.3f} nm")
print(f"  Spectral FWHM (wavelength):     {delta_lambda:.3f} nm")
print(f"  Spectral FWHM (frequency):      {delta_nu_THz:.3f} THz")
print(f"\nFOURIER-LIMITED PULSE:")
print(f"  Pulse Duration (FWHM):          {fwhm_fs:.3f} fs")
print(f"  Time-Bandwidth Product:         {TBP:.4f}")
print(f"\nMETHOD:")
print(f"  • Direct numerical integration of Fourier integral")
print(f"  • Simpson's rule for maximum accuracy")
print(f"  • No FFT interpolation artifacts")
print(f"  • Transform-limited (zero phase)")
print("="*80)

# 显示图形
plt.show()
