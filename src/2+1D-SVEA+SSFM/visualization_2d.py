"""
(2+1)D 可视化模块 - 简化版本
专门用于处理 (2+1)D 仿真数据的可视化
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D  # 导入3D绘图工具

# 字体设置
matplotlib.rcParams['font.family'] = 'sans-serif'
matplotlib.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus'] = False


def analyze_results_2d(sim, A_evolution):
    """分析 (2+1)D 仿真结果"""
    # A_evolution 形状: (Nz, Nx, Nt)
    Power = np.abs(A_evolution)**2
    
    # 分析轴上 (x=0) 的脉冲
    center_x_idx = sim.Nx // 2
    Power_on_axis = Power[:, center_x_idx, :]
    
    # 计算轴上频谱
    Spectrum_on_axis = np.abs(np.fft.fft(A_evolution[:, center_x_idx, :], axis=1))**2
    
    # 计算轴上FWHM
    fwhm_in = sim._compute_fwhm(Power_on_axis[0, :], sim.t)
    fwhm_out = sim._compute_fwhm(Power_on_axis[-1, :], sim.t)
    
    # 计算空间分布
    center_t_idx = sim.Nt // 2
    spatial_profile_in = Power[0, :, center_t_idx]
    spatial_profile_out = Power[-1, :, center_t_idx]
    
    results = {
        'power_on_axis': Power_on_axis,
        'spectrum_on_axis': Spectrum_on_axis,
        'spatial_profile_in': spatial_profile_in,
        'spatial_profile_out': spatial_profile_out,
        'peak_power_in': np.max(Power_on_axis[0, :]),
        'peak_power_out': np.max(Power_on_axis[-1, :]),
        'energy_in': sim._compute_energy_2d(A_evolution[0, :, :]),
        'energy_out': sim._compute_energy_2d(A_evolution[-1, :, :]),
        'fwhm_in': fwhm_in,
        'fwhm_out': fwhm_out,
    }
    
    return results


def plot_xt_map(sim, A, z_dist_mm, title):
    """绘制 (x, t) 二维热图"""
    plt.figure(figsize=(10, 6))
    extent = [sim.t[0]*1e12, sim.t[-1]*1e12, sim.x[0]*1e3, sim.x[-1]*1e3]
    plt.imshow(np.abs(A)**2, aspect='auto', origin='lower', extent=extent, cmap='jet')
    plt.colorbar(label='Intensity (arb. units)')
    plt.xlabel('Time (ps)')
    plt.ylabel('Position x (mm)')
    plt.title(f'{title} at z = {z_dist_mm:.2f} mm')
    plt.tight_layout()
    plt.show()


def plot_4d_scatter(sim, A_evolution, threshold_ratio=0.1, downsample_factor=4, title="4D (z, x, t, Intensity) Visualization"):
    """
    使用3D散点图可视化 (z, x, t) 空间中的强度分布。

    Args:
        sim: 仿真器实例，用于获取坐标轴信息。
        A_evolution (np.ndarray): 仿真的完整结果，形状为 (Nz, Nx, Nt)。
        threshold_ratio (float): 强度阈值比例。只绘制强度大于 (峰值强度 * 此比例) 的点。
        downsample_factor (int): 降采样因子，用于减少绘图点数，避免卡顿。
        title (str): 图表标题。
    """
    print(f"\n生成4D散点图... (阈值={threshold_ratio}, 降采样={downsample_factor})")
    
    # 1. 计算强度并降采样
    # 使用切片操作进行降采样，例如 [::4] 表示每隔4个点取一个
    power = np.abs(A_evolution[::downsample_factor, ::downsample_factor, ::downsample_factor])**2
    
    # 获取降采样后的坐标轴
    z_coords = sim.z[::downsample_factor]
    x_coords = sim.x[::downsample_factor]
    t_coords = sim.t[::downsample_factor]
    
    # 2. 设置强度阈值
    max_power = np.max(power)
    if max_power == 0:
        print("警告：最大强度为0，无法绘制4D散点图。")
        return
        
    threshold = max_power * threshold_ratio
    
    # 3. 找出所有强度高于阈值的点的索引
    z_indices, x_indices, t_indices = np.where(power > threshold)
    
    if len(z_indices) == 0:
        print(f"警告：在阈值 {threshold_ratio} 下没有找到足够强的点，请尝试降低阈值。")
        return
        
    # 4. 从索引获取物理坐标和对应的强度值
    z_points = z_coords[z_indices]
    x_points = x_coords[x_indices]
    t_points = t_coords[t_indices]
    intensity_values = power[z_indices, x_indices, t_indices]
    
    print(f"找到 {len(z_points)} 个点进行绘制。")

    # 5. 创建3D散点图
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection='3d')
    
    # 使用scatter函数绘制点，c参数用于颜色映射
    sc = ax.scatter(z_points * 1e3,  # z轴，单位转为mm
                    x_points * 1e3,  # x轴，单位转为mm
                    t_points * 1e15, # t轴，单位转为fs
                    c=intensity_values, 
                    cmap='viridis',      # 使用viridis颜色映射
                    marker='o',
                    s=5,               # 点的大小
                    alpha=0.6)         # 点的透明度

    # 设置坐标轴标签和标题
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel('Propagation Distance z (mm)')
    ax.set_ylabel('Position x (mm)')
    ax.set_zlabel('Time t (fs)')
    
    # 添加颜色条
    cbar = fig.colorbar(sc, shrink=0.6)
    cbar.set_label('Intensity (arb. units)')
    
    # 调整视角
    ax.view_init(elev=20, azim=-65)
    
    plt.tight_layout()
    plt.show() # 显示图形


def plot_results_2d(sim, A_evolution, save_path=None):
    """
    绘制 (2+1)D 主要仿真结果
    """
    results = analyze_results_2d(sim, A_evolution)
    
    fig = plt.figure(figsize=(16, 12))
    
    # 1. 输入脉冲的 (x, t) 分布
    ax1 = plt.subplot(2, 3, 1)
    extent_input = [sim.t[0]*1e12, sim.t[-1]*1e12, sim.x[0]*1e3, sim.x[-1]*1e3]
    im1 = ax1.imshow(np.abs(A_evolution[0, :, :])**2, aspect='auto', 
                     extent=extent_input, origin='lower', cmap='hot')
    ax1.set_xlabel('Time (ps)')
    ax1.set_ylabel('Position x (mm)')
    ax1.set_title('Input Pulse (x, t) Distribution', fontweight='bold')
    plt.colorbar(im1, ax=ax1, label='Intensity')
    
    # 2. 输出脉冲的 (x, t) 分布
    ax2 = plt.subplot(2, 3, 2)
    extent_output = [sim.t[0]*1e12, sim.t[-1]*1e12, sim.x[0]*1e3, sim.x[-1]*1e3]
    im2 = ax2.imshow(np.abs(A_evolution[-1, :, :])**2, aspect='auto', 
                     extent=extent_output, origin='lower', cmap='hot')
    ax2.set_xlabel('Time (ps)')
    ax2.set_ylabel('Position x (mm)')
    ax2.set_title('Output Pulse (x, t) Distribution', fontweight='bold')
    plt.colorbar(im2, ax=ax2, label='Intensity')
    
    # 3. 轴上时间演化
    ax3 = plt.subplot(2, 3, 3)
    extent_axis = [sim.t[0]*1e12, sim.t[-1]*1e12, 0, sim.z[-1]*1e3]
    im3 = ax3.imshow(results['power_on_axis'], aspect='auto', 
                     extent=extent_axis, origin='lower', cmap='viridis')
    ax3.set_xlabel('Time (ps)')
    ax3.set_ylabel('Distance (mm)')
    ax3.set_title('On-Axis Temporal Evolution', fontweight='bold')
    plt.colorbar(im3, ax=ax3, label='Power (W)')
    
    # 4. 轴上频谱对比
    ax4 = plt.subplot(2, 3, 4)
    
    # 使用 fftshift 来获得以0为中心的频率轴
    freq = np.fft.fftshift(np.fft.fftfreq(sim.Nt, sim.dt))
    wavelength = 2 * np.pi * sim.c / (sim.omega0 + 2*np.pi*freq) * 1e9
    
    # 同样对频谱数据进行 fftshift
    spec_in = np.fft.fftshift(results['spectrum_on_axis'][0, :])
    spec_out = np.fft.fftshift(results['spectrum_on_axis'][-1, :])
    
    # 归一化
    spec_in_norm = spec_in / np.max(spec_in)
    spec_out_norm = spec_out / np.max(spec_out)
    
    # 对波长排序以确保绘图正确
    sort_idx = np.argsort(wavelength)
    ax4.plot(wavelength[sort_idx], spec_in_norm[sort_idx], 'b-', linewidth=2, label='Input Spectrum')
    ax4.plot(wavelength[sort_idx], spec_out_norm[sort_idx], 'r--', linewidth=2, label='Output Spectrum')
    
    ax4.set_xlabel('Wavelength (nm)')
    ax4.set_ylabel('Normalized Intensity')
    ax4.set_title('On-Axis Spectral Comparison', fontweight='bold')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    # 设置一个合理的波长显示范围
    center_wl = sim.lambda0 * 1e9
    ax4.set_xlim(center_wl - 100, center_wl + 100)
    
    # 5. 空间分布对比
    ax5 = plt.subplot(2, 3, 5)
    ax5.plot(sim.x*1e3, results['spatial_profile_in'], 'b-', linewidth=2, label='Input')
    ax5.plot(sim.x*1e3, results['spatial_profile_out'], 'r--', linewidth=2, label='Output')
    ax5.set_xlabel('Position x (mm)')
    ax5.set_ylabel('Intensity')
    ax5.set_title('Spatial Profile Comparison', fontweight='bold')
    ax5.legend()
    ax5.grid(True, alpha=0.3)
    
    # 6. 轴上时间对比
    ax6 = plt.subplot(2, 3, 6)
    ax6.plot(sim.t*1e12, results['power_on_axis'][0, :], 'b-', linewidth=2, label='Input')
    ax6.plot(sim.t*1e12, results['power_on_axis'][-1, :], 'r--', linewidth=2, label='Output')
    ax6.set_xlabel('Time (ps)')
    ax6.set_ylabel('Power (W)')
    ax6.set_title('On-Axis Temporal Comparison', fontweight='bold')
    ax6.legend()
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"(2+1)D 图像已保存: {save_path}")
    
    plt.show()
    
    # 打印结果
    print(f"\n{'='*60}")
    print("(2+1)D 仿真结果:")
    print(f"{'='*60}")
    print(f"  输入: {results['peak_power_in']:.1f} W | {results['energy_in']*1e6:.1f} µJ | {results['fwhm_in']*1e12:.2f} ps")
    print(f"  输出: {results['peak_power_out']:.1f} W | {results['energy_out']*1e6:.1f} µJ | {results['fwhm_out']*1e12:.2f} ps")
    print(f"  能量守恒: {results['energy_out']/results['energy_in']*100:.1f}%")
    
    # 计算光束宽度变化
    x_rms_in = np.sqrt(np.trapz(results['spatial_profile_in'] * sim.x**2, sim.x) / 
                       np.trapz(results['spatial_profile_in'], sim.x))
    x_rms_out = np.sqrt(np.trapz(results['spatial_profile_out'] * sim.x**2, sim.x) / 
                        np.trapz(results['spatial_profile_out'], sim.x))
    
    print(f"  光束宽度变化: {x_rms_in*1e3:.2f} mm -> {x_rms_out*1e3:.2f} mm")
    print(f"{'='*60}\n")


def plot_evolution_slices(sim, A_evolution):
    """
    绘制脉冲沿传播距离 z 的演化切片图。
    1. (z, t) 图: 沿光束中心 (x=0) 的时间演化。
    2. (z, x) 图: 沿脉冲中心 (t=0) 的空间演化。
    """
    print("正在生成演化切片图 (z-t 和 z-x)...")
    
    Power = np.abs(A_evolution)**2
    center_x_idx = sim.Nx // 2
    center_t_idx = sim.Nt // 2

    # 提取切片数据
    power_zt = Power[:, center_x_idx, :]  # (Nz, Nt)
    power_zx = Power[:, :, center_t_idx]  # (Nz, Nx)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig.suptitle('Pulse Evolution Slices', fontsize=16, fontweight='bold')

    # 1. z-t 图 (时间演化)
    extent_zt = [sim.t[0]*1e12, sim.t[-1]*1e12, sim.z[0]*1e3, sim.z[-1]*1e3]
    im1 = ax1.imshow(power_zt, aspect='auto', origin='lower', extent=extent_zt, cmap='hot')
    ax1.set_xlabel('Time (ps)', fontsize=12)
    ax1.set_ylabel('Propagation Distance z (mm)', fontsize=12)
    ax1.set_title('On-Axis Temporal Evolution (x=0)', fontweight='bold')
    fig.colorbar(im1, ax=ax1, label='Intensity (arb. units)')

    # 2. z-x 图 (空间演化)
    extent_zx = [sim.x[0]*1e3, sim.x[-1]*1e3, sim.z[0]*1e3, sim.z[-1]*1e3]
    im2 = ax2.imshow(power_zx, aspect='auto', origin='lower', extent=extent_zx, cmap='hot')
    ax2.set_xlabel('Position x (mm)', fontsize=12)
    ax2.set_ylabel('Propagation Distance z (mm)', fontsize=12)
    ax2.set_title('Peak-Time Spatial Evolution (t=0)', fontweight='bold')
    fig.colorbar(im2, ax=ax2, label='Intensity (arb. units)')

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_1d_metrics(sim, A_evolution):
    """
    计算并绘制关键一维物理量随传播距离 z 的演化。
    1. 峰值功率
    2. 总能量
    3. 时间脉宽 (FWHM)
    4. 空间束宽 (FWHM)
    """
    print("正在生成一维简化度量图...")
    
    Nz = sim.Nz
    z_axis = sim.z * 1e3 # z轴，单位 mm

    # 初始化存储数组
    peak_power = np.zeros(Nz)
    total_energy = np.zeros(Nz)
    temporal_fwhm = np.zeros(Nz)
    spatial_fwhm = np.zeros(Nz)

    Power = np.abs(A_evolution)**2
    center_x_idx = sim.Nx // 2
    center_t_idx = sim.Nt // 2

    # 循环遍历每个z步来计算指标
    for i in range(Nz):
        A_slice = A_evolution[i, :, :]
        Power_slice = Power[i, :, :]
        
        # 1. 峰值功率
        peak_power[i] = np.max(Power_slice)
        
        # 2. 总能量
        total_energy[i] = sim._compute_energy_2d(A_slice)
        
        # 3. 轴上时间脉宽 FWHM
        power_on_axis_t = Power_slice[center_x_idx, :]
        temporal_fwhm[i] = sim._compute_fwhm(power_on_axis_t, sim.t) * 1e12 # ps
        
        # 4. 中心时刻空间束宽 FWHM
        power_at_center_t = Power_slice[:, center_t_idx]
        spatial_fwhm[i] = sim._compute_fwhm(power_at_center_t, sim.x) * 1e3 # mm

    # 绘图
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    fig.suptitle('1D Simplified Metrics vs. Propagation Distance', fontsize=16, fontweight='bold')
    
    # 归一化峰值功率
    axes[0, 0].plot(z_axis, peak_power / peak_power[0], 'r-', linewidth=2)
    axes[0, 0].set_title('Normalized Peak Power', fontweight='bold')
    axes[0, 0].set_xlabel('Distance z (mm)')
    axes[0, 0].set_ylabel('P(z) / P(0)')
    axes[0, 0].grid(True, alpha=0.4)

    # 归一化总能量
    axes[0, 1].plot(z_axis, total_energy / total_energy[0], 'b-', linewidth=2)
    axes[0, 1].set_title('Normalized Total Energy', fontweight='bold')
    axes[0, 1].set_xlabel('Distance z (mm)')
    axes[0, 1].set_ylabel('E(z) / E(0)')
    axes[0, 1].grid(True, alpha=0.4)
    axes[0, 1].set_ylim(bottom=max(0, 2 * np.min(total_energy / total_energy[0]) - 1)) # 调整Y轴范围

    # 时间 FWHM
    axes[1, 0].plot(z_axis, temporal_fwhm, 'g-', linewidth=2)
    axes[1, 0].set_title('On-Axis Temporal FWHM', fontweight='bold')
    axes[1, 0].set_xlabel('Distance z (mm)')
    axes[1, 0].set_ylabel('FWHM (ps)')
    axes[1, 0].grid(True, alpha=0.4)

    # 空间 FWHM
    axes[1, 1].plot(z_axis, spatial_fwhm, 'm-', linewidth=2)
    axes[1, 1].set_title('Peak-Time Spatial FWHM', fontweight='bold')
    axes[1, 1].set_xlabel('Distance z (mm)')
    axes[1, 1].set_ylabel('FWHM (mm)')
    axes[1, 1].grid(True, alpha=0.4)

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.show()


def plot_beam_evolution(sim, A_evolution, save_path=None):
    """绘制光束演化过程"""
    fig = plt.figure(figsize=(12, 8))
    
    # 选择几个关键位置
    z_positions = [0, sim.Nz//4, sim.Nz//2, 3*sim.Nz//4, sim.Nz-1]
    colors = ['blue', 'green', 'orange', 'red', 'purple']
    labels = ['z=0', f'z={sim.z[sim.Nz//4]*1e3:.1f}mm', 
              f'z={sim.z[sim.Nz//2]*1e3:.1f}mm', 
              f'z={sim.z[3*sim.Nz//4]*1e3:.1f}mm', 
              f'z={sim.z[-1]*1e3:.1f}mm']
    
    # 1. 空间分布演化
    ax1 = plt.subplot(1, 2, 1)
    center_t_idx = sim.Nt // 2
    for i, (pos, color, label) in enumerate(zip(z_positions, colors, labels)):
        spatial_profile = np.abs(A_evolution[pos, :, center_t_idx])**2
        ax1.plot(sim.x*1e3, spatial_profile, color=color, linewidth=2, label=label)
    
    ax1.set_xlabel('Position x (mm)')
    ax1.set_ylabel('Intensity')
    ax1.set_title('Spatial Profile Evolution', fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. 轴上时间分布演化
    ax2 = plt.subplot(1, 2, 2)
    center_x_idx = sim.Nx // 2
    for i, (pos, color, label) in enumerate(zip(z_positions, colors, labels)):
        temporal_profile = np.abs(A_evolution[pos, center_x_idx, :])**2
        ax2.plot(sim.t*1e12, temporal_profile, color=color, linewidth=2, label=label)
    
    ax2.set_xlabel('Time (ps)')
    ax2.set_ylabel('Power (W)')
    ax2.set_title('On-Axis Temporal Evolution', fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"光束演化图像已保存: {save_path}")
    
    plt.show()
