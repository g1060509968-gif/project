"""
高斯脉冲测试脚本 - 使用 ERK43IP_FullDispersion 求解器
基于 RK4IP 模块，参考 PHASE+FIXED 版本的结构
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 导入 RK4IP 模块
from erk43ip_method import ERK43IP_FullDispersion
from visualization import plot_results, pulse_width

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def test_gaussian_pulse():
    """测试高斯脉冲传播"""
    print("=" * 70)
    print("高斯脉冲传播测试 - ERK43IP 求解器")
    print("=" * 70)

    # 1. 设置仿真参数
    crystal_length = 1182e-3  # 3.168m 晶体长度
    time_window = 50e-12   # 50ps 时间窗口
    num_t_points = 4096    # 时间采样点数
    center_wavelength = 1064e-9  # 中心波长
    beam_radius = 0.35e-3   # 光斑半径
    
    print(f"\n[1/6] 初始化 ERK43IP 求解器...")
    
    # 2. 初始化 ERK43IP 求解器
    solver = ERK43IP_FullDispersion(
        material='fused_silica',
        n2=2.7e-20,           # 非线性折射率系数 (m²/W)
        beam_radius=beam_radius,
        center_wavelength=center_wavelength,
        use_raman=True,      # 关闭拉曼效应
        use_self_steepening=True  # 开启自陡峭效应
    )
    
    print(f"  ✅ 求解器初始化成功")
    print(f"    非线性系数 gamma: {solver.gamma:.6f} W^-1 m^-1")
    print(f"    中心波长: {center_wavelength*1e9:.1f} nm")
    print(f"    光斑半径: {beam_radius*1e6:.1f} um")

    # 3. 生成时间网格
    t = np.linspace(-time_window/2, time_window/2, num_t_points)
    dt = t[1] - t[0]
    
    print(f"\n[2/6] 生成理想高斯脉冲...")
    
    # 4. 生成理想高斯脉冲
    pulse_energy = 2000e-6     # 200 µJ 脉冲能量
    pulse_width_fwhm = 10e-12 # 10ps 脉冲宽度 (FWHM)
    chirp_parameter = 0       # 啁啾参数 (0表示变换受限脉冲)
    
    try:
        A_initial = solver.generate_gaussian_pulse(
            t=t,
            pulse_energy=pulse_energy,
            pulse_fwhm=pulse_width_fwhm,
            chirp=chirp_parameter
        )
        
        # 计算脉冲参数
        initial_power = np.abs(A_initial)**2
        peak_power = np.max(initial_power)
        pulse_energy_calc = np.trapz(initial_power, t)
        pulse_width_calc = pulse_width([A_initial], t*1e12)[0]  # 转换为ps
        
        print(f"  ✅ 高斯脉冲生成成功")
        print(f"    脉冲能量: {pulse_energy_calc*1e6:.2f} µJ (目标: {pulse_energy*1e6:.2f} µJ)")
        print(f"    峰值功率: {peak_power/1e3:.1f} kW")
        print(f"    脉冲宽度: {pulse_width_calc:.2f} ps (FWHM)")
        print(f"    啁啾参数: C = {chirp_parameter}")
        
    except Exception as e:
        print(f"  ❌ 高斯脉冲生成失败: {e}")
        return

    # 5. 运行传播仿真
    print(f"\n[3/6] 运行传播仿真...")
    try:
        z_array, A_evolution, omega = solver.propagate(
            A_initial, t, L=crystal_length, 
            tol=1e-5, max_step=1e-2
        )
        print(f"  ✅ 传播仿真成功")
        print(f"    传播距离: {crystal_length:.3f} m")
        print(f"    步数: {len(z_array)}")
        print(f"    最终位置: {z_array[-1]:.3f} m")
        
    except Exception as e:
        print(f"  ❌ 传播仿真失败: {e}")
        return

    # 6. 创建适配器对象用于可视化
    print(f"\n[4/6] 准备可视化数据...")
    
    class SimAdapter:
        """适配器类，用于兼容 visualization 模块"""
        def __init__(self, z_array, t, A_evolution, solver):
            self.z = z_array
            self.t = t
            self.A_evolution = A_evolution
            self.Nt = len(t)
            self.dt = t[1] - t[0]
            self.c = solver.c
            self.omega0 = solver.omega0
            self.lambda0 = solver.lambda0
    
    sim_adapter = SimAdapter(z_array, t, A_evolution, solver)

    # 7. 绘制结果
    print(f"\n[5/6] 绘制仿真结果...")
    try:
        plot_results(sim_adapter, A_evolution, 
                    save_path="erk43ip_gaussian_pulse_results.png")
        print("  ✅ 结果绘图成功")
    except Exception as e:
        print(f"  ❌ 结果绘图失败: {e}")
        # 如果标准绘图失败，使用简化绘图
        plot_simple_results(sim_adapter, A_evolution)

    # 8. 详细结果分析
    print(f"\n[6/6] 详细结果分析...")
    analyze_results(sim_adapter, A_evolution, solver)


    # ... 仿真结束后 ...
    A_out = A_evolution[-1]

    # 计算残差 
    # threshold=0.01 表示只看 >1% 强度的部分
    res = solver.analyze_spectral_phase(A_out, t, fit_order=1, plot_threshold=0.01)

    if res:
        import matplotlib.pyplot as plt
        wl = res['wavelength_nm']
        spec = res['spectral_intensity'] # 线性归一化强度
        resid = res['residual_phase']
        mask = res['mask']
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # 左轴：画光谱 (线性)
        color1 = 'tab:blue'
        ax1.plot(wl, spec, color=color1, label='Normalized Spectrum')
        ax1.set_ylabel('Normalized Intensity', color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_ylim([0, 1.05]) # 线性刻度 0到1
        
        # 设置波长显示范围 (自动聚焦在有效区域附近)
        valid_wl = wl[mask]
        if len(valid_wl) > 0:
            span = (valid_wl.max() - valid_wl.min()) * 1.5
            center = (valid_wl.max() + valid_wl.min()) / 2
            ax1.set_xlim([center - span/2, center + span/2])
        
        # 右轴：画相位残差
        ax2 = ax1.twinx()
        color2 = 'tab:red'
        # 只画 mask 区域内的残差，显得干净
        ax2.plot(wl[mask], resid[mask], color=color2, linewidth=2, label='Phase Residual')
        ax2.set_ylabel('Phase Residual (rad)', color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)
        
        plt.title('Spectral Phase Analysis')
        plt.show()
    
    # =================================================================
    # 新增：参数表功能测试 (使用 SimResultAdapter)
    # =================================================================
    print(f"\n{'='*70}")
    print("参数表功能测试 - 使用 SimResultAdapter")
    print(f"{'='*70}")
    
    try:
        # 导入新增的类和函数
        from erk43ip_method import SimResultAdapter
        from visualization import draw_parameter_table
        
        # 1. 创建适配器 (将 ERK43IP 的分散数据打包)
        adapter = SimResultAdapter(solver, z_array, t, A_evolution)
        
        # 2. 准备脉冲参数字典
        # 这里的参数最好是实际测量值，或者你设定的初始值
        pulse_params = {
            'energy': pulse_energy,      # 你的设定能量
            'fwhm': pulse_width_fwhm,    # 你的设定脉宽
            'peak_power': np.max(np.abs(A_evolution[0])**2), # 实际计算的峰值功率
            'chirp': 0              # 你的设定 chirp
        }
        
        # 3. 绘制表格
        print("正在绘制仿真参数表...")
        draw_parameter_table(adapter, pulse_params, save_path="simulation_parameter_report.png")
        print("  ✅ 参数表已保存为 simulation_parameter_report.png")
        
    except ImportError as e:
        print(f"  ⚠️ 导入失败: {e}")
        print("  请确保已正确添加 SimResultAdapter 类和 draw_parameter_table 函数")
    except Exception as e:
        print(f"  ⚠️ 参数表功能测试失败: {e}")
        import traceback
        traceback.print_exc()
 

def plot_simple_results(sim, A_evolution):
    """简化结果绘图"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 时域演化
    Power = np.abs(A_evolution)**2
    ax = axes[0, 0]
    extent = (sim.t[0]*1e12, sim.t[-1]*1e12, 0, sim.z[-1])
    im = ax.imshow(Power, aspect='auto', extent=extent,
                  origin='lower', cmap='hot')
    ax.set_xlabel('Time (ps)')
    ax.set_ylabel('Distance (m)')
    ax.set_title('Temporal Evolution')
    plt.colorbar(im, ax=ax, label='Power (W)')
    
    # 输入输出对比
    ax = axes[0, 1]
    ax.plot(sim.t*1e12, Power[0, :], 'b-', linewidth=2, label='Input')
    ax.plot(sim.t*1e12, Power[-1, :], 'r--', linewidth=2, label='Output')
    ax.set_xlabel('Time (ps)')
    ax.set_ylabel('Power (W)')
    ax.set_title('Pulse Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 频谱对比
    ax = axes[1, 0]
    from scipy.fft import fft, fftfreq, fftshift
    freq = fftfreq(sim.Nt, sim.dt)
    wavelength = 2 * np.pi * sim.c / (sim.omega0 + 2*np.pi*freq) * 1e9
    
    spec_in = np.abs(fft(A_evolution[0, :]))**2
    spec_out = np.abs(fft(A_evolution[-1, :]))**2
    
    ax.plot(wavelength, spec_in/np.max(spec_in), 'b-', label='Input')
    ax.plot(wavelength, spec_out/np.max(spec_out), 'r--', label='Output')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_ylabel('Normalized Spectrum')
    ax.set_title('Spectral Comparison')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([sim.lambda0*1e9 - 10, sim.lambda0*1e9 + 10])
    
    # 峰值功率演化
    ax = axes[1, 1]
    peak_power = np.max(Power, axis=1)
    ax.plot(sim.z, peak_power/1e3, 'g-', linewidth=2)
    ax.set_xlabel('Propagation Distance (m)')
    ax.set_ylabel('Peak Power (kW)')
    ax.set_title('Peak Power Evolution')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig("erk43ip_simple_results.png", dpi=150)
    plt.show()

def analyze_results(sim, A_evolution, solver):
    """分析仿真结果"""
    print(f"\n{'='*70}")
    print("详细结果分析")
    print(f"{'='*70}")
    
    # 计算输入输出参数
    Power = np.abs(A_evolution)**2
    Spectrum = np.abs(np.fft.fft(A_evolution, axis=1))**2
    
    # 输入参数
    power_in = Power[0, :]
    peak_power_in = np.max(power_in)
    energy_in = np.trapz(power_in, sim.t)
    fwhm_in = pulse_width([A_evolution[0, :]], sim.t*1e12)[0]
    
    # 输出参数
    power_out = Power[-1, :]
    peak_power_out = np.max(power_out)
    energy_out = np.trapz(power_out, sim.t)
    fwhm_out = pulse_width([A_evolution[-1, :]], sim.t*1e12)[0]
    
    # 频谱展宽分析
    spectrum_in = Spectrum[0, :]
    spectrum_out = Spectrum[-1, :]
    
    # 计算频谱宽度 (FWHM)
    freq = np.fft.fftfreq(sim.Nt, sim.dt)
    wavelength = 2 * np.pi * sim.c / (sim.omega0 + 2*np.pi*freq) * 1e9
    
    def spectral_fwhm(wavelength, spectrum):
        max_val = np.max(spectrum)
        indices = np.where(spectrum >= max_val/2)[0]
        if len(indices) > 1:
            return wavelength[indices[-1]] - wavelength[indices[0]]
        return 0
    
    spectral_width_in = spectral_fwhm(wavelength, spectrum_in)
    spectral_width_out = spectral_fwhm(wavelength, spectrum_out)
    
    print(f"输入脉冲:")
    print(f"  能量: {energy_in*1e6:.2f} µJ")
    print(f"  峰值功率: {peak_power_in/1e3:.1f} kW")
    print(f"  脉冲宽度: {fwhm_in:.2f} ps (FWHM)")
    print(f"  频谱宽度: {spectral_width_in:.2f} nm (FWHM)")
    
    print(f"\n输出脉冲:")
    print(f"  能量: {energy_out*1e6:.2f} µJ (守恒率: {energy_out/energy_in*100:.1f}%)")
    print(f"  峰值功率: {peak_power_out/1e3:.1f} kW")
    print(f"  脉冲宽度: {fwhm_out:.2f} ps (FWHM)")
    print(f"  频谱宽度: {spectral_width_out:.2f} nm (FWHM)")
    
    print(f"\n非线性效应:")
    print(f"  频谱展宽因子: {spectral_width_out/spectral_width_in:.2f}")
    print(f"  脉冲展宽因子: {fwhm_out/fwhm_in:.2f}")
    print(f"  峰值功率变化: {peak_power_out/peak_power_in:.2f}")
    
    # 非线性长度估算
    if solver.gamma > 0:
        L_NL = 1 / (solver.gamma * peak_power_in)
        print(f"  非线性长度 L_NL: {L_NL*1e3:.1f} mm")
        print(f"  传播距离/L_NL: {sim.z[-1]/L_NL:.1f}")
    
    print(f"{'='*70}\n")


if __name__ == "__main__":
    # 运行主测试
    test_gaussian_pulse()
