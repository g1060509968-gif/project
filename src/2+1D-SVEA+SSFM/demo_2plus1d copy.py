"""
(2+1)D 仿真器演示脚本
展示修复后的 (2+1)D 仿真器的完整功能，并包含参数验证。
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# 将项目根目录添加到 Python 路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from simulator_core import NonlinearCrystalSimulator
from visualization_2d import plot_xt_map, plot_evolution_slices, plot_1d_metrics, plot_4d_scatter

def validate_parameters_for_spatial_effects(params):
    """
    验证参数是否有利于观察空间效应（如自聚焦）。

    Args:
        params (dict): 包含所有仿真参数的字典。
    """
    print("-" * 20)
    print("开始 (2+1)D 参数验证...")

    # 解包参数
    p = params
    n0, n2 = p['n0'], p['n2']
    lambda0 = p['central_wavelength']
    peak_power = p['peak_power']
    x_fwhm = p['x_fwhm']
    x_window = p['x_window']
    num_z_steps = p['num_z_steps']
    crystal_length = p['crystal_length']

    # 1. 检查 n2 是否足以引起自聚焦
    if n2 <= 0:
        print(f"[警告] 非线性折射率 n2 = {n2:.2e} <= 0。不会发生自聚焦现象。")
        return # 后续检查无意义

    # 2. 计算临界自聚焦功率 P_cr 并与峰值功率比较
    # P_cr = (alpha * lambda^2) / (4 * pi * n0 * n2), alpha for Gaussian beam is ~3.77
    alpha = 3.77 
    P_cr = (alpha * lambda0**2) / (4 * np.pi * n0 * n2)
    
    print(f"  [信息] 估算的临界自聚焦功率 P_cr ≈ {P_cr/1e6:.4f} MW")
    print(f"  [信息] 输入的脉冲峰值功率 P_peak = {peak_power/1e6:.4f} MW")

    if peak_power < P_cr:
        print(f"[警告] 峰值功率 P_peak 低于临界功率 P_cr。")
        print("       预期结果：衍射将占主导，光斑会发散，不会看到明显的自聚焦。")
        print("       建议：提高 'peak_power' 或 'n2'。")
    elif peak_power > 10 * P_cr:
        print(f"[警告] 峰值功率 P_peak ({peak_power/P_cr:.1f} 倍 P_cr) 非常高。")
        print("       预期结果：可能会发生剧烈的、灾变性的自聚焦，对数值稳定性要求极高。")
        print("       建议：确保 'num_z_steps' 和 'num_x_points' 足够大以解析剧烈变化。")
    else:
        print(f"[成功] 峰值功率 P_peak ({peak_power/P_cr:.1f} 倍 P_cr) 合理，适合观察自聚焦。")

    # 3. 检查空间网格是否足够大
    if x_window < 4 * x_fwhm:
        print(f"[警告] 空间窗口 x_window ({x_window*1e3:.2f} mm) 可能太小，不足以容纳初始光斑 ({x_fwhm*1e3:.2f} mm)。")
        print("       衍射的能量可能会到达边界，产生非物理的反射伪影。")
        print(f"       建议：将 'x_window' 增加到至少 {4 * x_fwhm * 1e3:.2f} mm。")
    else:
        print(f"[成功] 空间窗口 x_window 足够大 ( > 4 * x_fwhm )。")

    # 4. 检查传播步长 dz 是否足够小
    # 瑞利长度 z_R = pi * w0^2 / lambda, w0 = x_fwhm / (2*sqrt(ln(2)))
    w0 = x_fwhm / (2 * np.sqrt(np.log(2)))
    z_R = np.pi * w0**2 * n0 / lambda0
    
    # 非线性长度 L_NL = 1 / (k0 * n2 * I0), I0 = 2*P_peak/(pi*w0^2)
    I0 = 2 * peak_power / (np.pi * w0**2)
    k0 = 2 * np.pi / lambda0
    L_NL = 1 / (k0 * n2 * I0)
    
    dz = crystal_length / num_z_steps
    
    print(f"  [信息] 衍射特征长度 (瑞利长度) z_R ≈ {z_R*100:.4f} cm")
    print(f"  [信息] 非线性特征长度 L_NL ≈ {L_NL*100:.4f} cm")
    print(f"  [信息] 仿真步长 dz = {dz*100:.4f} cm")

    if dz > z_R / 20 or dz > L_NL / 20:
        print("[警告] 仿真步长 dz 可能过大，无法精确解析衍射或非线性效应。")
        print(f"       建议：将 'num_z_steps' 增加到 > {int(crystal_length / min(z_R, L_NL) * 20)}。")
    else:
        print("[成功] 仿真步长 dz 足够小，可以很好地解析物理过程。")
        
    print("参数验证结束。")
    print("-" * 20)


def demo_2plus1d():
    """运行一个 (2+1)D 仿真并显示结果"""
    
    # --- 1. 定义仿真参数 ---
    # 将所有参数放入一个字典中，方便传递
    params = {
        # 晶体和传播参数
        'crystal_length': 0.05,      # 晶体长度 (m), 5 cm
        'num_z_steps': 500,          # 传播步数
        'n0': 1.5,                   # 线性折射率 (近似值)
        'n2': 2.5e-20,               # 非线性折射率 (m^2/W), 这是自聚焦的关键
        'gvd': 0,                    # 群速度色散 (s^2/m), 暂时忽略
        
        # 时间网格参数
        'time_window': 200e-15,      # 时间窗口 (s)
        'num_t_points': 512,         # 时间点数
        
        # 空间网格参数 (x-维度)
        'x_window': 2e-3,            # 空间窗口 (m), 2 mm
        'num_x_points': 512,         # 空间点数
        
        # 初始脉冲参数
        'central_wavelength': 1030e-9, # 中心波长 (m)
        'peak_power': 4e6,           # 峰值功率 (W), 4 MW
        't_fwhm': 50e-15,            # 时间脉宽 (s), FWHM
        'x_fwhm': 0.5e-3,            # 空间束宽 (m), FWHM, 0.5 mm
    }

    # --- 2. 运行参数验证 ---
    validate_parameters_for_spatial_effects(params)

    # --- 3. 初始化仿真器 ---
    sim = NonlinearCrystalSimulator(
        crystal_length=params['crystal_length'],
        num_z_steps=params['num_z_steps'],
        time_window=params['time_window'],
        num_t_points=params['num_t_points'],
        x_window=params['x_window'],
        num_x_points=params['num_x_points'],
        center_wavelength=params['central_wavelength'],
        material='fused_silica',
        dispersion_mode='sellmeier'
    )
    
    # 设置非线性参数
    sim.set_parameters(n2=params['n2'])

    # --- 4. 创建初始脉冲 ---
    # (t, x) 网格上的高斯脉冲
    # 计算脉冲能量：能量 = 峰值功率 * 脉宽 * sqrt(pi/(4*ln(2)))
    pulse_energy = params['peak_power'] * params['t_fwhm'] * np.sqrt(np.pi/(4*np.log(2)))
    A0 = sim.gaussian_pulse(
        pulse_energy=pulse_energy,
        pulse_width_fwhm=params['t_fwhm'],
        beam_waist_x=params['x_fwhm'] / (2 * np.sqrt(np.log(2)))  # 转换为束腰半径
    )

    # --- 5. 运行仿真 ---
    print("\n开始 (2+1)D 仿真...")
    A_evolution = sim.propagate(A0)
    print("仿真完成！")

    # --- 6. 可视化结果 ---
    print("生成可视化结果...")
    
    # 绘制输入和输出的 (x, t) 能量分布图
    plot_xt_map(sim, A_evolution[0, :, :], 0, "Input Pulse (x, t) Distribution")
    plot_xt_map(sim, A_evolution[-1, :, :], sim.z[-1]*1e3, "Output Pulse (x, t) Distribution")
    
    # 绘制 z-t 和 z-x 演化图
    plot_evolution_slices(sim, A_evolution)
    
    # 绘制 1D 关键指标随 z 的演化
    plot_1d_metrics(sim, A_evolution)
    
    # 绘制 4D 散点图可视化
    plot_4d_scatter(sim, A_evolution, threshold_ratio=0.05, downsample_factor=4)
    
    plt.show()


if __name__ == '__main__':
    demo_2plus1d()
