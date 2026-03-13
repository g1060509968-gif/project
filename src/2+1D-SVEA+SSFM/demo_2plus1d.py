"""
(2+1)D 仿真器演示脚本
展示修复后的 (2+1)D 仿真器的完整功能
"""

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(__file__))

from simulator_core import NonlinearCrystalSimulator
# 从 visualization_2d 导入新的可视化函数
from visualization_2d import plot_evolution_slices, plot_1d_metrics, plot_xt_map, plot_4d_scatter

def demo_2plus1d():
    """演示 (2+1)D 仿真器的完整功能"""
    print("=" * 70)
    print("(2+1)D 仿真器完整功能演示")
    print("=" * 70)
    
    # 设置仿真参数
    crystal_length = 50e-3    # 50mm
    num_z_steps = 200         # 200步
    time_window = 50e-12      # 50ps
    num_t_points = 256        # 256点
    x_window = 10e-3          # 10mm 空间窗口
    num_x_points = 128        # 128个空间采样点
    
    print(f"\n初始化 (2+1)D 仿真器...")
    sim = NonlinearCrystalSimulator(
        crystal_length=crystal_length, num_z_steps=num_z_steps,
        time_window=time_window, num_t_points=num_t_points,
        x_window=x_window, num_x_points=num_x_points,
        center_wavelength=1064e-9, material='fused_silica'
    )
    
    # 设置非线性参数
    # n2值较高，以观察明显的非线性效应
    sim.set_parameters(n2=2.6e-20, alpha=0, self_steepening=True)

    # 生成 (2+1)D 高斯脉冲
    print(f"\n生成 (2+1)D 高斯脉冲...")
    pulse_energy = 2e-3     # 100 µJ
    beam_waist_x = 0.3e-3     # 0.5mm 光束腰斑
    
    A_initial = sim.gaussian_pulse(
        pulse_energy=pulse_energy,
        pulse_width_fwhm=1e-12,  # 10ps 脉冲宽度
        beam_waist_x=beam_waist_x,
        C=0
    )
    
    # 显示初始脉冲的 (x, t) 分布
    print(f"\n显示初始脉冲的 (x, t) 分布...")
    plot_xt_map(sim, A_initial, 0, "Input Pulse")
    
    # 运行 (2+1)D 传播仿真
    print(f"\n运行 (2+1)D 传播仿真...")
    A_evolution = sim.propagate(A_initial, verbose=True)
    print("✓ 仿真完成")

    # --- 新的可视化调用 ---
    print(f"\n开始生成可视化结果...")

    # 显示输出脉冲的 (x, t) 分布
    print(f"\n显示输出脉冲的 (x, t) 分布...")
    plot_xt_map(sim, A_evolution[-1, :, :], crystal_length * 1e3, "Output Pulse")

    # 1. 绘制新的演化切片图
    plot_evolution_slices(sim, A_evolution)
    
    # 2. 绘制新的1D简化数据图
    plot_1d_metrics(sim, A_evolution)
    
    # 3. 绘制4D散点图可视化
    plot_4d_scatter(sim, A_evolution, threshold_ratio=0.05, downsample_factor=4)

    print(f"\n{'='*70}")
    print("(2+1)D 仿真器演示完成！")
    print(f"{'='*70}")

if __name__ == "__main__":
    demo_2plus1d()
