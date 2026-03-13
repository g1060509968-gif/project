"""
测试4D可视化功能
"""

import numpy as np
import sys
import os

sys.path.append(os.path.dirname(__file__))

from simulator_core import NonlinearCrystalSimulator
from visualization_2d import plot_4d_scatter

def test_4d_visualization():
    """测试4D可视化功能"""
    print("测试4D可视化功能...")
    
    # 创建一个简单的仿真器实例用于测试
    sim = NonlinearCrystalSimulator(
        crystal_length=10e-3,    # 10mm
        num_z_steps=50,          # 50步
        time_window=20e-12,      # 20ps
        num_t_points=64,         # 64点
        x_window=5e-3,           # 5mm
        num_x_points=64,         # 64点
        center_wavelength=1064e-9,
        material='fused_silica'
    )
    
    # 创建一个简单的测试数据
    print("创建测试数据...")
    A_evolution = np.zeros((sim.Nz, sim.Nx, sim.Nt), dtype=complex)
    
    # 创建一个简单的高斯脉冲在时空中的演化
    center_z = sim.Nz // 2
    center_x = sim.Nx // 2
    center_t = sim.Nt // 2
    
    # 在中心位置创建一个高斯脉冲
    for i in range(sim.Nz):
        # 随着传播，脉冲会稍微变化
        z_factor = 1.0 - 0.5 * (i - center_z)**2 / center_z**2
        for j in range(sim.Nx):
            x_factor = np.exp(-((j - center_x) / 10)**2)
            for k in range(sim.Nt):
                t_factor = np.exp(-((k - center_t) / 15)**2)
                A_evolution[i, j, k] = z_factor * x_factor * t_factor * (1 + 0.1j)
    
    print(f"测试数据形状: {A_evolution.shape}")
    
    # 测试4D可视化
    print("调用 plot_4d_scatter 函数...")
    try:
        plot_4d_scatter(sim, A_evolution, threshold_ratio=0.1, downsample_factor=2)
        print("✓ 4D可视化函数调用成功")
    except Exception as e:
        print(f"✗ 4D可视化函数调用失败: {e}")
        return False
    
    return True

if __name__ == "__main__":
    success = test_4d_visualization()
    if success:
        print("\n🎉 4D可视化功能测试成功！")
    else:
        print("\n❌ 4D可视化功能测试失败！")
