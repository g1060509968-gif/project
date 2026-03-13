"""
高斯脉冲测试脚本 - 强制关闭色散以复现纯 SPM M型光谱 (修复版)
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
    print("=" * 70)
    print("高斯脉冲传播测试 - 纯 SPM (色散关闭)")
    print("=" * 70)

    # 1. 参数设置 (保持不变)
    center_wavelength = 800e-9
    crystal_length = 1.89        # 1.05 m
    beam_radius = 200e-6         
    
    pulse_width_fwhm = 83e-15    # 83 fs
    pulse_energy = 177e-9        # 177 nJ
    
    time_window = 4e-12          
    num_t_points = 8192          

    print(f"\n[1/6] 初始化 ERK43IP 求解器...")
    
    # 2. 初始化
    solver = ERK43IP_FullDispersion(
        material='fused_silica',
        n2=6.0e-20,
        beam_radius=beam_radius,
        center_wavelength=center_wavelength,
        use_raman=False,      
        use_self_steepening=False  
    )
    
    # =========================================================
    # 【关键修改】强制关闭色散 (Dispersion OFF) - 修复版
    # =========================================================
    print(f"  ⚡ HACK: 正在强制将色散算子置零...")

    # 定义一个 Hack 函数：不管输入什么频率，永远返回全 0 数组
    def zero_dispersion_hack(omega_grid):
        return np.zeros_like(omega_grid, dtype=complex)

    # Monkey Patch: 动态替换实例的方法
    # 这样当求解器后续在内部调用 _compute_dispersion_operator 时，
    # 实际上执行的是我们的 zero_dispersion_hack
    solver._compute_dispersion_operator = zero_dispersion_hack
    
    # 确保缓存为空，强制求解器在第一步时调用我们的 Hack 函数
    solver._cached_D = None
    
    print(f"    ✅ 已通过方法替换 (Monkey Patch) 彻底禁用色散。")
    # =========================================================

    # 3. 生成网格
    t = np.linspace(-time_window/2, time_window/2, num_t_points)
    
    # 4. 生成脉冲
    print(f"\n[2/6] 生成高斯脉冲...")
    A_initial = solver.generate_gaussian_pulse(
        t=t,
        pulse_energy=pulse_energy,
        pulse_fwhm=pulse_width_fwhm,
        chirp=0
    )
    
    # 5. 运行传播
    print(f"\n[3/6] 运行传播仿真 (纯 SPM)...")
    z_array, A_evolution, omega = solver.propagate(
        A_initial, t, L=crystal_length, 
        tol=1e-5, max_step=1e-2 
    )

    # 6. 适配器
    class SimAdapter:
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

    # 7. 绘图
    print(f"\n[5/6] 绘制结果...")
    plot_simple_results(sim_adapter, A_evolution)

def plot_simple_results(sim, A_evolution):
    """绘图函数"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # 时域 (检查脉冲是否保持形状)
    Power = np.abs(A_evolution)**2
    ax = axes[0, 0]
    ax.plot(sim.t*1e15, Power[0, :], 'b--', label='Input')
    ax.plot(sim.t*1e15, Power[-1, :], 'r-', label='Output')
    ax.set_title('Pulse Shape (Should barely change in time)')
    ax.set_xlabel('Time (fs)')
    ax.set_xlim([-200, 200])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 频谱 (检查 M 型)
    ax = axes[0, 1]
    from scipy.fft import fft, fftfreq
    freq = fftfreq(sim.Nt, sim.dt)
    omega_abs = sim.omega0 + 2*np.pi*freq
    valid = omega_abs > 1e12
    wl = np.zeros_like(omega_abs)
    wl[valid] = 2 * np.pi * sim.c / omega_abs[valid] * 1e9
    
    spec_in = np.abs(fft(A_evolution[0, :]))**2
    spec_out = np.abs(fft(A_evolution[-1, :]))**2
    
    sort = np.argsort(wl)
    ax.plot(wl[sort], spec_in[sort]/spec_in.max(), 'b--', label='Input')
    ax.plot(wl[sort], spec_out[sort]/spec_out.max(), 'r-', label='Output', linewidth=2)
    ax.set_title('Spectrum (Expect M-Shape)')
    ax.set_xlabel('Wavelength (nm)')
    ax.set_xlim([650, 1000])
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 峰值功率 (应该保持稳定)
    ax = axes[1, 1]
    peak = np.max(Power, axis=1)
    ax.plot(sim.z, peak/1e6, 'k-')
    ax.set_title('Peak Power (Should be flat)')
    ax.set_ylabel('MW')
    ax.set_ylim([0, 2.5])
    ax.grid(True)
    
    plt.tight_layout()
    plt.savefig("erk43ip_pure_spm.png", dpi=150)
    plt.show()

if __name__ == "__main__":
    test_gaussian_pulse()