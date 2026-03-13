"""
高斯脉冲测试脚本 - 使用理想高斯光束作为初始脉冲
"""
import numpy as np
import matplotlib
# Use non-interactive backend to avoid blocking
#matplotlib.use('Agg')
import matplotlib.pyplot as plt
import sys
import os

sys.path.append(os.path.dirname(__file__))

from simulator_core import NonlinearCrystalSimulator
from visualization import plot_results,  plot_compressibility, draw_parameter_table

def test_gaussian_pulse():
    """测试高斯脉冲传播"""
    print("=" * 70)
    print("高斯脉冲传播测试")
    print("=" * 70)

    # 1. 设置仿真参数
    crystal_length = 500e-3  # 3.1m 晶体长度
    num_z_steps = 1000        # 传播步数
    time_window = 50e-12     # 50ps 时间窗口
    num_t_points = 4096       # 时间采样点数
    
    print(f"\n初始化仿真器...")
    sim = NonlinearCrystalSimulator(
        crystal_length=crystal_length, num_z_steps=num_z_steps,
        time_window=time_window, num_t_points=num_t_points,
        center_wavelength=1064e-9, beam_radius=0.35e-3,
        material='fused_silica', dispersion_mode='sellmeier'
    )
    
    # 2. 设置物理参数
    sim.set_parameters(
        n2=2.7e-20,           # 非线性折射率系数 (m²/W)
        alpha=0,              # 损耗系数 (m⁻¹)
        self_steepening=False, # 自陡峭效应
        raman_response=False   # 拉曼响应
    )
    
    # 3. 生成理想高斯脉冲
    print(f"\n生成理想高斯脉冲...")
    pulse_energy = 2000e-6     # 200 µJ 脉冲能量
    pulse_width_fwhm = 2070e-15 # 10ps 脉冲宽度 (FWHM)


    try:
        # 生成脉冲
        # pulse_width_fwhm = 200e-15  (决定光谱宽度)
        # target_chirped_width = 10e-12 (决定实际时域宽度)
        A_initial = sim.gaussian_pulse(
            pulse_energy=pulse_energy, 
            pulse_width_fwhm=200e-15, 
            target_chirped_width=pulse_width_fwhm
        )
        
        # 计算脉冲参数
        initial_power = np.abs(A_initial)**2
        peak_power = np.max(initial_power)
        pulse_energy_calc = sim._compute_energy(A_initial)
        pulse_width_calc = sim._compute_fwhm(initial_power, sim.t)
        
        print(f"  ✅ 高斯脉冲生成成功")
        print(f"    脉冲能量: {pulse_energy_calc*1e6:.2f} uJ (目标: {pulse_energy*1e6:.2f} uJ)")
        print(f"    峰值功率: {peak_power/1e3:.1f} kW")
        print(f"    脉冲宽度: {pulse_width_calc*1e12:.2f} ps (FWHM)")
        
    except Exception as e:
        print(f"  ❌ 高斯脉冲生成失败: {e}")
        return

    # 4. 运行传播仿真
    print(f"\n运行传播仿真...")
    try:
        A_evolution = sim.propagate(A_initial, verbose=True, track_phase_evolution=True)
        print("  ✅ 传播仿真成功")
    except Exception as e:
        print(f"  ❌ 传播仿真失败: {e}")
        return

    # 5. 绘制结果
    print(f"\n绘制仿真结果...")
    try:
        plot_results(sim, A_evolution, 
                    save_path="gaussian_pulse_simulation_results.png")
        print("  ✅ 结果绘图成功")
    except Exception as e:
        print(f"  ❌ 结果绘图失败: {e}")
        return

    # 7. 可压缩性分析
    print(f"\n进行可压缩性分析...")
    try:
        A_final = A_evolution[-1, :]
        plot_compressibility(sim, A_final, threshold_db=-20)
        print("  ✅ 可压缩性分析成功")
    except Exception as e:
        print(f"  ❌ 可压缩性分析失败: {e}")

    # 8. 详细结果分析
    print(f"\n{'='*70}")
    print("详细结果分析")
    print(f"{'='*70}")
    
    # 计算输入输出参数
    Power = np.abs(A_evolution)**2
    Spectrum = np.abs(np.fft.fft(A_evolution, axis=1))**2
    
    # 输入参数
    power_in = Power[0, :]
    peak_power_in = np.max(power_in)
    energy_in = sim._compute_energy(A_evolution[0, :])
    fwhm_in = sim._compute_fwhm(power_in, sim.t)
    
    # 输出参数
    power_out = Power[-1, :]
    peak_power_out = np.max(power_out)
    energy_out = sim._compute_energy(A_evolution[-1, :])
    fwhm_out = sim._compute_fwhm(power_out, sim.t)
    
    # 频谱展宽分析
    spectrum_in = Spectrum[0, :]
    spectrum_out = Spectrum[-1, :]
    
    # 计算频谱宽度 (FWHM)
    freq = np.fft.fftfreq(sim.Nt, sim.dt)
    wavelength = 2 * np.pi * sim.c / (sim.omega0 + 2*np.pi*freq) * 1e9
    
    # 找到频谱峰值位置
    peak_idx_in = np.argmax(spectrum_in)
    peak_idx_out = np.argmax(spectrum_out)
    
    # 计算频谱FWHM
    def spectral_fwhm(wavelength, spectrum):
        max_val = np.max(spectrum)
        indices = np.where(spectrum >= max_val/2)[0]
        if len(indices) > 1:
            return wavelength[indices[-1]] - wavelength[indices[0]]
        return 0
    
    spectral_width_in = spectral_fwhm(wavelength, spectrum_in)
    spectral_width_out = spectral_fwhm(wavelength, spectrum_out)
    
    print(f"输入脉冲:")
    print(f"  能量: {energy_in*1e6:.2f} uJ")
    print(f"  峰值功率: {peak_power_in/1e3:.1f} kW")
    print(f"  脉冲宽度: {fwhm_in*1e12:.2f} ps (FWHM)")
    print(f"  频谱宽度: {spectral_width_in:.2f} nm (FWHM)")
    
    print(f"\n输出脉冲:")
    print(f"  能量: {energy_out*1e6:.2f} uJ (守恒率: {energy_out/energy_in*100:.1f}%)")
    print(f"  峰值功率: {peak_power_out/1e3:.1f} kW")
    print(f"  脉冲宽度: {fwhm_out*1e12:.2f} ps (FWHM)")
    print(f"  频谱宽度: {spectral_width_out:.2f} nm (FWHM)")
    
    print(f"\n非线性效应:")
    print(f"  频谱展宽因子: {spectral_width_out/spectral_width_in:.2f}")
    print(f"  脉冲展宽因子: {fwhm_out/fwhm_in:.2f}")
    print(f"  峰值功率变化: {peak_power_out/peak_power_in:.2f}")
    
    # 非线性长度估算
    if sim.gamma > 0:
        L_NL = 1 / (sim.gamma * peak_power_in)
        print(f"  非线性长度 L_NL: {L_NL*1e3:.1f} mm")
        print(f"  传播距离/L_NL: {sim.L/L_NL:.1f}")
    
    print(f"{'='*70}\n")

    pulse_params = {
        'energy': pulse_energy_calc,      # 使用计算出的实际能量
        'fwhm': pulse_width_calc,         # 使用计算出的实际宽度
        'peak_power': peak_power,
        'chirp': 0  # 啁啾参数不再使用，设为0
    }
    
    # === 新增：绘制参数表 ===
    print(f"\n生成参数报告...")
    draw_parameter_table(sim, pulse_params, save_path="simulation_parameters.png")

if __name__ == "__main__":
    # 运行主测试
    test_gaussian_pulse()
