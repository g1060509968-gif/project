"""
色散脉冲生成测试脚本 - 使用 ERK43IP_FullDispersion 求解器
包含:
1. GDD/TOD 脉冲生成测试
2. 传播仿真
3. 结果可视化 (时域/频域/相位分析)
4. [新增] 最佳脉冲压缩效果检查 (De-chirp Check)
5. [新增] 最终光谱数据导出 (CSV)
"""

import numpy as np
import matplotlib.pyplot as plt
import csv  # [新增] 用于导出数据
from scipy.fft import fft, ifft, fftfreq

# 导入必要模块
from erk43ip_method import ERK43IP_FullDispersion, SimResultAdapter
from visualization import plot_results, pulse_width, analyze_results, draw_parameter_table

# 设置中文字体支持 (防止绘图乱码)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def test_dispersed_pulse_zero_dispersion():
    """测试 generate_dispersed_pulse 及后续压缩分析"""
    print("=" * 70)
    print("色散脉冲生成与压缩测试 (GDD/TOD Analysis)")
    print("=" * 70)

    # 1. 设置仿真参数
    crystal_length = 300e-3   # 晶体长度
    time_window = 50e-12      # 50ps 时间窗口
    num_t_points = 8192       # 时间采样点数 (增加点数以提高FFT精度)
    center_wavelength = 1064e-9 # 中心波长
    beam_radius = 0.3e-3      # 光斑半径
    
    print(f"\n[1/6] 初始化 ERK43IP 求解器...")
    
    # 2. 初始化求解器
    solver = ERK43IP_FullDispersion(
        material='fused_silica',
        n2=2.7e-20,           # 非线性折射率
        beam_radius=beam_radius,
        center_wavelength=center_wavelength,
        use_raman=True,       # 开启拉曼
        use_self_steepening=True  # 开启自陡峭
    )
    
    # 3. 生成时间网格
    t = np.linspace(-time_window/2, time_window/2, num_t_points)
    
    print(f"\n[2/6] 生成初始脉冲...")
    
    # 4. 生成脉冲 
    pulse_energy = 1000e-6     # 200 µJ
    pulse_width_fwhm = 200e-15  # 200fs
    
    # 在此处设定初始色散
    input_gdd_fs2 = 160000         # 例如 0 fs^2
    input_tod_fs3 = 0          # 例如 0 fs^3
    
    try:
        A_initial = solver.generate_dispersed_pulse(
            t=t,
            pulse_energy=pulse_energy,
            pulse_fwhm=pulse_width_fwhm,
            GDD=input_gdd_fs2 * 1e-30,  
            TOD=input_tod_fs3 * 1e-45   
        )
        
        # 验证生成的脉冲
        initial_power = np.abs(A_initial)**2
        peak_power = np.max(initial_power)
        print(f"  ✅ 脉冲生成成功")
        print(f"    输入能量: {pulse_energy*1e6:.1f} µJ")
        print(f"    设定 GDD: {input_gdd_fs2} fs²")
        print(f"    生成峰值功率: {peak_power/1e6:.2f} MW")
        
    except Exception as e:
        print(f"❌ 生成失败: {e}")
        return

    # 5. 运行传播仿真
    print(f"\n[3/6] 运行传播仿真...")
    try:
        z_array, A_evolution, omega = solver.propagate(
            A_initial, t, L=crystal_length, 
            tol=1e-5, max_step=1e-2
        )
        print(f"  ✅ 传播完成: {len(z_array)} 步")
        
    except Exception as e:
        print(f"❌ 传播仿真失败: {e}")
        return

    # 准备数据适配器 (用于后续绘图和分析)
    sim_adapter = SimResultAdapter(solver, z_array, t, A_evolution)
    A_out = A_evolution[-1]

    # 6. 绘制基础结果
    print(f"\n[4/6] 绘制基础结果...")
    try:
        plot_results(sim_adapter, A_evolution, 
                    save_path="test_dispersed_pulse_basic.png")
        print("  ✅ 演化图像已保存为 test_dispersed_pulse_basic.png")
    except Exception as e:
        print(f"  ⚠️ 绘图错误: {e}")

    # =================================================================
    # 相位残差分析
    # =================================================================
    print(f"\n[5/6] 频谱相位分析...")
    res = solver.analyze_spectral_phase(A_out, t, fit_order=2, plot_threshold=0.01)
    
    if res:
        wl = res['wavelength_nm']
        spec = res['spectral_intensity']
        resid = res['residual_phase']
        mask = res['mask']
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        color1 = 'tab:blue'
        ax1.plot(wl, spec, color=color1, label='Normalized Spectrum')
        ax1.set_ylabel('Normalized Intensity', color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_ylim([0, 1.05])
        
        valid_wl = wl[mask]
        if len(valid_wl) > 0:
            span = (valid_wl.max() - valid_wl.min()) * 1.5
            center = (valid_wl.max() + valid_wl.min()) / 2
            ax1.set_xlim([center - span/2, center + span/2])
        
        ax2 = ax1.twinx()
        color2 = 'tab:red'
        ax2.plot(wl[mask], resid[mask], color=color2, linewidth=2, label='Phase Residual')
        ax2.set_ylabel('Phase Residual (rad)', color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)
        
        plt.title('Spectral Phase Analysis')
        plt.savefig("dispersed_pulse_phase_analysis.png", dpi=150, bbox_inches='tight')
        plt.show()
        print("  ✅ 相位分析图已保存")

    # =================================================================
    # [新增] 脉冲压缩潜力检查 (De-chirp Check)
    # =================================================================
    print(f"\n[6/6] 脉冲压缩能力验证 (De-chirp Check)")
    print(f"{'='*50}")
    
    # 扫描 GDD 寻找最佳压缩脉宽
    best_fwhm = 999.0
    best_gdd = 0.0
    
    # 设定扫描范围 (fs^2)
    # 这里设定为 ±100000 fs^2，步长密集一点
    # 注意：通常 SPM 会引入正啁啾，所以需要负 GDD 来补偿
    scan_range = np.linspace(-900000, 200000, 2000) * 1e-30 
    
    # 预计算频域场 (使用 numpy fft 配合 solver 参数)
    # 注意：solver.propagate 返回的 omega 是 angular frequency
    # 但为了方便，我们重新基于 t 计算 frequency axis
    omega_axis = 2 * np.pi * np.fft.fftfreq(len(t), t[1] - t[0])
    A_out_freq = np.fft.fft(A_out)
    
    for gdd_test in scan_range:
        # 应用二阶相位补偿: exp(i * 0.5 * GDD * w^2)
        phase_comp = np.exp(1j * 0.5 * gdd_test * omega_axis**2)
        A_comp_freq = A_out_freq * phase_comp
        A_comp_time = np.fft.ifft(A_comp_freq)
        
        # 测量脉宽
        try:
            # 使用 list 包装以适配 pulse_width 函数
            wid = pulse_width([A_comp_time], t*1e12)[0] 
            # pulse_width 返回的是 ps，如果没找到宽度可能返回 nan
            if not np.isnan(wid) and wid < best_fwhm:
                best_fwhm = wid
                best_gdd = gdd_test
        except:
            continue
            
    # 输出压缩结果
    initial_fwhm_res = pulse_width([A_out], t*1e12)[0]
    
    print(f"  当前输出脉宽: {initial_fwhm_res:.3f} ps")
    print(f"  最佳压缩脉宽: {best_fwhm:.3f} ps")
    print(f"  所需补偿 GDD: {best_gdd*1e30:.0f} fs²")
    print(f"  压缩比:       {initial_fwhm_res/best_fwhm:.2f}x")

    # =================================================================
    # [新增] 生成详细参数表 (含压缩结果)
    # =================================================================
    try:
        final_energy = np.trapz(np.abs(A_out)**2, t)
        pulse_params = {
            'energy': final_energy,
            'fwhm': pulse_width_fwhm,    # 初始物理脉宽
            'peak_power': np.max(np.abs(A_out)**2),
            'chirp': best_gdd*1e30,      # 记录为了压缩所需补偿的 GDD
            'note': f"Compressed: {best_fwhm:.3f}ps" # 额外备注
        }
        draw_parameter_table(sim_adapter, pulse_params, save_path="dispersed_pulse_parameter_report.png")
        print("  ✅ 参数报告已保存为 dispersed_pulse_parameter_report.png")
    except Exception as e:
        print(f"  ⚠️ 参数表生成失败: {e}")

    # =================================================================
    # [新增] 导出最终光谱数据到 CSV (含雅可比修正)
    # =================================================================
    output_csv_name = "output_dispersed_pulse_spectrum.csv"
    print(f"\n[正在导出光谱数据: {output_csv_name}]")
    
    try:
        # 1. 计算 FFT
        freqs = np.fft.fftfreq(len(t), t[1] - t[0])
        spectrum_complex = np.fft.fft(A_out)
        
        # 2. 移频（将零频移到中心）
        freqs_shifted = np.fft.fftshift(freqs)
        spectrum_complex_shifted = np.fft.fftshift(spectrum_complex)
        spectrum_intensity_freq = np.abs(spectrum_complex_shifted)**2
        
        # 3. 频率转波长 (Absolute Freq = f_rel + c/lambda0)
        # 避免除以零
        f_abs = freqs_shifted + (solver.c / center_wavelength)
        valid_mask = f_abs > 0  # 仅保留正频率
        
        f_abs = f_abs[valid_mask]
        spectrum_intensity_freq = spectrum_intensity_freq[valid_mask]
        
        wl_nm = (solver.c / f_abs) * 1e9
        
        # 4. [关键] 雅可比修正: I(lambda) = I(nu) * f^2
        # 这一步是为了修正“频率谱画在波长轴上”导致的峰值错位问题
        intensity_corrected = spectrum_intensity_freq * (f_abs**2)
        
        # 5. 归一化强度
        if np.max(intensity_corrected) > 0:
            intensity_norm = intensity_corrected / np.max(intensity_corrected)
        else:
            intensity_norm = intensity_corrected
            
        # 6. 排序（因为频率和波长成反比，FFT后的波长是从大到小的，需要反转以方便Origin绘图）
        sort_indices = np.argsort(wl_nm)
        wl_sorted = wl_nm[sort_indices]
        int_sorted = intensity_norm[sort_indices]
        
        # 7. 写入 CSV
        with open(output_csv_name, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # 写入表头
            writer.writerow(["Wavelength (nm)", "Normalized Intensity"])
            # 写入数据
            for w, i in zip(wl_sorted, int_sorted):
                # 限制保存范围 (例如只保存中心波长附近的有效数据，减小体积)
                # 这里设置为 800nm 到 1400nm
                if 800 <= w <= 1400:
                    writer.writerow([f"{w:.6f}", f"{i:.8e}"])
                    
        print(f"  ✅ 光谱数据已保存至 {output_csv_name}")
        
    except Exception as e:
        print(f"  ❌ 数据导出失败: {e}")

if __name__ == "__main__":
    test_dispersed_pulse_zero_dispersion()