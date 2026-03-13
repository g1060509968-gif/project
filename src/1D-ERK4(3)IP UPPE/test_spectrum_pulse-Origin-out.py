"""
从 CSV 读取光谱并进行脉冲传播测试
修改内容：
1. 实现33次循环的变光斑传播（光斑大小周期性变化）
2. 引入实验测定的传输损耗（总效率 53.9%）
3. [新增] 输出最终光谱数据到 CSV 文件 (含雅可比修正)
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import os
import time

# 导入 RK4IP 模块
from erk43ip_method import ERK43IP_FullDispersion
from visualization import plot_results, pulse_width

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_spectrum_csv(filename):
    """
    [Origin 参数重构版]
    基于 Origin 双高斯拟合结果直接生成数学光谱。
    彻底消除噪声，完美保留肩峰细节。
    """
    import numpy as np
    
    print(f"正在根据 Origin 拟合参数生成完美光谱...")

    # 1. 生成干净的波长轴 (范围覆盖你的数据，分辨率足够高)
    # 根据你的截图，数据大概在 1063.5 到 1066 之间
    wl_arr = np.linspace(1063.5, 1066.0, 8192) 
    
    # 2. 定义高斯函数 (Standard Gaussian)
    def gaussian(x, amp, mu, sigma):
        return amp * np.exp(- (x - mu)**2 / (2 * sigma**2))

    # ==========================================
    # 核心：填入 Origin 拟合出的参数
    # ==========================================
    
    # --- 主峰 (Peak 1) ---
    mu1    = 1064.458
    sigma1 = 0.12009 / 2  # w / 2
    amp1   = 0.947        # A / w (近似相对高度)

    # --- 肩峰 (Peak 2) ---
    mu2    = 1064.605
    sigma2 = 0.18511 / 2  # w / 2
    amp2   = 0.581        # A / w (近似相对高度)

    # 3. 叠加两个高斯峰
    intensities = gaussian(wl_arr, amp1, mu1, sigma1) + \
                  gaussian(wl_arr, amp2, mu2, sigma2)

    # 4. 归一化 (最高点设为 1.0)
    intensities = intensities / np.max(intensities)
    
    # 5. 再次清理极小的底噪 (可选，保证数学上的绝对零底)
    intensities[intensities < 1e-4] = 0

    print(f"  ✅ 光谱生成完成 (基于 Origin R^2=0.998 拟合)")
    print(f"     主峰位置: {mu1} nm")
    print(f"     肩峰位置: {mu2} nm")
    
    return wl_arr, intensities

def test_spectrum_pulse():
    print("=" * 70)
    print("光谱重建脉冲传播测试 (变光斑多通模式 + 损耗模拟)")
    print("=" * 70)

    # ==========================================
    # 1. 基础仿真参数
    # ==========================================
    segment_length = 12e-3   # 每一小段介质厚度 12mm
    time_window = 120e-12    # 120ps (根据需要可调整为 200ps 防止溢出)
    num_t_points = 8192      # 采样点数
    center_wavelength = 1064e-9
    
    # 设定目标脉冲参数
    target_energy = 200e-6   # 200 uJ
    #target_gdd_fs2 = 23600000
    target_gdd_fs2 =  23600000
    # ==========================================
    # 2. 定义光斑变化序列 (单位: mm -> m)
    # ==========================================
    # 阶段1: 变大
    radii_phase1 = np.array([0.2222, 0.2427, 0.2737, 0.3120]) * 1e-3
    # 阶段2: 变小 (倒序)
    radii_phase2 = radii_phase1[::-1] 
    
    # 合并为一个完整的周期序列 (共8个值)
    radii_cycle = np.concatenate((radii_phase1, radii_phase2))
    
    num_cycles = 22      # 循环次数
    num_segments_per_cycle = len(radii_cycle)
    total_segments = num_cycles * num_segments_per_cycle
    
    # ==========================================
    # 3. 计算损耗因子 (基于论文数据)
    # ==========================================
    # 论文指出 MPC 总功率传输效率为 53.9% (0.539) 
    # 我们将这个总损耗均匀分配到每一次穿过介质的过程中
    # 功率 P_out = P_in * (eff_total)^(1/N)
    # 振幅 A_out = A_in * sqrt(power_factor)
    
    total_efficiency_power = 0.539
    attenuation_per_segment_amp = np.power(total_efficiency_power, 1.0 / (2.0 * total_segments))
    
    print(f"损耗参数计算:")
    print(f"  总传输效率 (功率): {total_efficiency_power*100:.1f}%")
    print(f"  总段数: {total_segments}")
    print(f"  单段振幅保持率: {attenuation_per_segment_amp:.6f}")
    
    # ==========================================
    # 4. 初始化求解器
    # ==========================================
    initial_radius = radii_cycle[0]
    solver = ERK43IP_FullDispersion(
        material='fused_silica',
        n2=2.7e-20,
        beam_radius=initial_radius,
        center_wavelength=center_wavelength,
        use_raman=True,
        use_self_steepening=True
    )

    # 5. 生成时间网格
    t = np.linspace(-time_window/2, time_window/2, num_t_points)
    
    # 6. 读取 CSV 并生成初始脉冲
    csv_filename = "ClipboardImage1.csv"
    #csv_filename = "input_spectrum.csv"
    
    # 注意：这里的 load_spectrum_csv 实际上是忽略文件名直接生成数据的
    # 但为了保持代码结构兼容，我们保留调用形式
    wl_data, int_data = load_spectrum_csv(csv_filename)

    print(f"读取成功: {len(wl_data)} 个数据点")

    # 重建脉冲
    try:
        A_initial = solver.generate_pulse_from_spectrum(
            t=t,
            wavelengths_nm=wl_data,
            intensities=int_data,
            pulse_energy=target_energy,
            GDD=target_gdd_fs2 * 1e-30, 
            TOD=0,
            use_jacobian=True,
            interp_kind='cubic',
            exact_fourier=False
        )
        initial_fwhm = pulse_width([A_initial], t*1e12)[0]
        print(f"  ✅ 脉冲重建成功, 宽度: {initial_fwhm:.3f} ps")
    except Exception as e:
        print(f"  ❌ 脉冲重建失败: {e}")
        return

    # ==========================================
    # 7. 核心循环：多段变光斑传播
    # ==========================================
    print(f"\n[开始多段变光斑传播仿真]")
    print(f"  单段长度: {segment_length*1e3:.1f} mm")
    
    # 用于存储完整的演化历史
    z_total_history = [0.0]
    A_total_history = [A_initial.copy()]
    
    current_A = A_initial.copy()
    total_distance_accumulated = 0.0
    
    start_time = time.time()
    
    # 外层循环：33次
    for cycle_idx in range(num_cycles):
        # 内层循环：遍历光斑序列 (8个值)
        for step_idx, r_current in enumerate(radii_cycle):
            
            # --- A. 动态更新求解器参数 ---
            solver.beam_radius = r_current
            solver.beam_area = np.pi * r_current**2
            # 重新计算非线性系数 gamma = (2*pi/lambda) * (n2/A_eff)
            solver.gamma = (2 * np.pi / solver.lambda0) * (solver.n2 / solver.beam_area)
            
            # --- B. 运行单小段传播 ---
            # max_step 限制为 segment_length 确保不会跳过物理过程
            z_seg, A_seg, _ = solver.propagate(
                current_A, t, L=segment_length, 
                tol=1e-5, max_step=segment_length 
            )
            
            # --- C. 提取该段终点 ---
            segment_out_A = A_seg[-1]
            
            # --- D. 拼接数据 (用于绘图) ---
            # 注意：z_seg 是相对距离，需要加上之前的累积距离
            z_absolute = z_seg[1:] + total_distance_accumulated
            
            z_total_history.extend(z_absolute)
            A_total_history.extend(A_seg[1:]) # 添加这一段的演化过程
            
            # --- E. 应用损耗 (模拟镜片/表面损耗) ---
            # 这是下一次循环的起点
            current_A = segment_out_A * attenuation_per_segment_amp
            
            # 累加总距离
            total_distance_accumulated += segment_length
            
        # 打印进度
        if (cycle_idx + 1) % 5 == 0 or (cycle_idx + 1) == num_cycles:
             # 计算当前能量
             curr_energy = np.trapz(np.abs(current_A)**2, t) * 1e6
             print(f"  >>> 完成循环 {cycle_idx + 1}/{num_cycles}, 总厚度: {total_distance_accumulated*1000:.1f} mm, 当前能量: {curr_energy:.2f} uJ")

    end_time = time.time()
    print(f"\n✅ 仿真完成!")
    print(f"  耗时: {end_time - start_time:.2f} 秒")
    print(f"  最后经过的介质总厚度: {total_distance_accumulated:.4f} m")

    # 转换历史数据为 numpy 数组
    z_final_array = np.array(z_total_history)
    A_final_evolution = np.array(A_total_history)

    # ==========================================
    # 8. 结果处理与绘图
    # ==========================================
    
    # 构造适配器
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
            # 注意：这里的 beam_radius 只是为了兼容接口，实际上光斑是变化的
            self.beam_radius = getattr(solver, 'beam_radius', 0.0)
    
    sim_adapter = SimAdapter(z_final_array, t, A_final_evolution, solver)

    # 绘图 1: 演化色图
    try:
        plot_results(sim_adapter, A_final_evolution, save_path="spectrum_pulse_results_multipass.png")
        print("绘图完成，已保存为 spectrum_pulse_results_multipass.png")
    except Exception as e:
        print(f"绘图模块报错: {e}")
        
    # 分析: 最终输出脉冲
    A_out = A_final_evolution[-1]
    
    # 绘图 2: 相位分析
    res = solver.analyze_spectral_phase(A_out, t, fit_order=3, plot_threshold=0.01)

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
        
        plt.title(f'Spectral Phase Analysis (After {total_distance_accumulated*1000:.1f}mm)')
        plt.savefig("spectrum_pulse_phase_analysis_multipass.png", dpi=150, bbox_inches='tight')
        plt.show()
        print("  ✅ 相位残差图像已保存")
    
    # 绘图 3: 参数表报告
    try:
        from erk43ip_method import SimResultAdapter
        from visualization import draw_parameter_table
        
        adapter = SimResultAdapter(solver, z_final_array, t, A_final_evolution)
        
        # 计算最终能量
        final_energy = np.trapz(np.abs(A_out)**2, t)
        
        pulse_params = {
            'energy': final_energy,          # 最终能量
            'fwhm': initial_fwhm,            # 初始脉宽(参考)
            'peak_power': np.max(np.abs(A_out)**2),
            'chirp': 0,
            'total_length': total_distance_accumulated
        }
        
        draw_parameter_table(adapter, pulse_params, save_path="spectrum_pulse_parameter_report_multipass.png")
        print("  ✅ 参数表已保存")
        
    except Exception as e:
        print(f"  ⚠️ 参数表生成失败: {e}")

    # ==========================================
    # 9. [新增] 导出最终光谱数据到 CSV (Origin 专用)
    # ==========================================
    output_csv_name = "output_final_spectrum_origin.csv"
    print(f"\n[正在导出数据: {output_csv_name}]")
    
    try:
        # 1. 计算 FFT
        # t 已经在作用域中定义
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
        
        # lambda = c / f
        wl_nm = (solver.c / f_abs) * 1e9
        
        # 4. [关键] 雅可比修正: I(lambda) = I(nu) * f^2
        # 这一步是为了修正“频率谱画在波长轴上”导致的峰值错位问题
        intensity_corrected = spectrum_intensity_freq * (f_abs**2)
        
        # 5. 归一化强度
        if np.max(intensity_corrected) > 0:
            intensity_norm = intensity_corrected / np.max(intensity_corrected)
        else:
            intensity_norm = intensity_corrected
            
        # 6. 排序（因为频率和波长成反比，FFT后的波长是从大到小的，需要反转以方便Origin）
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
                # 限制保存范围（例如 600nm - 2000nm）以减小文件体积，或者保存全部
                if 1060 <= w <= 1070:
                    writer.writerow([f"{w:.6f}", f"{i:.8e}"])
                    
        print(f"  ✅ 光谱数据已保存至 {output_csv_name}")
        
    except Exception as e:
        print(f"  ❌ 数据导出失败: {e}")

if __name__ == "__main__":
    test_spectrum_pulse()