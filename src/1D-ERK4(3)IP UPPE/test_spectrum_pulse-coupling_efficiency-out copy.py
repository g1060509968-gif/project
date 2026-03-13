"""
test_spectrum_pulse.py
----------------------
从 CSV (或数学模型) 读取光谱并进行 MPC 脉冲传播测试。
修正内容：
1. 引入 [耦合效率] (Coupling Efficiency) 降低入腔有效能量，修正过强的非线性效应。
2. 动态计算腔内损耗，保证总传输效率符合论文 (53.9%)。
3. 使用 Origin 拟合参数直接生成无噪输入光谱。
4. 包含最终脉冲压缩检查 (De-chirp check)。
5. [新增] 输出最终光谱数据到 CSV 文件 (含雅可比修正)。
6. [新增] 输出初始与最终时域波形对比图 (已修复视窗裁剪问题)。
"""

import numpy as np
import matplotlib.pyplot as plt
import csv  # 新增 csv 模块
import os
import time

# 导入核心求解器和可视化工具
# 请确保 erk43ip_method.py 和 visualization.py 在同一目录下
from erk43ip_method import ERK43IP_FullDispersion, SimResultAdapter
from visualization import plot_results, pulse_width, draw_parameter_table

# 设置绘图风格
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_spectrum_mathematical():
    """
    [Origin 参数重构版]
    不依赖外部 CSV，直接基于 Origin 双高斯拟合结果生成数学光谱。
    彻底消除噪声，完美保留肩峰细节。
    """
    print(f"正在根据 Origin 拟合参数生成完美光谱...")

    # 1. 生成干净的波长轴 (覆盖 1063.5 - 1066 nm)
    wl_arr = np.linspace(1063.5, 1066.0, 8192) 
    
    # 2. 定义高斯函数
    def gaussian(x, amp, mu, sigma):
        return amp * np.exp(- (x - mu)**2 / (2 * sigma**2))

    # ==========================================
    # 核心：填入 Origin 拟合出的参数 (R^2=0.998)
    # ==========================================
    
    # --- 主峰 (Peak 1) ---
    mu1    = 1064.458
    sigma1 = 0.12009 / 2  
    amp1   = 0.947        

    # --- 肩峰 (Peak 2) ---
    mu2    = 1064.605
    sigma2 = 0.18511 / 2  
    amp2   = 0.581        

    # 3. 叠加生成
    intensities = gaussian(wl_arr, amp1, mu1, sigma1) + \
                  gaussian(wl_arr, amp2, mu2, sigma2)

    # 4. 归一化与去底噪
    intensities = intensities / np.max(intensities)
    intensities[intensities < 1e-4] = 0

    print(f"  ✅ 光谱生成完成: 主峰 {mu1}nm, 肩峰 {mu2}nm")
    
    return wl_arr, intensities

def test_spectrum_pulse():
    print("=" * 70)
    print("光谱重建脉冲传播测试 (最终物理修正版 + CSV导出 + 时域对比修复)")
    print("=" * 70)

    # ==========================================
    # 1. 基础仿真参数
    # ==========================================
    segment_length = 12e-3   # 12mm
    time_window = 120e-12    # 120ps
    num_t_points = 8192      # 采样点数
    center_wavelength = 1064e-9
    
    # ==========================================
    # 2. 能量与耦合设置 (关键修正)
    # ==========================================
    laser_output_energy = 200e-6   # 激光器输出 200 uJ
    
    # [耦合效率]: 模拟入腔前的透镜组(L1-L3)损耗。
    # 建议值: 0.85 (170uJ) - 0.90 (180uJ)。
    # 降低此值可减弱非线性，帮助恢复光谱双峰结构。
    coupling_efficiency = 1   
    
    effective_energy = laser_output_energy * coupling_efficiency
    
    print(f"能量参数设置:")
    print(f"  激光器输出能量: {laser_output_energy*1e6:.1f} uJ")
    print(f"  入腔耦合效率:   {coupling_efficiency*100:.1f}%")
    print(f"  有效入腔能量:   {effective_energy*1e6:.1f} uJ (用于非线性计算)")

    # ==========================================
    # 3. 损耗与效率计算
    # ==========================================
    # 论文总效率 53.9% (Total_Out / Laser_Out)
    target_total_efficiency = 0.539
    
    # 腔内传输效率 = 总效率 / 耦合效率
    # (排除掉入腔损耗后，光在多通腔内部传输的纯效率)
    cavity_efficiency = target_total_efficiency / coupling_efficiency
    
    # 循环设定
    num_cycles = 33
    # 光斑序列 (用户指定)
    radii_phase1 = np.array([0.2222, 0.2427, 0.2737, 0.3120]) * 1e-3
    radii_phase2 = radii_phase1[::-1]
    radii_cycle = np.concatenate((radii_phase1, radii_phase2))
    
    total_segments = num_cycles * len(radii_cycle)
    
    # 计算单段衰减因子 (振幅)
    # P_out = P_in * (eff_cavity)
    # A_factor = sqrt(eff_cavity^(1/N))
    attenuation_per_segment_amp = np.power(cavity_efficiency, 1.0 / (2.0 * total_segments))
    
    print(f"损耗计算:")
    print(f"  目标总效率:     {target_total_efficiency*100:.1f}%")
    print(f"  腔内纯传输效率: {cavity_efficiency*100:.1f}%")
    print(f"  单段振幅保持率: {attenuation_per_segment_amp:.6f}")

    # ==========================================
    # 4. GDD 与 初始化
    # ==========================================
    target_gdd_fs2 = 23600000  # 用户确认的 GDD 值
    
    initial_radius = radii_cycle[0]
    solver = ERK43IP_FullDispersion(
        material='fused_silica',
        n2=2.7e-20,
        beam_radius=initial_radius,
        center_wavelength=center_wavelength,
        use_raman=True,
        use_self_steepening=True
    )

    t = np.linspace(-time_window/2, time_window/2, num_t_points)
    
    # ==========================================
    # 5. 生成初始脉冲
    # ==========================================
    # 使用数学重构的光谱
    wl_data, int_data = load_spectrum_mathematical()

    A_initial = solver.generate_pulse_from_spectrum(
        t=t,
        wavelengths_nm=wl_data,
        intensities=int_data,
        pulse_energy=effective_energy,  # 使用有效能量
        GDD=target_gdd_fs2 * 1e-30, 
        TOD=0,
        use_jacobian=True,
        interp_kind='cubic',
        exact_fourier=False
    )
    initial_fwhm = pulse_width([A_initial], t*1e12)[0]
    print(f"  ✅ 初始脉冲生成成功, FWHM: {initial_fwhm:.3f} ps")

    # ==========================================
    # 6. 传播主循环
    # ==========================================
    print(f"\n[开始传播: {num_cycles}次循环, 总长 {num_cycles * len(radii_cycle) * segment_length:.3f}m]")
    
    z_total_history = [0.0]
    A_total_history = [A_initial.copy()]
    
    current_A = A_initial.copy()
    total_distance_accumulated = 0.0
    
    start_time = time.time()
    
    for cycle_idx in range(num_cycles):
        for step_idx, r_current in enumerate(radii_cycle):
            
            # 1. 更新求解器参数 (变光斑)
            solver.beam_radius = r_current
            solver.beam_area = np.pi * r_current**2
            solver.gamma = (2 * np.pi / solver.lambda0) * (solver.n2 / solver.beam_area)
            
            # 2. 传播单段
            z_seg, A_seg, _ = solver.propagate(
                current_A, t, L=segment_length, 
                tol=1e-5, max_step=segment_length 
            )
            
            # 3. 记录数据
            segment_out_A = A_seg[-1]
            z_absolute = z_seg[1:] + total_distance_accumulated
            z_total_history.extend(z_absolute)
            A_total_history.extend(A_seg[1:])
            
            # 4. 应用损耗 (衰减)
            current_A = segment_out_A * attenuation_per_segment_amp
            
            # 5. 累加距离
            total_distance_accumulated += segment_length
            
        # 进度打印
        if (cycle_idx + 1) % 5 == 0 or (cycle_idx + 1) == num_cycles:
             curr_energy = np.trapz(np.abs(current_A)**2, t) * 1e6
             print(f"  >>> 循环 {cycle_idx + 1}/{num_cycles} 完成, 当前能量: {curr_energy:.2f} uJ")

    end_time = time.time()
    print(f"\n✅ 仿真完成! 耗时: {end_time - start_time:.2f} 秒")

    # 转为数组
    z_final_array = np.array(z_total_history)
    A_final_evolution = np.array(A_total_history)
    A_out = A_final_evolution[-1]

    # ==========================================
    # 7. 结果绘图与分析
    # ==========================================
    
    # --- 绘图 1: 常规演化图 ---
    sim_adapter = SimResultAdapter(solver, z_final_array, t, A_final_evolution)
    try:
        plot_results(sim_adapter, A_final_evolution, save_path="spectrum_pulse_results_multipass.png")
        print("  ✅ 演化图已保存")
    except Exception as e:
        print(f"  ⚠️ 绘图失败: {e}")

    # --- 绘图 2: 光谱相位分析 (重点检查双峰) ---
    res = solver.analyze_spectral_phase(A_out, t, fit_order=3, plot_threshold=0.01)
    if res:
        wl = res['wavelength_nm']
        spec = res['spectral_intensity']
        resid = res['residual_phase']
        mask = res['mask']
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # 绘制输出光谱
        color1 = 'tab:red'
        ax1.plot(wl, spec, color=color1, linewidth=2, label='Output Spectrum')
        ax1.set_ylabel('Normalized Intensity', color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)
        ax1.set_ylim([0, 1.05])
        
        # 绘制输入光谱轮廓 (虚线) 用于对比
        # 简单计算输入光谱
        spec_in = np.abs(np.fft.fftshift(np.fft.fft(A_initial)))**2
        spec_in /= np.max(spec_in)
        # 频率轴映射 (近似)
        N = len(t); dt = t[1]-t[0]
        freq = np.fft.fftshift(np.fft.fftfreq(N, dt))
        wl_axis = 3e8 / (solver.c/solver.lambda0 + freq) * 1e9
        # 截取范围绘制
        valid_idx = (wl_axis > wl.min()) & (wl_axis < wl.max())
        ax1.plot(wl_axis[valid_idx], spec_in[valid_idx], 'b--', alpha=0.6, label='Input Spectrum')

        # 绘制相位
        ax2 = ax1.twinx()
        color2 = 'tab:blue'
        ax2.plot(wl[mask], resid[mask], color=color2, linewidth=1.5, label='Residual Phase')
        ax2.set_ylabel('Phase Residual (rad)', color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)
        
        # 设置范围
        center_wl = 1064.5
        ax1.set_xlim([center_wl - 3, center_wl + 3])
        
        plt.title(f'Spectral Broadening Check (Eff Energy={effective_energy*1e6:.1f}uJ)')
        lines, labels = ax1.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines + lines2, labels + labels2, loc='upper left')
        
        plt.savefig("spectrum_pulse_phase_analysis_final.png", dpi=150, bbox_inches='tight')
        print("  ✅ 光谱分析图已保存: spectrum_pulse_phase_analysis_final.png")
    
    # ==========================================
    # [新增] 7.5 初始与最终时域对比图 (修正版)
    # ==========================================
    print(f"\n{'='*30}")
    print("生成时域对比图")
    print(f"{'='*30}")
    
    try:
        # 新建一个图形
        plt.figure(figsize=(10, 6))
        
        # 转换时间单位为 ps
        t_ps = t * 1e12
        
        # 计算功率 (MW)
        P_in_MW = np.abs(A_initial)**2 * 1e-6
        P_out_MW = np.abs(A_out)**2 * 1e-6
        
        # 绘制
        plt.plot(t_ps, P_in_MW, 'b--', label='Initial Pulse', linewidth=1.5, alpha=0.7)
        plt.plot(t_ps, P_out_MW, 'r-', label='Final Output', linewidth=2.0, alpha=0.9)
        
        # 标签和标题
        plt.xlabel('Time (ps)', fontsize=12)
        plt.ylabel('Power (MW)', fontsize=12)
        plt.title(f'Time Domain Comparison\n(Input: {effective_energy*1e6:.1f}uJ vs Output: {np.trapz(np.abs(A_out)**2, t)*1e6:.1f}uJ)', fontsize=14)
        plt.legend(fontsize=12)
        plt.grid(True, linestyle='--', alpha=0.4)
        
        # [修正逻辑] 自动寻找峰值位置进行对齐，防止切光
        peak_idx = np.argmax(P_out_MW)
        t_peak = t_ps[peak_idx]
        
        # 设置显示范围：以峰值为中心，左右各扩展 30 ps
        # 如果脉冲漂移到了 20ps，显示范围将自动调整为 [-10, 50]
        plt.xlim([t_peak - 30, t_peak + 30])
        
        save_name_td = "time_domain_comparison.png"
        plt.savefig(save_name_td, dpi=300, bbox_inches='tight')
        plt.close() # 释放图形内存
        print(f"  ✅ 时域对比图已保存: {save_name_td}")
        
    except Exception as e:
        print(f"  ⚠️ 时域绘图失败: {e}")

    # ==========================================
    # 8. 压缩潜力检查 (De-chirp Check)
    # ==========================================
    print(f"\n{'='*30}")
    print("脉冲压缩能力验证")
    print(f"{'='*30}")
    
    # 扫描 GDD 寻找最佳压缩脉宽
    best_fwhm = 999.0
    best_gdd = 0.0
    
    # 粗略扫描范围: -2.0e6 到 -0.5e6 fs2 (通常SPM产生正啁啾，需负色散补偿)
    scan_range = np.linspace(-2.5e6, 2.5e6, 2000) * 1e-30 
    
    # 预计算频域场
    omega = 2 * np.pi * np.fft.fftfreq(len(t), t[1] - t[0])
    A_out_freq = np.fft.fft(A_out)
    
    for gdd_test in scan_range:
        # 应用相位补偿
        phase_comp = np.exp(0.5j * gdd_test * omega**2)
        A_comp_freq = A_out_freq * phase_comp
        A_comp_time = np.fft.ifft(A_comp_freq)
        
        # 测量脉宽
        try:
            wid = pulse_width([A_comp_time], t*1e12)[0]
            if wid < best_fwhm:
                best_fwhm = wid
                best_gdd = gdd_test
        except:
            continue
            
    print(f"  最佳压缩脉宽: {best_fwhm:.3f} ps (论文目标: 0.483 ps)")
    print(f"  所需补偿 GDD: {best_gdd*1e30:.0f} fs²")
    
    if abs(best_fwhm - 0.483) < 0.1:
        print("  🌟 结果判定: 完美吻合! 物理过程已复现。")
    else:
        print("  ℹ️ 结果判定: 接近目标。如需更精准，可微调 coupling_efficiency。")
    
    # 生成参数表
    try:
        final_energy = np.trapz(np.abs(A_out)**2, t)
        pulse_params = {
            'energy': final_energy,
            'fwhm': initial_fwhm,
            'peak_power': np.max(np.abs(A_out)**2),
            'chirp': best_gdd*1e30, # 记录所需的补偿量
            'total_length': total_distance_accumulated
        }
        draw_parameter_table(sim_adapter, pulse_params, save_path="spectrum_pulse_parameter_report_final.png")
        print("  ✅ 参数报告已保存")
    except:
        pass

    # ==========================================
    # 9. [新增] 导出最终光谱数据到 CSV
    # ==========================================
    output_csv_name = "output_final_spectrum_coupling.csv"
    print(f"\n[正在导出数据: {output_csv_name}]")
    
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