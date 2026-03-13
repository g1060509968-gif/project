"""
test_gaussian_pulse_cascaded.py
-------------------------------
级联高斯脉冲测试脚本
模拟分三段传播：2mm -> 2mm -> 2mm
光斑半径逐级减小 (400->300->200um)，实现非线性渐进增强。
包含最终脉冲压缩能力验证 (De-chirp Check)。
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import time

# 导入 RK4IP 模块
from erk43ip_method import ERK43IP_FullDispersion, SimResultAdapter
from visualization import plot_results, pulse_width, draw_parameter_table

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def test_cascaded_propagation():
    """测试级联变光斑传播"""
    print("=" * 70)
    print("级联脉冲传播测试 (3段式变光斑 + 最终压缩验证)")
    print("=" * 70)

    # =================================================================
    # 1. 级联配置 (用户在此处设置)
    # =================================================================
    # 三段晶体的长度 (每段 2mm)
    stage_lengths = [2e-3, 2e-3, 2e-3] 
    
    # 三段的光斑半径 (用户设定: 逐渐聚焦 400um -> 300um -> 200um)
    # 光斑越小，Gamma越大，非线性效应越强
    stage_radii   = [0.8e-3, 0.3e-3, 0.2e-3]
    
    # 打印配置信息
    print(f"\n[1/6] 仿真配置确认:")
    total_length = sum(stage_lengths)
    for i, (L, r) in enumerate(zip(stage_lengths, stage_radii)):
        print(f"  Stage {i+1}: 长度={L*1e3:.1f} mm, 半径={r*1e6:.1f} um")
    print(f"  总传播距离: {total_length*1e3:.1f} mm")

    # 基础物理参数
    time_window = 10e-12   # 10ps
    num_t_points = 4096    # 时间点数
    center_wavelength = 1064e-9
    
    # =================================================================
    # 2. 初始化求解器
    # =================================================================
    print(f"\n[2/6] 初始化求解器...")
    # 先用第一段的半径初始化
    solver = ERK43IP_FullDispersion(
        material='fused_silica',
        n2=2.7e-20,
        beam_radius=stage_radii[0], # 初始半径
        center_wavelength=center_wavelength,
        use_raman=True,
        use_self_steepening=True
    )
    
    # 生成时间网格
    t = np.linspace(-time_window/2, time_window/2, num_t_points)
    dt = t[1] - t[0]
    
    # =================================================================
    # 3. 生成初始脉冲
    # =================================================================
    print(f"\n[3/6] 生成初始脉冲...")
    pulse_energy = 1000e-6     # 1000 uJ (1mJ)
    pulse_width_fwhm = 200e-15 # 200 fs
    
    try:
        A_initial = solver.generate_gaussian_pulse(
            t=t,
            pulse_energy=pulse_energy,
            pulse_fwhm=pulse_width_fwhm,
            chirp=0
        )
        
        initial_fwhm = pulse_width([A_initial], t*1e12)[0]
        print(f"  ✅ 脉冲生成: {initial_fwhm:.3f} ps, {pulse_energy*1e6:.1f} uJ")
        
    except Exception as e:
        print(f"  ❌ 脉冲生成失败: {e}")
        return

    # =================================================================
    # 4. 分段传播循环 (核心逻辑)
    # =================================================================
    print(f"\n[4/6] 开始级联传播...")
    
    # 用于存储每一段的数据，最后拼接
    z_history_full = []
    A_history_full = []
    
    current_A = A_initial.copy()
    current_z_offset = 0.0
    
    start_time = time.time()
    
    for i in range(3):
        # 获取当前段参数
        L_stage = stage_lengths[i]
        r_stage = stage_radii[i]
        
        # --- 关键步骤: 动态更新 Gamma ---
        # 模拟理想透镜聚焦：功率(Power)不变，但光斑面积变小，导致强度(Intensity)增加。
        # 在 GNLSE 方程中，这体现在非线性系数 Gamma 变大。
        solver.beam_radius = r_stage
        solver.beam_area = np.pi * r_stage**2
        solver.gamma = (2 * np.pi / solver.lambda0) * (solver.n2 / solver.beam_area)
        
        print(f"  >>> Stage {i+1} (z={current_z_offset*1e3:.1f}-{ (current_z_offset+L_stage)*1e3:.1f} mm):")
        print(f"      半径: {r_stage*1e6:.0f} um | Gamma: {solver.gamma:.6f} /W/m")
        
        # --- 传播计算 ---
        # 注意: 传入 current_A (功率归一化振幅)，无需手动缩放振幅，因为 Power 守恒
        z_seg, A_seg, _ = solver.propagate(
            current_A, t, L=L_stage,
            tol=1e-5, max_step=L_stage/50 # 限制步长以保证精度
        )
        
        # --- 数据记录 ---
        # 1. 记录绝对坐标 (加上之前的偏移量)
        z_seg_absolute = z_seg + current_z_offset
        
        # 2. 存入历史列表
        z_history_full.append(z_seg_absolute)
        A_history_full.append(A_seg)
        
        # 3. 更新下一段的输入
        current_A = A_seg[-1]
        current_z_offset += L_stage
        
    # --- 数据合并 ---
    # 将列表中的多个数组拼接成一个大的长数组，方便统一画图
    z_total = np.concatenate(z_history_full)
    A_total = np.concatenate(A_history_full, axis=0)
    A_out = A_total[-1]
    
    print(f"\n  ✅ 级联传播完成! 耗时: {time.time() - start_time:.2f}s")
    
    # =================================================================
    # 5. 结果可视化
    # =================================================================
    print(f"\n[5/6] 生成可视化结果...")
    
    # 使用适配器包裹拼接后的数据
    sim_adapter = SimResultAdapter(solver, z_total, t, A_total)
    
    # 绘制标准演化图
    try:
        plot_results(sim_adapter, A_total, save_path="cascaded_propagation_results.png")
        print("  ✅ 演化图已保存: cascaded_propagation_results.png")
    except Exception as e:
        print(f"  ⚠️ 绘图警告: {e}")

    # 简要光谱分析
    res = solver.analyze_spectral_phase(A_out, t, fit_order=2, plot_threshold=0.01)
    if res:
        plot_spectral_phase(res)

    # =================================================================
    # 6. 脉冲压缩能力验证 (De-chirp Check)
    # =================================================================
    print(f"\n{'='*70}")
    print("[6/6] 脉冲压缩能力验证 (De-chirp Check)")
    print(f"{'='*70}")
    
    # 扫描 GDD 寻找最佳压缩脉宽
    best_compressed_fwhm = 999.0
    best_gdd = 0.0
    
    # 设置扫描范围: 级联后可能会有较强积累，适当扩大扫描范围
    scan_range = np.linspace(-10000, 5000, 500) * 1e-30 
    
    # 预计算频域场
    omega_axis = 2 * np.pi * np.fft.fftfreq(len(t), t[1] - t[0])
    A_out_freq = np.fft.fft(A_out)
    
    print(f"  正在扫描最佳色散补偿值 ({len(scan_range)} points)...")
    
    for gdd_test in scan_range:
        # 应用相位补偿: exp(i * 0.5 * GDD * w^2)
        phase_comp = np.exp(0.5j * gdd_test * omega_axis**2)
        A_comp_freq = A_out_freq * phase_comp
        A_comp_time = np.fft.ifft(A_comp_freq)
        
        # 测量脉宽
        try:
            wid = pulse_width([A_comp_time], t*1e12)[0]
            if wid < best_compressed_fwhm:
                best_compressed_fwhm = wid
                best_gdd = gdd_test
        except:
            continue
            
    final_fwhm = pulse_width([A_out], t*1e12)[0]
    compression_ratio = final_fwhm / best_compressed_fwhm if best_compressed_fwhm > 0 else 0
    
    print(f"  当前输出脉宽: {final_fwhm:.3f} ps")
    print(f"  最佳压缩脉宽: {best_compressed_fwhm:.3f} ps")
    print(f"  最大压缩比:   {compression_ratio:.2f}x")
    print(f"  所需补偿 GDD: {best_gdd*1e30:.0f} fs²")
    
    if best_compressed_fwhm < initial_fwhm:
        print("  🌟 效果: 脉冲可被压缩至窄于初始宽度 (光谱展宽有效)")
    else:
        print("  ℹ️ 效果: 压缩受限 (可能受高阶色散或非线性相位畸变影响)")

    # =================================================================
    # 7. 生成最终参数报告表
    # =================================================================
    print(f"\n正在生成参数报告...")
    try:
        pulse_params = {
            'energy': pulse_energy,
            'fwhm': initial_fwhm*1e-12, 
            'peak_power': np.max(np.abs(A_initial)**2),
            'chirp': 0,
            'compressed_fwhm': best_compressed_fwhm,
            'req_gdd': best_gdd
        }
        
        draw_parameter_table(sim_adapter, pulse_params, save_path="cascaded_parameter_report.png")
        print("  ✅ 参数表已保存: cascaded_parameter_report.png")
        
    except Exception as e:
        print(f"  ⚠️ 参数表生成失败: {e}")

# =================================================================
# 辅助函数
# =================================================================

def plot_spectral_phase(res):
    """绘制光谱和残留相位"""
    wl = res['wavelength_nm']
    spec = res['spectral_intensity']
    resid = res['residual_phase']
    mask = res['mask']
    
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    color1 = 'tab:blue'
    ax1.plot(wl, spec, color=color1, label='Spectrum')
    ax1.set_ylabel('Norm. Intensity', color=color1)
    ax1.tick_params(axis='y', labelcolor=color1)
    ax1.set_ylim([0, 1.05])
    
    # 自动缩放 X 轴
    valid_wl = wl[mask]
    if len(valid_wl) > 0:
        center = (valid_wl.max() + valid_wl.min()) / 2
        span = (valid_wl.max() - valid_wl.min()) * 2.0
        ax1.set_xlim([center - span/2, center + span/2])
    
    ax2 = ax1.twinx()
    color2 = 'tab:red'
    ax2.plot(wl[mask], resid[mask], color=color2, linewidth=1.5, linestyle='--', label='Resid. Phase')
    ax2.set_ylabel('Phase Residual (rad)', color=color2)
    ax2.tick_params(axis='y', labelcolor=color2)
    
    plt.title('Spectral Phase Analysis (Cascaded Output)')
    plt.tight_layout()
    plt.savefig('spectral_phase_analysis.png', dpi=150)
    print("  ✅ 光谱相位分析图已保存")

if __name__ == "__main__":
    test_cascaded_propagation()