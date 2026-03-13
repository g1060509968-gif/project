"""
色散脉冲生成测试脚本 - 使用 ERK43IP_FullDispersion 求解器
调用 generate_dispersed_pulse 方法，GDD=0, TOD=0
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fftfreq

# 导入必要模块
from erk43ip_method import ERK43IP_FullDispersion
from visualization import plot_results, pulse_width, analyze_results

# 设置中文字体支持 (防止绘图乱码)
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def test_dispersed_pulse_zero_dispersion():
    """测试 generate_dispersed_pulse (零色散情况)"""
    print("=" * 70)
    print("色散脉冲生成测试 (GDD=0, TOD=0) - ERK43IP 求解器")
    print("=" * 70)

    # 1. 设置仿真参数 (保持与原示例一致)
    crystal_length = 120e-3   # 1.182m 晶体长度
    time_window = 50e-12       # 50ps 时间窗口
    num_t_points = 4096*2        # 时间采样点数
    center_wavelength = 1064e-9 # 中心波长
    beam_radius = 0.4e-3      # 光斑半径
    
    print(f"\n[1/5] 初始化 ERK43IP 求解器...")
    
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
    
    print(f"\n[2/5] 生成零色散高斯脉冲 (频域相位法)...")
    
    # 4. 生成脉冲 (使用新方法 generate_dispersed_pulse)
    pulse_energy = 2000e-6     # 200 µJ
    pulse_width_fwhm = 200e-15  # 10ps
    target_gdd_fs2 = 150000
    #target_gdd_fs2 = 75000
    # === 关键修改点 ===
    try:
        A_initial = solver.generate_dispersed_pulse(
            t=t,
            pulse_energy=pulse_energy,
            pulse_fwhm=pulse_width_fwhm,
            GDD=target_gdd_fs2 * 1e-30,  # 二阶色散设为 0
            TOD=0   # 三阶色散设为 0
        )
        
        # 验证生成的脉冲
        initial_power = np.abs(A_initial)**2
        peak_power = np.max(initial_power)
        print(f"  ✅ 脉冲生成成功")
        print(f"    输入能量: {pulse_energy*1e6:.1f} µJ")
        print(f"    设定 GDD: 0 fs²")
        print(f"    设定 TOD: 0 fs³")
        print(f"    生成峰值功率: {peak_power/1e3:.2f} kW")
        
    except AttributeError:
        print("❌ 错误: ERK43IP_FullDispersion 类中找不到 'generate_dispersed_pulse' 方法。")
        print("   请确保你已经将该方法添加到了 erk43ip_method.py 文件中。")
        return
    except Exception as e:
        print(f"❌ 生成失败: {e}")
        return

    # 5. 运行传播仿真
    print(f"\n[3/5] 运行传播仿真...")
    try:
        z_array, A_evolution, omega = solver.propagate(
            A_initial, t, L=crystal_length, 
            tol=1e-5, max_step=1e-2
        )
        print(f"  ✅ 传播完成: {len(z_array)} 步")
        
    except Exception as e:
        print(f"❌ 传播仿真失败: {e}")
        return

    # 6. 准备可视化数据适配器
    # visualization.py 需要一个包含特定属性的对象，这里创建一个简单的适配器
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
            # 添加 analyze_results 需要的辅助方法引用 (如果有的话)
            self.gamma = solver.gamma 
            if hasattr(solver, '_compute_energy'):
                self._compute_energy = lambda A: np.trapz(np.abs(A)**2, t)
            
    sim_adapter = SimAdapter(z_array, t, A_evolution, solver)

    # 7. 绘制结果
    print(f"\n[4/5] 绘制结果...")
    try:
        plot_results(sim_adapter, A_evolution, 
                    save_path="test_dispersed_pulse_zero.png")
        print("  ✅ 图像已保存为 test_dispersed_pulse_zero.png")
    except Exception as e:
        print(f"  ⚠️ 绘图出现警告或错误: {e}")

    # 8. 简单分析
    print(f"\n[5/5] 结果分析...")
    # 手动计算并在控制台输出关键指标
    P_in = np.abs(A_evolution[0])**2
    P_out = np.abs(A_evolution[-1])**2
    fwhm_in = pulse_width([A_evolution[0]], t*1e12)[0]
    fwhm_out = pulse_width([A_evolution[-1]], t*1e12)[0]
    
    print(f"  输入脉宽: {fwhm_in:.3f} ps")
    print(f"  输出脉宽: {fwhm_out:.3f} ps")
    print(f"  展宽比:   {fwhm_out/fwhm_in:.3f}")
    
    # =================================================================
    # 新增：相位残差分析
    # =================================================================
    print(f"\n{'='*70}")
    print("相位残差分析")
    print(f"{'='*70}")
    
    A_out = A_evolution[-1]
    
    # 计算残差 
    # threshold=0.01 表示只看 >1% 强度的部分
    res = solver.analyze_spectral_phase(A_out, t, fit_order=1, plot_threshold=0.01)
    
    if res:
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
        plt.savefig("dispersed_pulse_phase_analysis.png", dpi=150, bbox_inches='tight')
        plt.show()
        print("  ✅ 相位残差图像已保存为 dispersed_pulse_phase_analysis.png")
    
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
        draw_parameter_table(adapter, pulse_params, save_path="dispersed_pulse_parameter_report.png")
        print("  ✅ 参数表已保存为 dispersed_pulse_parameter_report.png")
        
    except ImportError as e:
        print(f"  ⚠️ 导入失败: {e}")
        print("  请确保已正确添加 SimResultAdapter 类和 draw_parameter_table 函数")
    except Exception as e:
        print(f"  ⚠️ 参数表功能测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_dispersed_pulse_zero_dispersion()
