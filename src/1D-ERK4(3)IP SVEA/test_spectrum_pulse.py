"""
从 CSV 读取光谱并进行脉冲传播测试
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import os

# 导入 RK4IP 模块
from erk43ip_method import ERK43IP_FullDispersion
from visualization import plot_results, pulse_width

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

def load_spectrum_csv(filename):
    """
    读取 CSV 文件中的波长和强度数据
    兼容格式: 第一列为波长(nm), 第二列为强度
    会自动过滤掉非数字的行 (如 等标签)
    """
    wavelengths = []
    intensities = []
    
    print(f"正在读取光谱文件: {filename} ...")
    
    with open(filename, 'r', encoding='utf-8') as f:
        reader = csv.reader(f)
        for row in reader:
            # 跳过空行
            if not row: continue
            
            # 尝试解析数字，处理可能存在的文本干扰
            try:
                # 假设第一列是波长，第二列是强度
                # 移除可能存在的 等前缀，只保留数字部分
                clean_row = []
                for item in row:
                    # 简单的清洗逻辑：尝试转换，如果不行为空
                    item_str = item.strip()
                    # 如果包含 source 这种文字，尝试分割获取后面的数字
                    if "]" in item_str:
                        item_str = item_str.split("]")[-1].strip()
                    if item_str:
                        clean_row.append(float(item_str))
                
                if len(clean_row) >= 2:
                    wavelengths.append(clean_row[0])
                    intensities.append(clean_row[1])
            except ValueError:
                # 如果这一行无法转换为数字，则跳过（例如表头）
                continue
                
    return np.array(wavelengths), np.array(intensities)

def test_spectrum_pulse():
    print("=" * 70)
    print("光谱重建脉冲传播测试")
    print("=" * 70)

    # 1. 设置仿真参数
    crystal_length = 3.168  # m
    time_window = 50e-12    # 50ps
    num_t_points = 8192     # 增加点数以获得更好的频率分辨率
    center_wavelength = 1064e-9
    beam_radius = 0.35e-3
    
    # 2. 初始化求解器
    solver = ERK43IP_FullDispersion(
        material='fused_silica',
        n2=2.7e-20,
        beam_radius=beam_radius,
        center_wavelength=center_wavelength,
        use_raman=True,
        use_self_steepening=True
    )

    # 3. 生成时间网格
    t = np.linspace(-time_window/2, time_window/2, num_t_points)
    
    # ==========================================
    # 核心修改：读取 CSV 并生成脉冲
    # ==========================================
    csv_filename = "input_spectrum.csv"  # 使用当前目录下的文件
    
    # 检查文件是否存在
    if not os.path.exists(csv_filename):
        print(f"错误: 找不到文件 {csv_filename}")
        return

    # 读取数据
    wl_data, int_data = load_spectrum_csv(csv_filename)
    
    if len(wl_data) == 0:
        print("错误: 未从 CSV 中读取到有效数据")
        return

    print(f"读取成功: 包含 {len(wl_data)} 个数据点")
    print(f"波长范围: {np.min(wl_data):.2f} nm - {np.max(wl_data):.2f} nm")

    # 设定目标脉冲能量 (光谱只有形状信息，没有能量信息)
    target_energy = 200e-6 # 200 uJ
    target_gdd_fs2 = 5000000

    # 重建脉冲
    try:
        A_initial = solver.generate_pulse_from_spectrum(
            t=t,
            wavelengths_nm=wl_data,
            intensities=int_data,
            pulse_energy=target_energy,
            GDD=target_gdd_fs2 * 1e-30, 
            TOD=0
        )
        
        # 计算初始脉冲宽度
        initial_fwhm = pulse_width([A_initial], t*1e12)[0]
        print(f"  ✅ 脉冲重建成功")
        print(f"    重建后的脉冲宽度: {initial_fwhm:.3f} ps")
        
    except Exception as e:
        print(f"  ❌ 脉冲重建失败: {e}")
        return

    # 5. 运行传播仿真 (同原有逻辑)
    print(f"\n[运行传播仿真...]")
    z_array, A_evolution, omega = solver.propagate(
        A_initial, t, L=crystal_length, 
        tol=1e-5, max_step=1e-2
    )

    # 6. 可视化适配器
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
    try:
        plot_results(sim_adapter, A_evolution, save_path="spectrum_pulse_results.png")
        print("绘图完成，已保存为 spectrum_pulse_results.png")
    except Exception as e:
        print(f"绘图模块报错 (可能是缺少 visualization.py): {e}")
        # 这里可以使用简单的 matplotlib 绘图作为备用
        plt.figure()
        plt.plot(t*1e12, np.abs(A_initial)**2, label='Initial Pulse')
        plt.plot(t*1e12, np.abs(A_evolution[-1])**2, label='Output Pulse')
        plt.xlabel('Time (ps)')
        plt.ylabel('Power (W)')
        plt.legend()
        plt.show()
        # ... 仿真结束后 ...
    A_out = A_evolution[-1]

    # 计算残差 
    # threshold=0.01 表示只看 >1% 强度的部分
    res = solver.analyze_spectral_phase(A_out, t, fit_order=3, plot_threshold=0.01)

    if res:
        import matplotlib.pyplot as plt
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
        plt.savefig("spectrum_pulse_phase_analysis.png", dpi=150, bbox_inches='tight')
        plt.show()
        print("  ✅ 相位残差图像已保存为 spectrum_pulse_phase_analysis.png")
    
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
            'energy': target_energy,      # 你的设定能量
            'fwhm': initial_fwhm,         # 重建后的脉冲宽度
            'peak_power': np.max(np.abs(A_evolution[0])**2), # 实际计算的峰值功率
            'chirp': 0              # 你的设定 chirp
        }
        
        # 3. 绘制表格
        print("正在绘制仿真参数表...")
        draw_parameter_table(adapter, pulse_params, save_path="spectrum_pulse_parameter_report.png")
        print("  ✅ 参数表已保存为 spectrum_pulse_parameter_report.png")
        
    except ImportError as e:
        print(f"  ⚠️ 导入失败: {e}")
        print("  请确保已正确添加 SimResultAdapter 类和 draw_parameter_table 函数")
    except Exception as e:
        print(f"  ⚠️ 参数表功能测试失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_spectrum_pulse()
