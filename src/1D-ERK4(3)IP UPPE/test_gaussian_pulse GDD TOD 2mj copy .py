"""
双凹型多通腔 (MPC) 非线性压缩仿真脚本 (修复增强版)
基于 ERK43IP_FullDispersion 求解器

更新内容:
1. 修复空气段光斑计算：空气段现在也会根据距离动态计算光斑大小并更新 Gamma。
2. 支持晶体位置移动：通过 crystal_offset 参数调整晶体相对于束腰的位置。
3. 增加 TOD 支持：镜片反射加入三阶色散项。
4. 性能优化：优化数据记录频率，防止内存溢出。
"""

import numpy as np
import matplotlib.pyplot as plt
import csv
import time
from scipy.fft import fft, ifft, fftfreq, fftshift

# 导入必要模块 (假设这些文件在同一目录下)
from erk43ip_method import ERK43IP_FullDispersion, SimResultAdapter
from visualization import plot_results, pulse_width, analyze_results, draw_parameter_table

# 设置绘图字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# ==============================================================================
# 1. 扩展求解器以支持空气色散
# ==============================================================================
class MPCSolver(ERK43IP_FullDispersion):
    """扩展版求解器，增加空气材料支持，并修复系数长度不匹配问题"""
    
    def _get_sellmeier_coeffs(self, material):
        if material == 'air':
            # 空气的近似 Sellmeier 系数 (Ciddor equation simplified)
            return {
                'B': [0.00004365879, 0.0002298715], # 只有2项
                'C': [0.0634289**2, 0.171804**2],   # 只有2项
                'valid_range': (0.2e-6, 2.0e-6)
            }
        return super()._get_sellmeier_coeffs(material)

    def _compute_refractive_index(self, wavelength):
        coeffs = self.sellmeier_coeffs
        B, C = coeffs['B'], coeffs['C']
        
        wl_um = wavelength * 1e6
        wl_sq = wl_um**2
        n_squared = np.ones_like(wavelength, dtype=float)
        
        # 使用 len(B) 动态适应不同项数的 Sellmeier 方程
        for i in range(len(B)):
            n_squared += (B[i] * wl_sq) / (wl_sq - C[i])
            
        return np.sqrt(n_squared)

# ==============================================================================
# 2. 高斯光束计算工具
# ==============================================================================
class GaussianBeam:
    def __init__(self, wavelength, cavity_length, R_mirror):
        self.lam = wavelength
        self.L = cavity_length
        self.R = R_mirror
        
        # 验证稳定性条件
        g = 1 - self.L / self.R
        if abs(g) > 1:
            raise ValueError(f"腔体不稳定! g = {g:.2f} (需要 |g| <= 1)")
            
        # 计算束腰半径 w0 (假设束腰在腔中心 z=0)
        # 瑞利长度 z_R = sqrt( (L(2R-L)) / 4 )
        term = (self.L * (2 * self.R - self.L))
        if term < 0:
             raise ValueError("参数导致复数瑞利长度，请检查 L 和 R")
        self.z_R = np.sqrt(term) / 2
        
        # w0 = sqrt(lambda * z_R / pi)
        self.w0 = np.sqrt(self.lam * self.z_R / np.pi)
        
        print(f"--- 腔模参数 ---")
        print(f"  腔长 L: {self.L*1e3:.1f} mm")
        print(f"  曲率 R: {self.R*1e3:.1f} mm")
        print(f"  束腰 w0: {self.w0*1e6:.1f} um (位于 z=0)")
        print(f"  瑞利长度 zR: {self.z_R*1e3:.1f} mm")
        print(f"----------------")

    def radius_at(self, z_distance_from_waist):
        """计算距离束腰 z 处的光斑半径 w(z)"""
        return self.w0 * np.sqrt(1 + (z_distance_from_waist / self.z_R)**2)

# ==============================================================================
# 3. 主仿真逻辑
# ==============================================================================
def test_multipass_cavity():
    print("=" * 70)
    print("多通腔 (MPC) 非线性光谱展宽仿真 (增强版)")
    print("=" * 70)

    # --- A. 基础参数设置 ---
    
    # 1. 激光参数
    center_wavelength = 1064e-9  # 中心波长
    pulse_energy = 2000e-6        # 单脉冲能量 (J) [注意：保持100uJ以确保稳定，若需1mJ请修改此处]
    pulse_fwhm = 200e-15         # 脉冲宽度 (s)
    input_GDD = 141000 * 1e-30            # 初始啁啾 (s^2)
    
    # 2. 腔体几何参数
    n_passes = 54               # 总通过次数
    L_cavity = 0.5           # 腔长 (m)
    R_mirror = 1                # 镜面曲率半径 (m)
    
    # 3. 介质参数与位置 [修改点: 支持 Offset]
    material_type = 'fused_silica'
    L_crystal = 3e-3             # 晶体厚度 (m)
    crystal_offset = 0        # 晶体中心相对于腔中心(束腰)的偏移量 (m)，正向靠近右侧镜片
    n2_crystal = 2.7e-20         # 晶体非线性系数
    
    # 4. 空气参数 [修改点: 支持空气非线性]
    n2_air = 3e-23                 # 空气非线性系数 (如需开启可设为 3e-23 左右)
    
    # 5. 镜片损耗与色散 [修改点: 增加 TOD]
    mirror_loss = 0.005          # 单次反射损耗 (0.5%)
    mirror_GDD = -50e-30         # 单次反射二阶色散 (fs^2)
    mirror_TOD = 0.0             # 单次反射三阶色散 (fs^3) [文件名提到的TOD在此设置]
    
    # 6. 仿真精度 (切片数)
    crystal_slices = 5           # 晶体分层数
    air_slices = 10              # 空气分层数 (单侧)，用于动态计算光斑
    
    # 7. 时间网格
    time_window = 10e-12          # 时间窗口
    num_t_points = 4096          # 时间点数
    
    # --- B. 初始化 ---
    
    # 计算高斯光束分布
    gb = GaussianBeam(center_wavelength, L_cavity, R_mirror)
    
    # 初始化求解器
    # 注意：这里的 gamma 后续会在循环中动态覆盖，初始化值仅占位
    solver_air = MPCSolver(
        material='air', n2=n2_air, beam_radius=1.0, 
        center_wavelength=center_wavelength, use_raman=False
    )
    
    solver_crystal = MPCSolver(
        material=material_type, n2=n2_crystal, beam_radius=gb.w0,
        center_wavelength=center_wavelength, use_raman=True
    )
    
    # 生成时间轴和初始脉冲
    t = np.linspace(-time_window/2, time_window/2, num_t_points)
    # GDD 和 TOD 可以在生成时加入，这里假设初始脉冲只有 GDD
    A_current = solver_crystal.generate_dispersed_pulse(
        t, pulse_energy, pulse_fwhm, GDD=input_GDD
    )
    
    # 数据记录容器
    total_z_history = [0.0]
    total_A_history = [A_current.copy()]
    current_z_cumulative = 0.0
    
    # --- C. 几何计算 (含 Offset) ---
    # 坐标系：z=0 为腔中心(束腰位置)
    # 腔体范围: [-L/2, +L/2]
    # 晶体中心物理位置: z = crystal_offset
    # 晶体范围: [offset - Lc/2, offset + Lc/2]
    
    half_crystal = L_crystal / 2.0
    z_crystal_front = crystal_offset - half_crystal
    z_crystal_back = crystal_offset + half_crystal
    z_cavity_left = -L_cavity / 2.0
    z_cavity_right = L_cavity / 2.0
    
    # 计算两侧空气段长度
    L_air_left = z_crystal_front - z_cavity_left
    L_air_right = z_cavity_right - z_crystal_back
    
    if L_air_left < 0 or L_air_right < 0:
        raise ValueError(f"晶体偏移过大或太厚，超出了腔体范围! Offset={crystal_offset}")
        
    print(f"几何布局:")
    print(f"  Mirror 1 -> Air ({L_air_left*1000:.1f}mm) -> Crystal ({L_crystal*1000:.1f}mm) -> Air ({L_air_right*1000:.1f}mm) -> Mirror 2")
    print(f"  晶体中心位置: z={crystal_offset*1000:.1f} mm (相对于束腰)")

    # --- D. 多通循环仿真 ---
    print(f"\n开始多通传播仿真 (共 {n_passes} Passes)...")
    start_time = time.time()

    # 预计算频率轴用于镜片色散
    dt = t[1] - t[0]
    freqs = fftfreq(len(t), dt)
    omega = 2 * np.pi * freqs

    for p in range(n_passes):
        print(f"  Pass {p+1}/{n_passes} ...", end="\r")
        
        # =====================================================
        # Segment 1: 左侧空气 (Mirror 1 -> Crystal Front)
        # =====================================================
        # 物理坐标从 z_cavity_left 到 z_crystal_front
        air_left_coords = np.linspace(z_cavity_left, z_crystal_front, air_slices + 1)
        dz_air_left = L_air_left / air_slices
        
        for i in range(air_slices):
            # 当前切片中心在腔坐标系的位置
            z_abs_center = (air_left_coords[i] + air_left_coords[i+1]) / 2.0
            
            # 1. 根据位置计算光斑 (radius_at 接受距离束腰的距离)
            w_local = gb.radius_at(z_abs_center)
            
            # 2. 更新空气 Gamma (如果 n2_air > 0)
            if n2_air > 0:
                solver_air.gamma = (2 * np.pi / center_wavelength) * (n2_air / (np.pi * w_local**2))
            else:
                solver_air.gamma = 0.0
                
            # 3. 传播
            z_step_out, A_step_out, _ = solver_air.propagate(
                A_current, t, L=dz_air_left, tol=1e-3, max_step=dz_air_left
            )
            A_current = A_step_out[-1]
            current_z_cumulative += dz_air_left
            
        # 记录每段结束点 (节省内存)
        total_z_history.append(current_z_cumulative)
        total_A_history.append(A_current.copy())

        # =====================================================
        # Segment 2: 晶体 (Crystal Front -> Crystal Back)
        # =====================================================
        crystal_coords = np.linspace(z_crystal_front, z_crystal_back, crystal_slices + 1)
        dz_crystal = L_crystal / crystal_slices
        
        for i in range(crystal_slices):
            z_abs_center = (crystal_coords[i] + crystal_coords[i+1]) / 2.0
            w_local = gb.radius_at(z_abs_center)
            
            # 更新晶体 Gamma
            new_gamma = (2 * np.pi / center_wavelength) * (n2_crystal / (np.pi * w_local**2))
            solver_crystal.gamma = new_gamma
            
            z_step_out, A_step_out, _ = solver_crystal.propagate(
                A_current, t, L=dz_crystal, tol=1e-4, max_step=dz_crystal
            )
            A_current = A_step_out[-1]
            current_z_cumulative += dz_crystal
            
            # 晶体内部变化剧烈，建议记录每一步
            total_z_history.append(current_z_cumulative)
            total_A_history.append(A_current.copy())

        # =====================================================
        # Segment 3: 右侧空气 (Crystal Back -> Mirror 2)
        # =====================================================
        air_right_coords = np.linspace(z_crystal_back, z_cavity_right, air_slices + 1)
        dz_air_right = L_air_right / air_slices
        
        for i in range(air_slices):
            z_abs_center = (air_right_coords[i] + air_right_coords[i+1]) / 2.0
            w_local = gb.radius_at(z_abs_center)
            
            if n2_air > 0:
                solver_air.gamma = (2 * np.pi / center_wavelength) * (n2_air / (np.pi * w_local**2))
            else:
                solver_air.gamma = 0.0
            
            z_step_out, A_step_out, _ = solver_air.propagate(
                A_current, t, L=dz_air_right, tol=1e-3, max_step=dz_air_right
            )
            A_current = A_step_out[-1]
            current_z_cumulative += dz_air_right

        # 记录每段结束点
        total_z_history.append(current_z_cumulative)
        total_A_history.append(A_current.copy())
        
        # =====================================================
        # Mirror Interaction (反射 + 损耗 + GDD + TOD)
        # =====================================================
        # 1. 损耗
        A_current = A_current * np.sqrt(1 - mirror_loss)
        
        # 2. 色散补偿 (GDD + TOD)
        # phase = 0.5 * GDD * w^2 + (1/6) * TOD * w^3
        phase_mirror = 0.5 * mirror_GDD * omega**2 + (1.0/6.0) * mirror_TOD * omega**3
        
        A_freq = fft(A_current)
        A_freq = A_freq * np.exp(1j * phase_mirror)
        A_current = ifft(A_freq)

    print(f"\n仿真结束。总耗时: {time.time()-start_time:.2f}s")
    
    # 构造结果适配器 (用于绘图)
    A_final_out = A_current
    
    class DummySolver: pass
    dummy = DummySolver()
    dummy.material = material_type
    dummy.lambda0 = center_wavelength
    dummy.omega0 = 2 * np.pi * 3e8 / center_wavelength
    dummy.c = 3e8
    dummy.beam_radius = gb.w0
    dummy.n2 = n2_crystal
    dummy.gamma = solver_crystal.gamma 
    dummy.get_beta2 = lambda: 0.0
    
    sim_adapter = SimResultAdapter(dummy, np.array(total_z_history), t, np.array(total_A_history))
    
    # --- E. 结果可视化 ---
    print(f"\n[结果处理] 绘制演化图...")
    try:
        plot_results(sim_adapter, np.array(total_A_history), 
                    save_path="mpc_evolution.png")
    except Exception as e:
        print(f"绘图警告: {e}")

    # --- F. 脉冲压缩检查 (De-chirp Check) ---
    print(f"\n[脉冲压缩] 寻找最佳色散补偿...")
    best_fwhm = 999.0
    best_gdd = 0.0
    
    # 扫描 GDD 寻找最短脉冲
    scan_range = np.linspace(-50000, 10000, 1000) * 1e-30 
    
    A_out_freq = fft(A_final_out)
    for gdd_test in scan_range:
        # 这里只做简单的 GDD 补偿扫描，通常压缩级主要调节 GDD
        phase_comp = np.exp(1j * 0.5 * gdd_test * omega**2)
        A_temp = ifft(A_out_freq * phase_comp)
        try:
            wid = pulse_width([A_temp], t*1e12)[0]
            if not np.isnan(wid) and wid < best_fwhm:
                best_fwhm = wid
                best_gdd = gdd_test
        except:
            continue
            
    final_fwhm = pulse_width([A_final_out], t*1e12)[0]
    print(f"  直接输出脉宽: {final_fwhm:.3f} ps")
    print(f"  压缩后最短脉宽: {best_fwhm*1000:.1f} fs")
    print(f"  最佳补偿 GDD: {best_gdd*1e30:.0f} fs^2")

    # --- G. 生成参数表 ---
    try:
        final_energy = np.trapz(np.abs(A_final_out)**2, t)
        params = {
            'energy': final_energy,
            'fwhm': pulse_fwhm, 
            'peak_power': np.max(np.abs(A_final_out)**2),
            'note': f"MPC {n_passes} Passes\nOffset: {crystal_offset*1000}mm\nComp: {best_fwhm*1e12:.3f}ps"
        }
        draw_parameter_table(sim_adapter, params, save_path="mpc_report.png")
    except Exception as e:
        print(f"表生成失败: {e}")

    # --- H. 导出光谱 CSV ---
    output_csv = "mpc_output_spectrum.csv"
    try:
        freqs_shift = fftshift(fftfreq(len(t), dt))
        spec = fftshift(fft(A_final_out))
        intensity = np.abs(spec)**2
        
        # 频率转波长
        f_abs = freqs_shift + (3e8 / center_wavelength)
        valid = f_abs > 0
        wl_nm = (3e8 / f_abs[valid]) * 1e9
        inte_valid = intensity[valid] * (f_abs[valid]**2) # Jacobian correction
        
        # 归一化
        inte_norm = inte_valid / np.max(inte_valid)
        
        # 排序并写入
        sort_idx = np.argsort(wl_nm)
        
        with open(output_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(["Wavelength (nm)", "Normalized Intensity"])
            for w, i in zip(wl_nm[sort_idx], inte_norm[sort_idx]):
                if 600 <= w <= 1600: # 限制输出范围
                    writer.writerow([f"{w:.6f}", f"{i:.8e}"])
        print(f"  光谱数据已保存至 {output_csv}")
    except Exception as e:
        print(f"导出失败: {e}")

if __name__ == "__main__":
    test_multipass_cavity()