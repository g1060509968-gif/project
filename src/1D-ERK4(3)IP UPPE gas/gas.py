import numpy as np
import matplotlib.pyplot as plt

# 导入你上传的代码库模块
from erk43ip_method import ERK43IP_UPPE, SimResultAdapter
import visualization as vis

# =====================================================================
# 1. 气体折射率与非线性补丁 (严格 P, T 缩放)
# =====================================================================
orig_compute_ri = ERK43IP_UPPE._compute_refractive_index
orig_init = ERK43IP_UPPE.__init__

def patched_init(self, *args, **kwargs):
    # 提取气体专用的 P 和 T 参数 (默认 1 atm = 101325 Pa, 20 C = 293.15 K)
    self.gas_P = kwargs.pop('gas_P', 101325.0)
    self.gas_T = kwargs.pop('gas_T', 293.15)
    orig_init(self, *args, **kwargs)

def patched_compute_ri(self, wavelength):
    if getattr(self, 'material', '') == 'argon_gas':
        # --- 路线 A: 采用 Peck & Fisher (1964) 氩气折射率公式 ---
        # 参考条件: 15 °C (288.15 K), 1 atm (101325 Pa)
        P_ref = 101325.0
        T_ref = 288.15
        
        lam_um = wavelength * 1e6
        sigma_sq = 1.0 / (lam_um ** 2)
        
        # 计算参考条件下的折射率增量 (n0 - 1) * 10^8
        n0_minus_1_10e8 = (1205582.27 / (130.0 - sigma_sq)) + (317222.05 / (38.9 - sigma_sq))
        n0_minus_1 = n0_minus_1_10e8 * 1e-8
        
        # 理想气体密度缩放: rho/rho_0 = (P/P_ref) * (T_ref/T)
        scale = (self.gas_P / P_ref) * (T_ref / self.gas_T)
        
        return 1.0 + n0_minus_1 * scale
    else:
        # 固体介质退回原有 Sellmeier 逻辑
        return orig_compute_ri(self, wavelength)

# 动态应用补丁到求解器类
ERK43IP_UPPE.__init__ = patched_init
ERK43IP_UPPE._compute_refractive_index = patched_compute_ri


def calculate_cavity_mode(L_cavity, R_mirror, lambda0):
    """计算对称双凹腔的本征高斯模式参数"""
    g = 1 - L_cavity / R_mirror
    if not (0 <= g**2 <= 1):
        raise ValueError("腔参数处于不稳定区！")
        
    w0 = np.sqrt((L_cavity * lambda0) / (2 * np.pi) * np.sqrt((1 + g) / (1 - g)))
    zR = np.pi * w0**2 / lambda0
    return w0, zR

# =====================================================================
# 2. 主仿真设置与切片执行
# =====================================================================
def main():
    # --- 物理参数设置 ---
    center_wavelength = 1030e-9  # 1030 nm
    pulse_energy = 160e-6         # 50 uJ
    pulse_fwhm = 275e-15         # 100 fs
    
    # 气体状态 (显著影响色散和非线性)
    P_bar = 7.0                 # 设置为 10 bar 以观察明显的非线性展宽
    T_K = 293.15                 # 室温 20 °C
    P_Pa = P_bar * 1e5
    
    # 氩气基准非线性折射率 (参考: ~1.0e-23 m^2/W @ 1 bar, 293.15K)
    n2_ref = 1.0e-23
    # 根据密度缩放 n2
    n2_scaled = n2_ref * (P_bar / 1.0) * (293.15 / T_K)
    
    # 双凹腔几何
    L_cavity = 0.2865               # 腔长 1 m
    R_mirror = 0.3               # 曲率半径 1 m (共焦)
    
    w0, zR = calculate_cavity_mode(L_cavity, R_mirror, center_wavelength)
    print(f"环境参数: 氩气 {P_bar} bar, {T_K} K")
    print(f"模式参数: 束腰 w0 = {w0*1e6:.1f} um, 瑞利长度 zR = {zR*100:.1f} cm")
    print(f"有效 n2: {n2_scaled:.2e} m^2/W")
    
    # 初始化求解器 (显式传入气压气温，彻底关闭拉曼)
    solver = ERK43IP_UPPE(
        material='argon_gas',
        n2=n2_scaled,
        beam_radius=w0,              # 先传初始值
        center_wavelength=center_wavelength,
        use_raman=False,             # [核心修正] 氩气为单原子，无拉曼响应
        use_self_steepening=True,    # 宽谱时保留自陡峭
        gas_P=P_Pa,                  # 传入补丁所需的气压
        gas_T=T_K                    # 传入补丁所需的温度
    )
    
    # 初始化时间网格和脉冲
    T_window = 4e-12
    Nt = 4096
    t = np.linspace(-T_window/2, T_window/2, Nt)
    A0 = solver.generate_gaussian_pulse(t, pulse_energy, pulse_fwhm)
    
    # --- 切片策略设置 ---
    N_slices = 50  
    z_edges = np.linspace(-L_cavity/2, L_cavity/2, N_slices + 1)
    
    A_current = A0.copy()
    master_z_hist = []
    master_A_hist = []
    
    print(f"\n开始切片传播计算 (共 {N_slices} 段)...")
    
    for i in range(N_slices):
        z_start = z_edges[i]
        z_end = z_edges[i+1]
        dz = z_end - z_start
        z_mid = (z_start + z_end) / 2
        
        # 1. 动态获取当前切片的光斑大小
        w_current = w0 * np.sqrt(1 + (z_mid / zR)**2)
        
        # 2. 更新 solver 内部截面积和 gamma
        solver.beam_radius = w_current
        solver.beam_area = np.pi * w_current**2
        # n2 保持为 n2_scaled 不变，仅有效面积改变导致 gamma 改变
        solver.gamma = (2 * np.pi / solver.lambda0) * (solver.n2 / solver.beam_area)
        
        # 3. 执行单段传播 (缩小 step 确保在气体边界处平滑)
        z_hist, A_hist, omega = solver.propagate(
            A_current, t, L=dz, 
            tol=1e-5, max_step=dz/10, min_step=1e-8
        )
        
        # 4. 数据拼接处理 (转换为全局累积坐标)
        global_z_hist = z_hist + (z_start + L_cavity/2)
        
        if i == 0:
            master_z_hist.extend(global_z_hist[:-1])
            master_A_hist.extend(A_hist[:-1])
        else:
            master_z_hist.extend(global_z_hist[1:-1])
            master_A_hist.extend(A_hist[1:-1])
            
        A_current = A_hist[-1]
        
        if (i+1) % 10 == 0 or i == N_slices - 1:
            print(f"  已完成 {i+1}/{N_slices} 切片, z={z_end:.2f}m, 光斑 w={w_current*1e6:.1f}um, 当前 gamma={solver.gamma:.2e}")

    # 收尾最后一个点
    master_z_hist.append(L_cavity)
    master_A_hist.append(A_current)
    
    final_z_array = np.array(master_z_hist)
    final_A_evolution = np.array(master_A_hist)
    
    print("\n仿真完成，准备可视化...")
    
    # =====================================================================
    # 3. 结果可视化
    # =====================================================================
    adapter = SimResultAdapter(solver, final_z_array, t, final_A_evolution)
    adapter.L = L_cavity
    adapter.z = final_z_array
    adapter.dz = final_z_array[1] - final_z_array[0] if len(final_z_array) > 1 else 0
    
    # 输出可视化图像
    vis.plot_results(adapter, final_A_evolution)

if __name__ == "__main__":
    main()