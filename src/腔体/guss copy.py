import numpy as np

def calculate_air_b_integral():
    print("=" * 60)
    print("多通腔空气 B 积分 (B-integral) 估算")
    print("=" * 60)

    # ===============================
    # 1. 参数设置 (基于之前的仿真参数)
    # ===============================
    
    # 激光参数
    lambda0 = 1030e-9        # 波长 (m)
    energy = 1000e-6          # 脉冲能量 100 uJ
    tau_fwhm = 2200e-15       # 脉宽 250 fs
    
    # 腔体参数 (双凹对称腔)
    L_cavity = 0.5           # 腔长 400 mm
    R_mirror = 0.5           # 曲率 300 mm
    n_passes = 20            # 往返次数
    
    # 晶体参数 (用于扣除空间)
    L_crystal = 5e-3         # 晶体厚度 5 mm (位于中心)
    
    # 空气参数
    # 标准大气压下空气 n2 约为 3e-23 m^2/W (不同文献略有差异)
    n2_air = 3.0e-23         
    
    # ===============================
    # 2. 物理量计算
    # ===============================
    
    # --- A. 峰值功率 ---
    # 对于高斯脉冲 P(t) = P0 * exp(-4*ln2 * t^2 / tau^2)
    # 积分能量 E = P0 * tau * sqrt(pi / (4*ln2))
    # 因此 P0 = E / (tau * factor)
    factor = np.sqrt(np.pi / (4 * np.log(2)))
    peak_power = energy / (tau_fwhm * factor)
    
    print(f"脉冲参数:")
    print(f"  能量: {energy*1e6:.1f} uJ")
    print(f"  脉宽: {tau_fwhm*1e15:.0f} fs")
    print(f"  峰值功率: {peak_power/1e9:.2f} GW")
    
    # --- B. 腔模光斑分布 w(z) ---
    # 稳定性参数 g
    g = 1 - L_cavity / R_mirror
    if abs(g) >= 1:
        print("错误: 腔体不稳定!")
        return

    # 瑞利长度 z_R (假设束腰在中心 z=0)
    # z_R = 1/2 * sqrt( L * (2R - L) )
    z_R = 0.5 * np.sqrt(L_cavity * (2 * R_mirror - L_cavity))
    
    # 束腰半径 w0
    # w0 = sqrt(lambda * z_R / pi)
    w0 = np.sqrt(lambda0 * z_R / np.pi)
    
    print(f"\n腔模参数:")
    print(f"  瑞利长度 z_R: {z_R*1000:.1f} mm")
    print(f"  束腰半径 w0:  {w0*1e6:.1f} um")
    
    # 定义光斑函数 w(z)
    def beam_radius(z):
        return w0 * np.sqrt(1 + (z / z_R)**2)
    
    # 定义光强函数 I(z)
    def intensity(z):
        w_z = beam_radius(z)
        area = np.pi * w_z**2
        return peak_power / area
    
    # ===============================
    # 3. B 积分计算
    # ===============================
    # B = (2*pi / lambda) * n2 * integral( I(z) dz )
    
    # 积分范围: 
    # 整个腔是从 -L/2 到 +L/2
    # 晶体占据 -L_c/2 到 +L_c/2
    # 空气范围是 [-L/2, -L_c/2] 和 [+L_c/2, +L/2]
    # 由于对称性，我们算一半乘以 2 即可
    
    z_mirror = L_cavity / 2.0
    z_crystal_surface = L_crystal / 2.0
    
    # 使用数值积分 (梯形法则)
    num_points = 1000
    z_axis = np.linspace(z_crystal_surface, z_mirror, num_points)
    I_axis = intensity(z_axis)
    
    # 积分 I(z) dz (单侧空气)
    integral_I_one_side = np.trapz(I_axis, z_axis)
    
    # 计算 B 值
    k0 = 2 * np.pi / lambda0
    B_one_side = k0 * n2_air * integral_I_one_side
    
    # 单次通过 (Single Pass) = 左侧 + 右侧
    B_single_pass = B_one_side * 2
    
    # 总 B 积分
    B_total = B_single_pass * n_passes
    
    # ===============================
    # 4. 结果输出
    # ===============================
    print(f"\n计算结果 (空气介质, n2={n2_air:.1e}):")
    print(f"--------------------------------------------------")
    print(f"单次通过 (Single Pass) B 积分:")
    print(f"  B_air ≈ {B_single_pass:.4f} rad")
    
    print(f"\n总积累 ({n_passes} Passes) B 积分:")
    print(f"  B_total ≈ {B_total:.4f} rad")
    print(f"--------------------------------------------------")
    
    # 阈值提示
    print("\n[参考阈值]")
    print("一般认为 B_total < 3~5 rad 是安全的，以避免严重的自聚焦或光束质量退化。")
    if B_total > 3.0:
        print("⚠️ 警告: 空气 B 积分较高，可能影响最终压缩效果或导致光束在空气中崩塌。")
    else:
        print("✅ 状态: 空气 B 积分在安全范围内。")

if __name__ == "__main__":
    calculate_air_b_integral()