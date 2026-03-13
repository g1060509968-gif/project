import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import decimal
import math

# --- 1. 全局绘图设置 ---
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# --- 2. Herriott Cell 几何计算函数 ---

def evaluate_physics(pulse_energy_J, pulse_width_s, w0_mm, wm_mm):
    print("\n--- 物理参数评估 ---")
    w0_cm = w0_mm / 10.0
    wm_cm = wm_mm / 10.0
    
    # 峰值功率 (假设高斯脉冲)
    P_peak = 0.94 * pulse_energy_J / pulse_width_s
    print(f"峰值功率 P_peak: {P_peak/1e6:.2f} MW")
    
    # 镜面处的能量密度 (Fluence)
    fluence_mirror = 2 * pulse_energy_J / (np.pi * wm_cm**2) # J/cm^2
    print(f"镜面能量密度: {fluence_mirror:.4f} J/cm^2")
    
    # 焦点处的峰值光强 (Intensity)
    intensity_focus = 2 * P_peak / (np.pi * w0_cm**2) # W/cm^2
    print(f"焦点峰值光强: {intensity_focus/1e12:.4f} TW/cm^2")

def calculate_incident_angles(x0, y0, f, d):
    """计算入射角度以保证闭合轨迹"""
    if d >= 4 * f: raise ValueError(f"系统不稳定: d={d} >= 4*f={4*f}")
    if abs(x0 - y0) > 1e-10: raise ValueError("x0 和 y0 必须相等")
    decimal.getcontext().prec = 50
    x0_d, f_d, d_d = decimal.Decimal(str(x0)), decimal.Decimal(str(f)), decimal.Decimal(str(d))
    a = (4 * f_d**2) / (x0_d**2)
    b = (4 * f_d) / x0_d
    c = -(4 * f_d) / d_d + 2
    discriminant = b**2 - 4 * a * c
    if discriminant < 0: raise ValueError("无实数解：无法形成圆形光斑")
    sqrt_discriminant = discriminant.sqrt()
    y_p1 = (-b + sqrt_discriminant) / (2 * a)
    y_p2 = (-b - sqrt_discriminant) / (2 * a)
    x_p1 = -y_p1 - x0_d / f_d
    x_p2 = -y_p2 - x0_d / f_d
    return [(float(x_p1), float(y_p1)), (float(x_p2), float(y_p2))]

def calculate_spot_positions(x0, x_prime0, f, d, n_spots=20):
    """计算镜面上光斑的坐标"""
    cos_theta = 1 - d / (2 * f)
    cos_theta = np.clip(cos_theta, -1.0, 1.0)
    theta = np.arccos(cos_theta)
    
    A_squared = (4 * f / (4 * f - d)) * (x0**2 + d * x0 * x_prime0 + d * f * x_prime0**2)
    A = np.sqrt(A_squared)
    tan_alpha_num = np.sqrt(4 * f / d - 1)
    tan_alpha_den = 1 + 2 * f * (x_prime0 / x0)
    alpha = np.arctan2(tan_alpha_num, tan_alpha_den) if abs(tan_alpha_den) > 1e-10 else np.pi / 2
    n_values = np.arange(n_spots)
    angles = n_values * theta + alpha
    return A * np.sin(angles), A * np.cos(angles)

def find_configs_by_spot_size(f, lambda_nm, target_wm, tolerance=0.01, min_nu=10, max_nu=50, max_mu_limit=None):
    """
    搜索满足特定镜面光斑大小 (wm) 的所有腔型配置。
    控制参数:
    - min_nu, max_nu: 单镜光斑数 ν 的范围
    - max_mu_limit: 最大绕行圈数 μ 的限制
    """
    R = 2 * f
    lambda_mm = lambda_nm * 1e-6
    results = []

    mu_limit_str = f", μ <= {max_mu_limit}" if max_mu_limit else ", 无 μ 限制"
    print(f"\n====== 正在搜索目标 wm ≈ {target_wm} mm 的解 (ν={min_nu}~{max_nu}{mu_limit_str}) ======")
    
    # [修改点] 直接遍历单镜光斑数 v (ν)
    for v in range(min_nu, max_nu + 1):
        N = 2 * v # 标准 Herriott Cell 总次数是 v 的两倍
        
        # 确定 mu 的搜索范围
        search_range_end = v
        if max_mu_limit is not None:
            search_range_end = min(v, max_mu_limit + 1)
            
        # 遍历绕行圈数 mu
        for mu in range(1, search_range_end):
            # 1. 计算几何参数
            theta = mu * np.pi / v
            d_f_ratio = 2 * (1 - np.cos(theta))
            d_val = d_f_ratio * f
            L_plano = d_val / 2.0
            
            # 2. 计算物理参数 C = d/R
            C = d_val / R
            
            # 稳定性检查 (0 < C < 2)
            if C <= 0.001 or C >= 1.999:
                continue
            
            # 3. 计算光斑半径
            # wm (镜面/最大)
            wm_sq = (R * lambda_mm) / np.pi * np.sqrt(C / (2 - C))
            wm = np.sqrt(wm_sq)
            
            # w0 (腰斑/最小)
            w0_sq = (R * lambda_mm) / (2 * np.pi) * np.sqrt(C * (2 - C))
            w0 = np.sqrt(w0_sq)
            
            # 4. 筛选符合目标光斑大小的解
            if abs(wm - target_wm) <= tolerance:
                common_divisor = math.gcd(mu, v)
                is_coprime = (common_divisor == 1)
                
                results.append({
                    'N': N,
                    'v': v,
                    'mu': mu,
                    'gcd': common_divisor,
                    'is_coprime': is_coprime,
                    'd': d_val,
                    'L_plano': L_plano,
                    'w0': w0,
                    'wm': wm,
                    'diff': abs(wm - target_wm)
                })

    # 按与目标值的差值排序
    results.sort(key=lambda x: x['diff'])
    
    # 打印表格
    header = f"{'Idx':<4} | {'ν (光斑数)':<10} | {'模式(ν,μ)':<10} | {'d (双凹)':<10} | {'L (平凹)':<10} | {'w0 (mm)':<8} | {'wm (mm)':<8} | {'Type':<6}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))
    
    for i, res in enumerate(results):
        type_str = "标准" if res['is_coprime'] else f"重入(/{res['gcd']})"
        print(f"{i:<4} | {res['v']:<10} | ({res['v']}, {res['mu']}):<10 | {res['d']:.2f}{'':<4} | {res['L_plano']:.2f}{'':<4} | {res['w0']:.4f}   | {res['wm']:.4f}   | {type_str}")
    
    print("=" * len(header))
    return results

def visualize_spot_distribution(spots_x, spots_y, f, d, x0, y0, N_visible):
    """
    绘制镜面光斑分布图
    """
    print(f"\n--- 正在生成光斑分布图 (d={d:.2f} mm) ---")
    
    m1_x = spots_x[::2]
    m1_y = spots_y[::2]
    
    m2_x = spots_x[1::2]
    m2_y = spots_y[1::2]
    
    plt.figure(figsize=(8, 8))
    
    A_theoretical = np.sqrt(spots_x[0]**2 + spots_y[0]**2)
    
    # 绘制光斑
    plt.scatter(m1_x, m1_y, c='red', s=80, zorder=3, edgecolors='black', label='M1 (凹面镜)')
    plt.scatter(m2_x, m2_y, c='none', edgecolors='blue', s=60, zorder=2, alpha=0.5, label='M2 (平面镜/M2)')
    
    circle = Circle((0, 0), A_theoretical, fill=False, color='gray', linestyle='--', alpha=0.5, label='理论轨迹')
    plt.gca().add_patch(circle)
    
    plt.scatter([x0], [y0], c='green', s=150, marker='*', label='入射位置', zorder=4)
    
    # 计算 theta
    theta_rad = np.arccos(np.clip(1 - d / (2 * f), -1, 1))
    theta_deg = np.degrees(theta_rad)
    mirror_angle = 2 * theta_deg 
    
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlabel('X 坐标 (mm)')
    plt.ylabel('Y 坐标 (mm)')
    
    title_str = (f'Herriott 池光斑分布 (目标 wm≈0.5mm)\n'
                 f'可见单镜光斑数: {len(np.unique(np.round(m1_x, 4)))}\n'
                 f'(f={f}mm, d={d:.2f}mm)')
    plt.title(title_str)
    
    plt.axis('equal')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

def plot_stability_diagram_for_selection(f, d, results, max_mu_limit=None):
    """绘制稳定性图并标记搜索到的点"""
    plt.figure(figsize=(10, 6))
    
    # 绘制所有搜索到的解
    d_vals = [r['d']/f for r in results]
    v_vals = [r['v'] for r in results]
    
    plt.scatter(d_vals, v_vals, c='blue', alpha=0.6, label='满足光斑条件的解')
    
    # 标记当前选中的解
    current_df = d/f
    plt.axvline(x=current_df, color='red', linestyle='--', alpha=0.8, label=f'当前选择 d/f={current_df:.2f}')
    
    title_suffix = f"(μ <= {max_mu_limit})" if max_mu_limit else "(无 μ 限制)"
    plt.title(f"符合目标光斑大小的稳定性分布 {title_suffix}")
    plt.xlabel("镜间距与焦距之比 (d/f)")
    plt.ylabel("单面镜光斑数 (ν)")
    plt.xlim(0, 4)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

def analyze_beam_radius(R, lambda_nm, current_d, config):
    """计算并绘制光斑半径随 C 值变化"""
    print("\n--- 正在分析光斑半径与模体积 ---")
    lambda_mm = lambda_nm * 1e-6
    pi = np.pi
    current_C = current_d / R
    
    # 计算曲线
    C_axis = np.linspace(0.01, 1.99, 1000)
    term_w0 = C_axis * (2 - C_axis)
    w0_curve = np.sqrt((R * lambda_mm) / (2 * pi) * np.sqrt(term_w0))
    
    term_wm = C_axis / (2 - C_axis)
    wm_curve = np.sqrt((R * lambda_mm) / pi * np.sqrt(term_wm))

    plt.figure(figsize=(10, 6))
    plt.plot(C_axis, w0_curve, label='$w_0$ (腰斑)', linewidth=2)
    plt.plot(C_axis, wm_curve, label='$w_m$ (镜面光斑)', linewidth=2, linestyle='--')
    
    # 标记当前点
    plt.scatter([current_C], [config['w0']], color='red', s=100, zorder=5)
    plt.scatter([current_C], [config['wm']], color='red', s=100, zorder=5, label='当前设计点')
    
    # 添加辅助线，显示 wm=0.5 对应的位置
    plt.axhline(y=0.5, color='green', linestyle=':', alpha=0.7, label='目标 wm=0.5mm')

    plt.xlabel('C (d/R)', fontsize=12)
    plt.ylabel('光斑半径 w (mm)', fontsize=12)
    plt.title(f'光斑半径随腔长变化 (R={R}mm)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    plt.xlim(0, 2)
    # 限制Y轴范围以便看清0.5mm附近
    plt.ylim(0, 1.0) 
    plt.show()

# --- 4. 主程序 ---

def main():
    # ================= 参数设置区 =================
    f_val = 500         # 焦距 (mm)
    lambda_val = 1064   # 波长 (nm)
    
    # 目标参数
    target_wm = 0.5     # 目标镜面光斑半径 (mm)
    tolerance = 0.15    # 容差范围
    
    # [关键设置 1] 单镜光斑数 ν 的范围
    # 控制这个范围，避免计算出 ν=100 这种不切实际的解
    min_nu_val = 2
    max_nu_val = 80 
    
    # [关键设置 2] 绕行圈数 μ 的上限
    # 建议设为 5~10，因为 μ 过大代表光路极其复杂且不稳定
    max_mu_val = 5 
    
    # 初始几何参数
    r_circle = 20 
    x0 = r_circle * np.sqrt(2)/2; y0 = x0
    # ============================================

    # 1. 按光斑大小搜索配置
    valid_configs = find_configs_by_spot_size(
        f_val, lambda_val, target_wm, tolerance, 
        min_nu=min_nu_val, max_nu=max_nu_val,   # 传入 ν 范围
        max_mu_limit=max_mu_val                 # 传入 μ 上限
    )
    
    if not valid_configs:
        print(f"在 ν={min_nu_val}~{max_nu_val}, μ<={max_mu_val} 范围内未找到 wm≈{target_wm}mm 的解。")
        print("建议：增大 max_mu_val，或者放宽 tolerance 容差。")
        return

    # 2. 自动选择最接近的配置 (列表已排序，取第一个)
    selected_config = valid_configs[0]
    d_val = selected_config['d']
    N_val = selected_config['N']
    
    print("\n" + "#" * 50)
    print(f"【最接近目标 wm={target_wm}mm 的配置 (且 ν≤{max_nu_val}, μ≤{max_mu_val})】")
    print(f"  单镜光斑数 ν     : {selected_config['v']}")
    print(f"  模式 (ν, μ)      : ({selected_config['v']}, {selected_config['mu']})")
    
    # 构造类型字符串 (解决 f-string 嵌套问题)
    if selected_config['is_coprime']:
        type_info = "标准互质"
    else:
        type_info = f"重入 (GCD={selected_config['gcd']})"
    print(f"  类型             : {type_info}")
    
    print(f"  等效双凹腔长 d   : {d_val:.4f} mm")
    print(f"  平凹腔长 L       : {selected_config['L_plano']:.4f} mm")
    print(f"  凹面镜光斑 wm    : {selected_config['wm']:.5f} mm")
    print(f"  腰斑 w0          : {selected_config['w0']:.5f} mm")
    print("#" * 50)

    # 3. 绘制稳定性分布
    plot_stability_diagram_for_selection(f_val, d_val, valid_configs, max_mu_limit=max_mu_val)

    # 4. 绘制光斑分布
    try:
        incident_angles = calculate_incident_angles(x0, y0, f_val, d_val)
        x_prime0, y_prime0 = incident_angles[1]
        spots_x, spots_y = calculate_spot_positions(x0, x_prime0, f_val, d_val, n_spots=N_val)
        visualize_spot_distribution(spots_x, spots_y, f_val, d_val, x0, y0, N_val)
    except ValueError as e:
        print(f"几何计算出错: {e}")

    # 5. 分析光斑半径曲线
    analyze_beam_radius(2*f_val, lambda_val, d_val, selected_config)
    
    # 6. 物理评估
    pulse_energy = 200e-6 
    pulse_width = 10e-12  
    evaluate_physics(pulse_energy, pulse_width, selected_config['w0'], selected_config['wm'])

if __name__ == "__main__":
    main()