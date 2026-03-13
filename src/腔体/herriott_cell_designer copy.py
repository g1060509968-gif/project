import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import decimal
import math

# --- 1. 全局绘图设置 ---
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

# --- 2. Herriott Cell 几何计算函数 ---

def evaluate_physics(pulse_energy_J, pulse_width_s, w0_mm, wm_mm, n2=3e-20):
    print("\n--- 物理参数评估 ---")
    w0_cm = w0_mm / 10.0
    wm_cm = wm_mm / 10.0
    
    # 峰值功率 (假设高斯脉冲)
    P_peak = 0.94 * pulse_energy_J / pulse_width_s
    print(f"峰值功率 P_peak: {P_peak/1e6:.2f} MW")
    
    # 镜面处的能量密度 (Fluence)
    fluence_mirror = 2 * pulse_energy_J / (np.pi * wm_cm**2) # J/cm^2
    print(f"镜面能量密度: {fluence_mirror:.4f} J/cm^2 (通常应 < 2~5 J/cm^2)")
    
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
    # 这里的 theta 是单次传输的相移
    cos_theta = 1 - d / (2 * f)
    # 防止数值误差导致 arccos 越界
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

def find_valid_df_for_target_spots(target_total_spots, f, lambda_nm):
    """
    计算 d/f 值。
    [修改版]：移除了 math.gcd(mu, v) == 1 的强制限制，允许非互质解。
    """
    R = 2 * f
    lambda_mm = lambda_nm * 1e-6
    
    if target_total_spots % 2 != 0:
        print(f"⚠️ 警告: 输入的总光斑数 {target_total_spots} 是奇数，按 v={target_total_spots} 处理。")
        v = target_total_spots
    else:
        v = target_total_spots // 2
        
    print(f"\n====== 正在寻找总光斑数 N={target_total_spots} (单镜模式 v={v}) 的解 ======")
    print(f"提示：表中 'd' 为等效双凹腔长，'L_plano' 为对应的平凹腔长 (L = d/2)")
    
    results = []
    
    # 调整了表头，增加了 L_plano 和 类型 说明
    header = f"{'Idx':<4} | {'模式(ν,μ)':<10} | {'类型':<6} | {'d (双凹)':<10} | {'L (平凹)':<10} | {'w0 (mm)':<8} | {'wm (mm)':<8}"
    print("-" * len(header))
    print(header)
    print("-" * len(header))

    # 遍历 mu (绕行圈数)
    # 即使非互质 (gcd != 1) 也允许通过，以便找到论文中的特殊解
    for mu in range(1, v):
        
        # 检查互质性
        common_divisor = math.gcd(mu, v)
        is_coprime = (common_divisor == 1)
        type_str = "标准" if is_coprime else f"重入(/{common_divisor})"
        
        # 1. 几何计算
        theta = mu * np.pi / v
        d_f_ratio = 2 * (1 - np.cos(theta))
        d_val = d_f_ratio * f
        L_plano = d_val / 2.0  # 计算平凹腔长
        
        # 2. 物理计算 (光斑半径)
        C = d_val / R
        
        if C <= 0 or C >= 2:
            w0 = 0
            wm = 0
        else:
            # w0 (腰斑/最小)
            w0_sq = (R * lambda_mm) / (2 * np.pi) * np.sqrt(C * (2 - C))
            w0 = np.sqrt(w0_sq)
            # wm (镜面/最大)
            wm_sq = (R * lambda_mm) / np.pi * np.sqrt(C / (2 - C))
            wm = np.sqrt(wm_sq)
        
        results.append({
            'mu': mu,
            'v': v,
            'd_f': d_f_ratio,
            'd': d_val,
            'L_plano': L_plano,
            'w0': w0,
            'wm': wm,
            'is_coprime': is_coprime,
            'gcd': common_divisor
        })
        
        # 打印时高亮可能的论文解 (L 接近 212.5)
        marker = " <--" if abs(L_plano - 212.5) < 2 else ""
        print(f"{len(results)-1:<4} | ({v}, {mu}):<10 | {type_str:<6} | {d_val:.2f}{'':<4} | {L_plano:.2f}{'':<4} | {w0:.4f}   | {wm:.4f}{marker}")
            
    print("=" * len(header))
    return results

def visualize_spot_distribution(spots_x, spots_y, f, d, x0, y0):
    """
    绘制镜面光斑分布图
    """
    print(f"\n--- 正在生成光斑分布图 (d={d:.2f} mm) ---")
    
    m1_x = spots_x[::2]
    m1_y = spots_y[::2]
    m1_indices = np.arange(0, len(spots_x), 2)
    
    m2_x = spots_x[1::2]
    m2_y = spots_y[1::2]
    
    plt.figure(figsize=(8, 8))
    
    A_theoretical = np.sqrt(spots_x[0]**2 + spots_y[0]**2)
    
    plt.scatter(m1_x, m1_y, c='red', s=80, zorder=3, edgecolors='black', label='M1 (偶数次反射)')
    plt.scatter(m2_x, m2_y, c='none', edgecolors='blue', s=60, zorder=2, alpha=0.5, label='M2 (奇数次反射)')
    
    circle = Circle((0, 0), A_theoretical, fill=False, color='gray', linestyle='--', alpha=0.5, label='理论轨迹')
    plt.gca().add_patch(circle)
    
    plt.scatter([x0], [y0], c='green', s=150, marker='*', label='入射孔 (n=0)', zorder=4)
    
    # 标注前几个点，避免太乱
    for i, idx in enumerate(m1_indices):
        if idx == 0: continue
        # 只标前10个点和最后几个点，或者全部标
        if i < 5 or i > len(m1_indices)-3: 
            plt.annotate(f"{idx}", (m1_x[i], m1_y[i]), xytext=(5, 5), textcoords='offset points', fontsize=10, color='darkred')

    theta_rad = np.arccos(np.clip(1 - d / (2 * f), -1, 1))
    theta_deg = np.degrees(theta_rad)
    mirror_angle = 2 * theta_deg 
    
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.xlabel('X 坐标 (mm)')
    plt.ylabel('Y 坐标 (mm)')
    
    title_str = (f'Herriott 池光斑分布 (仿真)\n'
                 f'单镜光斑 v={len(m1_x)}, 2θ={mirror_angle:.1f}°\n'
                 f'(f={f}mm, 等效d={d:.2f}mm)')
    plt.title(title_str)
    
    plt.axis('equal')
    plt.legend(loc='upper right')
    plt.tight_layout()
    plt.show()

def plot_stability_diagram_with_current(f, d, v_max=50):
    """绘制稳定性图并标记当前点"""
    current_d_f = d / f
    plt.figure(figsize=(10, 6))
    
    x_vals, y_vals = [], []
    # 绘制背景点（包含非互质，用浅色表示）
    for v_iter in range(1, v_max + 1):
        for mu_iter in range(1, v_iter):
             # 不再限制 gcd，展示所有可能解
            val = 2 * (1 - np.cos(mu_iter * np.pi / v_iter))
            x_vals.append(val)
            y_vals.append(v_iter)
    
    plt.scatter(x_vals, y_vals, marker='.', s=10, alpha=0.2, color='gray', label='所有理论解 (含非互质)')
    
    plt.axvline(x=current_d_f, color='red', linestyle='--', alpha=0.8, linewidth=2, label=f'当前设计 d/f = {current_d_f:.4f}')
    
    # 可以在当前v值画一条横线
    # plt.axhline(y=33, color='blue', linestyle=':', alpha=0.5, label='v=33')

    plt.title("Herriott-Cell 稳定性图 (d/f vs 单镜光斑数 v)")
    plt.xlabel("镜间距与焦距之比 (d/f)")
    plt.ylabel("单面镜光斑数 (ν)")
    plt.xlim(0, 4)
    plt.ylim(0, v_max)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

def analyze_beam_radius(R, lambda_nm, current_d, mode_info):
    """计算并绘制光斑半径随 C 值变化"""
    print("\n--- 正在分析光斑半径与模体积 ---")
    lambda_mm = lambda_nm * 1e-6
    pi = np.pi
    current_C = current_d / R
    
    val_w0 = np.sqrt(current_C * (2 - current_C))
    current_w0 = np.sqrt((R * lambda_mm) / (2 * pi) * val_w0)
    
    val_wm = np.sqrt(current_C / (2 - current_C))
    current_wm = np.sqrt((R * lambda_mm) / pi * val_wm)
    
    # 计算平凹腔长
    L_plano = current_d / 2
    
    print(f"当前选定配置详细参数:")
    print(f"  模式 (ν, μ): ({mode_info['v']}, {mode_info['mu']})")
    print(f"  类型       : {'标准互质' if mode_info['is_coprime'] else '非互质/重入'}")
    print(f"  等效双凹 d : {current_d:.4f} mm")
    print(f"  平凹腔长 L : {L_plano:.4f} mm (论文对应值)")
    print(f"  聚焦光斑 w0: {current_w0:.4f} mm (论文值 ~0.215mm * 2 = 0.43)")
    print(f"  镜面光斑 wm: {current_wm:.4f} mm (论文值 ~0.395mm * 2 = 0.79)")

    C_axis = np.linspace(0.01, 1.99, 1000)
    term_w0 = C_axis * (2 - C_axis)
    w0_curve = np.sqrt((R * lambda_mm) / (2 * pi) * np.sqrt(term_w0))
    
    term_wm = C_axis / (2 - C_axis)
    wm_curve = np.sqrt((R * lambda_mm) / pi * np.sqrt(term_wm))

    plt.figure(figsize=(10, 6))
    plt.plot(C_axis, w0_curve, label='$w_0$ (聚焦光斑/平面镜)', linewidth=2)
    plt.plot(C_axis, wm_curve, label='$w_m$ (凹面镜光斑)', linewidth=2, linestyle='--')
    
    label_text = f'当前设计\nL={L_plano:.1f}mm\n(d={current_d:.1f})'
    plt.scatter([current_C], [current_w0], color='red', s=100, zorder=5, label=label_text)
    plt.scatter([current_C], [current_wm], color='red', s=100, zorder=5)
    
    plt.annotate(f'w0={current_w0:.4f}', xy=(current_C, current_w0), xytext=(current_C+0.1, current_w0),
                 arrowprops=dict(arrowstyle="->"))
    plt.annotate(f'wm={current_wm:.4f}', xy=(current_C, current_wm), xytext=(current_C+0.1, current_wm+0.05),
                 arrowprops=dict(arrowstyle="->"))

    plt.xlabel('C (d/R)', fontsize=12)
    plt.ylabel('光斑半径 w (mm)', fontsize=12)
    plt.title(f'光斑半径随腔长变化 (R={R}mm)', fontsize=14)
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend(fontsize=10)
    plt.xlim(0, 2)
    plt.ylim(0, max(current_wm, np.max(wm_curve[100:-100])) * 1.2)
    plt.show()

# --- 4. 主程序 ---

def main():
    # ================= 参数设置区 =================
    f_val = 150        # 焦距 (mm) -> R = 300 mm
    R_val = 2 * f_val     
    lambda_val = 1064     # 波长 (nm) - 论文使用的是 1064 nm
    
    # 想要的总反射次数 (Total Passes) = 2 * ν
    target_spots = 66
    r = 20
    x0 = r * np.sqrt(2)/2; y0 = x0      # 初始入射位置
    # ============================================

    # 1. 查找所有可能的腔长配置 (含非互质)
    valid_configs = find_valid_df_for_target_spots(target_spots, f_val, lambda_val)
    
    if not valid_configs:
        print("未找到满足该光斑数的有效腔长配置。")
        return

    # 2. 智能选择配置 (寻找最接近论文参数 L=212.5 -> d=425 的解)
    target_d = 425.0
    selected_config = None
    
    # 按照与 425mm 的差值排序
    valid_configs.sort(key=lambda x: abs(x['d'] - target_d))
    
    selected_config = valid_configs[0]
    
    # 检查是否找到了足够接近的解
    if abs(selected_config['d'] - target_d) > 10:
        print(f"⚠️ 警告: 未找到接近 d={target_d}mm (L=212.5mm) 的解。展示最接近的解。")
    else:
        print(f"\n✅ 成功匹配论文参数！")

    d_val = selected_config['d']
    
    print("\n" + "#" * 50)
    print(f"【最终选择并演示的配置】")
    print(f"  目标总反射次数 N : {target_spots}")
    print(f"  单镜光斑数 ν     : {selected_config['v']}")
    print(f"  绕行圈数 μ       : {selected_config['mu']} (GCD={selected_config['gcd']})")
    print(f"  等效双凹腔长 d   : {d_val:.4f} mm")
    print(f"  平凹腔长 L       : {selected_config['L_plano']:.4f} mm")
    print(f"  预估最小光斑 w0  : {selected_config['w0']:.4f} mm")
    print(f"  预估最大光斑 wm  : {selected_config['wm']:.4f} mm")
    print("#" * 50)

    # 3. 绘制稳定性图
    plot_stability_diagram_with_current(f_val, d_val)

    # 4. 绘制光斑分布
    try:
        incident_angles = calculate_incident_angles(x0, y0, f_val, d_val)
        x_prime0, y_prime0 = incident_angles[1]
        
        # 计算所有光斑位置
        spots_x, spots_y = calculate_spot_positions(x0, x_prime0, f_val, d_val, n_spots=target_spots)
        
        visualize_spot_distribution(spots_x, spots_y, f_val, d_val, x0, y0)
    except ValueError as e:
        print(f"几何计算出错: {e}")
        return

    # 5. 分析光斑半径
    analyze_beam_radius(R_val, lambda_val, d_val, selected_config)
    
    # 额外：物理损伤阈值评估
    pulse_energy = 100 / 500e3 # 100W, 500kHz -> 200uJ
    pulse_width = 10e-12       # 10ps
    evaluate_physics(pulse_energy, pulse_width, selected_config['w0'], selected_config['wm'])

if __name__ == "__main__":
    main()