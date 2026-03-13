import numpy as np
import pandas as pd

# ==========================================
# --- 1. 用户参数设置 (请根据实验实际修改) ---
# ==========================================
# 几何参数
d_waist = 0.43      # 束腰直径 (mm)
d_measured = 0.79   # 传播后的光斑直径 (mm)
z_measured = 215.5  # 传播距离 (mm)

# 激光脉冲参数 (关键：自聚焦取决于峰值功率)
lambda_nm = 800     # 波长 (nm)
pulse_energy_uj = 200 # 单脉冲能量 (uJ) - 【请修改此值】
pulse_width_fs = 10000 # 脉冲宽度 (fs) - 【请修改此值】

# 介质参数 (熔融石英 Fused Silica)
n0 = 1.45           # 线性折射率
n2_cm2_w = 2.5e-16  # 非线性折射率 (cm^2/W)
L_medium_mm = 12.0  # 介质厚度 (从数据差值 42-30 推断)

# 目标位置 (保持原样)
z_positions = [30, 42, 67, 79, 104, 116, 141, 153]

# ==========================================
# --- 2. 物理常数计算 ---
# ==========================================
# 基础单位转换
lambda_cm = lambda_nm * 1e-7
P_peak = (pulse_energy_uj * 1e-6) / (pulse_width_fs * 1e-15) # 峰值功率 (W)

# 计算临界功率 P_cr (Marburger公式)
# P_cr = (3.77 * lambda^2) / (8 * pi * n0 * n2)
P_cr = (3.77 * lambda_cm**2) / (8 * np.pi * n0 * n2_cm2_w)

print(f"--- 激光物理参数 ---")
print(f"峰值功率 P_peak: {P_peak/1e6:.2f} MW")
print(f"临界功率 P_cr:   {P_cr/1e6:.2f} MW")
if P_peak > P_cr:
    print("警告：峰值功率超过临界功率，可能导致灾难性塌缩或成丝！")
else:
    print(f"功率比 P/Pcr:    {P_peak/P_cr:.3f}")
print(f"-------------------\n")


# ==========================================
# --- 3. 瑞利长度反推模型 ---
# ==========================================
try:
    term = (d_measured / d_waist)**2 - 1
    if term <= 0:
        raise ValueError("测量点光斑必须明显大于束腰光斑")
    
    ZR_new = z_measured / np.sqrt(term)
    print(f"--- 几何模型构建成功 ---")
    print(f"反推瑞利长度 ZR: {ZR_new:.2f} mm")
    print(f"----------------------\n")

except Exception as e:
    print(f"构建模型失败: {e}")
    ZR_new = None

# ==========================================
# --- 4. 自聚焦效应计算与列表输出 ---
# ==========================================
if ZR_new:
    def calculate_beam_diameter(z, d0, ZR):
        return d0 * np.sqrt(1 + (z / ZR)**2)

    def calculate_kerr_focal_length(w_mm, P, n2, L_mm):
        """
        计算薄片介质的克尔透镜焦距 f_nl
        公式: f = (pi * w^4) / (8 * n2 * P * L)  (假设高斯光束)
        注意单位换算: w(mm)->cm, L(mm)->cm
        """
        w_cm = w_mm / 10.0
        L_cm = L_mm / 10.0
        # 防止除零
        if P == 0 or L_cm == 0: return np.inf
        
        f_cm = (np.pi * w_cm**4) / (8 * n2 * P * L_cm)
        return f_cm * 10.0 # 返回 mm

    def calculate_collapse_distance(z_R_mm, P, P_cr):
        """
        计算整体塌缩距离 z_sf (如果介质无限长)
        公式: z_sf = 0.367 * z_R / sqrt(sqrt(P/Pcr) - 0.852)
        这里用简化版: z_sf = z_R / sqrt(P/Pcr - 1)
        """
        if P <= P_cr:
            return np.inf # 不会塌缩
        return z_R_mm / np.sqrt(P/P_cr - 1)

    results = []
    
    # 遍历每一对位置 (代表一块晶体的入社和出射)
    for i in range(0, len(z_positions), 2):
        z_in = z_positions[i]
        z_out = z_positions[i+1]
        
        # 1. 几何光斑计算
        d_in = calculate_beam_diameter(z_in, d_waist, ZR_new)
        d_out = calculate_beam_diameter(z_out, d_waist, ZR_new)
        w_avg_mm = ((d_in + d_out) / 2) / 2 # 平均半径
        
        # 2. 三种自聚焦相关参数计算
        
        # (A) 克尔透镜焦距 f_Kerr (当前12mm晶体产生的透镜效应)
        # 这是最有用的值：告诉你这块晶体相当于一个多强凸透镜
        f_kerr = calculate_kerr_focal_length(w_avg_mm, P_peak, n2_cm2_w, L_medium_mm)
        
        # (B) 理论塌缩距离 z_collapse (假设介质无限长)
        # 基于当前的束腰参数和功率
        z_collapse = calculate_collapse_distance(ZR_new, P_peak, P_cr)
        


        results.append({
            "晶体位置 (mm)": f"{z_in}-{z_out}",
            "平均光斑半径 w (mm)": round(w_avg_mm, 3),
            "克尔透镜焦距 f_nl (mm)": round(f_kerr, 1),
            "塌缩距离 z_sf (mm)": "∞" if z_collapse == np.inf else round(z_collapse, 1),
        })

    # 输出表格
    df = pd.DataFrame(results)
    
    # 格式化输出，方便阅读
    print("--- 仿真结果：每块介质处的自聚焦效应 ---")
    print(df.to_string(index=False))
    
    print("\n说明:")
    print("1. 克尔透镜焦距 f_nl: 当前12mm厚石英片产生的等效正透镜焦距。数值越小，聚焦效应越强。")
    print("2. 塌缩距离 z_sf: 如果该处石英无限长，光束将在多少毫米后聚焦为一个点。")