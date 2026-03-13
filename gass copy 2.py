import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

# ==========================================
# 1. 参数设置与数据录入 (请修改这里)
# ==========================================

# 激光波长 (单位: nm)
wavelength_nm = 1064.0  
lambda_mm = wavelength_nm * 1e-6

# 数据录入格式：
# { Z轴位置(mm): [(X直径1, Y直径1), (X直径2, Y直径2), ... (X直径6, Y直径6)] }
# 注意：请输入“直径”，代码会自动转为“半径”进行计算
# 注意：单位默认为 um (微米)，如果是mm请在后面代码调整 scale_factor

raw_data = {
    # 示例数据 (请替换为你真实记录的数据)
    # 格式: 位置: [(x1, y1), (x2, y2), (x3, y3), (x4, y4), (x5, y5), (x6, y6)]
    20: [(2300, 2646), (2320, 2753), (2326, 2719), (2273, 2846), (2260, 2753), (2266, 2859)],
    25: [(2273, 2640), (2273, 2660), (2273, 2759), (2300, 2773), (2240, 2660), (2246, 2660)],
    30: [(1366, 1602), (1396, 1615), (1353, 1549), (1353, 1596), (1346, 1609), (1353, 1609)],
    35: [(1393, 1662), (1386, 1655), (1373, 1536), (1386, 1582), (1380, 1596), (1393, 1649)],
    40: [(460, 512), (453, 498), (446, 505), (446, 498), (453, 505), (453, 498)], # 假设束腰附近
    45: [(406, 452), (406, 452), (406, 458), (400, 452), (406, 452), (406, 452)],
    60: [(1560, 1915), (1580, 1855), (1573, 1935), (1560, 1921), (1546, 1895), (1533, 1968)],
    65: [(1660, 2008), (1586, 1901), (1706, 1928), (1646, 1975), (1666, 1975), (1620, 1968)],
    70: [(2673, 3145), (2580, 3185), (2193, 3231), (2200, 3211), (2673, 3165), (2573, 3165)],
    75: [(2406, 3258), (2606, 3211), (2580, 3118), (2506, 3245), (2613, 3238), (2666, 3185)],
}

# 数据单位转换因子 (如果你的数据是微米 um，设为 1e-3 转为 mm)
scale_factor = 1e-3 

# ==========================================
# 2. 数据处理与清洗函数
# ==========================================

# ==========================================
# 2. 数据处理与清洗函数 (修正版)
# ==========================================

def clean_and_process(data_dict):
    z_list = []
    
    x_mean_list = []
    x_err_list = []
    
    y_mean_list = []
    y_err_list = []
    
    print(f"{'Pos(mm)':<10} | {'X_raw':<10} | {'X_clean':<10} | {'Drop X':<5} || {'Y_raw':<10} | {'Y_clean':<10} | {'Drop Y':<5}")
    print("-" * 85)

    sorted_z = sorted(data_dict.keys())
    
    # 定义一个最小误差（防止除以零），设为 0.1 微米 (即 0.0001 mm)
    min_err = 0.0001 
    
    for z in sorted_z:
        measurements = np.array(data_dict[z])
        # 转换为半径 (Diameter / 2) 并转换单位 (mm)
        radii = (measurements / 2.0) * scale_factor
        
        x_radii = radii[:, 0]
        y_radii = radii[:, 1]
        
        # 清洗函数：剔除超过 2倍标准差的点
        def filter_outliers(arr):
            mean = np.mean(arr)
            std = np.std(arr)
            if std == 0: return arr, 0
            filtered = arr[np.abs(arr - mean) < 2.0 * std]
            dropped_count = len(arr) - len(filtered)
            # 如果剔除后没有数据了，就恢复原状
            if len(filtered) == 0: 
                return arr, 0
            return filtered, dropped_count

        x_clean, x_drop = filter_outliers(x_radii)
        y_clean, y_drop = filter_outliers(y_radii)
        
        z_list.append(z)
        
        # 计算 X 的均值和方差
        x_mean = np.mean(x_clean)
        x_std = np.std(x_clean)
        # 如果方差为0，强制给一个极小值，防止崩溃
        if x_std == 0: x_std = min_err
        x_mean_list.append(x_mean)
        x_err_list.append(x_std)
        
        # 计算 Y 的均值和方差
        y_mean = np.mean(y_clean)
        y_std = np.std(y_clean)
        # 如果方差为0，强制给一个极小值
        if y_std == 0: y_std = min_err
        y_mean_list.append(y_mean)
        y_err_list.append(y_std)
        
        print(f"{z:<10} | {len(x_radii):<10} | {len(x_clean):<10} | {x_drop:<5} || {len(y_radii):<10} | {len(y_clean):<10} | {y_drop:<5}")

    return np.array(z_list), np.array(x_mean_list), np.array(x_err_list), np.array(y_mean_list), np.array(y_err_list)

# ==========================================
# 3. 拟合模型与计算
# ==========================================

# 高斯光束传播方程: w(z) = w0 * sqrt(1 + ((z-z0)/zR)^2)
def gaussian_beam_model(z, w0, z0, zR):
    return w0 * np.sqrt(1 + ((z - z0) / zR)**2)

def calculate_m2(w0_mm, zR_mm, wavelength_mm):
    # M2 = (pi * w0^2) / (lambda * zR)
    return (np.pi * w0_mm**2) / (wavelength_mm * zR_mm)

# ==========================================
# 4. 执行主程序
# ==========================================

# 1. 处理数据
z_vals, x_means, x_errs, y_means, y_errs = clean_and_process(raw_data)

# 2. 初始猜测参数 [w0, z0, zR]
# 猜测：w0为最小测量值，z0为最小测量值对应的位置，zR猜一个典型值(e.g. 10mm)
min_idx_x = np.argmin(x_means)
p0_x = [x_means[min_idx_x], z_vals[min_idx_x], 10.0]

min_idx_y = np.argmin(y_means)
p0_y = [y_means[min_idx_y], z_vals[min_idx_y], 10.0]

# 3. 曲线拟合
try:
    popt_x, pcov_x = curve_fit(gaussian_beam_model, z_vals, x_means, p0=p0_x, sigma=x_errs, absolute_sigma=True, bounds=([0, -np.inf, 0], [np.inf, np.inf, np.inf]))
    popt_y, pcov_y = curve_fit(gaussian_beam_model, z_vals, y_means, p0=p0_y, sigma=y_errs, absolute_sigma=True, bounds=([0, -np.inf, 0], [np.inf, np.inf, np.inf]))
except RuntimeError:
    print("拟合失败！请检查数据是否呈现抛物线趋势。")
    exit()

# 4. 计算 M2
w0_x, z0_x, zR_x = popt_x
w0_y, z0_y, zR_y = popt_y

m2_x = calculate_m2(w0_x, zR_x, lambda_mm)
m2_y = calculate_m2(w0_y, zR_y, lambda_mm)

# ==========================================
# 5. 结果输出与绘图
# ==========================================

print("\n" + "="*30)
print("   高斯光束参数计算结果")
print("="*30)
print(f"X 方向 (水平):")
print(f"  束腰半径 w0 : {w0_x*1000:.2f} um")
print(f"  束腰位置 z0 : {z0_x:.2f} mm (相对于起始点)")
print(f"  瑞利长度 zR : {zR_x:.2f} mm")
print(f"  光束质量 M2 : {m2_x:.3f}")
print("-" * 30)
print(f"Y 方向 (垂直):")
print(f"  束腰半径 w0 : {w0_y*1000:.2f} um")
print(f"  束腰位置 z0 : {z0_y:.2f} mm (相对于起始点)")
print(f"  瑞利长度 zR : {zR_y:.2f} mm")
print(f"  光束质量 M2 : {m2_y:.3f}")
print("="*30)

# 绘图
plt.figure(figsize=(10, 6))

# 生成平滑曲线用于绘图
z_smooth = np.linspace(min(z_vals)-5, max(z_vals)+5, 200)
x_fit = gaussian_beam_model(z_smooth, *popt_x)
y_fit = gaussian_beam_model(z_smooth, *popt_y)

# 绘制 X 数据
plt.errorbar(z_vals, x_means*1000, yerr=x_errs*1000, fmt='ro', label='X Data (Mean)', capsize=3)
plt.plot(z_smooth, x_fit*1000, 'r--', label=f'X Fit ($M^2$={m2_x:.2f})')

# 绘制 Y 数据
plt.errorbar(z_vals, y_means*1000, yerr=y_errs*1000, fmt='bo', label='Y Data (Mean)', capsize=3)
plt.plot(z_smooth, y_fit*1000, 'b--', label=f'Y Fit ($M^2$={m2_y:.2f})')

plt.title(f'Gaussian Beam Caustic Measurement\n($\lambda$={wavelength_nm}nm, f=400mm setup)')
plt.xlabel('Position Z (mm)')
plt.ylabel('Beam Radius w (um)')
plt.legend()
plt.grid(True, which='both', linestyle='--', alpha=0.7)

# 显示
plt.show()