import numpy as np
import matplotlib.pyplot as plt

# --- 1. 参数设置 ---
# 时间轴范围 (fs)
t_min, t_max = -30, 30
# 数据点数量 (足够高以保证曲线平滑)
num_points = 2000
t = np.linspace(t_min, t_max, num_points)

# 物理参数估计
A = 1.0           # 振幅 (归一化为1)
sigma = 7.5       # 高斯包络宽度标准差 (决定了脉冲持续时间)
omega = 4.1       # 载波角频率 (rad/fs), 对应约 0.65 PHz 的频率
phase = 0         # 相位 (中心为余弦峰值)

# --- 2. 数据生成 ---
# 高斯包络 (Envelope)
envelope = A * np.exp(-(t**2) / (2 * sigma**2))

# 载波 (Carrier wave)
carrier = np.cos(omega * t + phase)

# 总电场 (Electric field)
electric_field = envelope * carrier

# --- 3. 绘图 ---
fig, ax = plt.subplots(figsize=(8, 5)) # 设置画布大小比例接近原图

# 绘制实线：电场振荡
# 使用一种稍浅的蓝色以接近原图风格
line_color = '#5b9bd5' 
ax.plot(t, electric_field, color=line_color, linewidth=1.5, label='Electric field')

# 绘制虚线：上下包络
# 使用相同的颜色，但设为点状线 (linestyle=':')，且稍微透明一点
ax.plot(t, envelope, color=line_color, linestyle=':', linewidth=1.5, alpha=0.8)
ax.plot(t, -envelope, color=line_color, linestyle=':', linewidth=1.5, alpha=0.8)

# --- 4. 样式调整 (复刻原图风格) ---
# 设置坐标轴标签和字体大小
ax.set_xlabel('time (fs)', fontsize=14)
ax.set_ylabel('electric field (a. u.)', fontsize=14)

# 设置坐标轴刻度字体大小
ax.tick_params(axis='both', which='major', labelsize=12)

# 设置坐标轴范围
ax.set_xlim(t_min, t_max)
ax.set_ylim(-1.1, 1.1)

# 设置特定的Y轴刻度点
ax.set_yticks([-1.0, -0.5, 0, 0.5, 1.0])

# 添加网格线 (虚线风格)
ax.grid(True, linestyle='--', color='gray', alpha=0.5)

# 增加边框厚度，使图看起来更专业
for spine in ax.spines.values():
    spine.set_linewidth(1.2)

# 紧凑布局
plt.tight_layout()

# 显示图像
plt.show()