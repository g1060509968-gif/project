import numpy as np
import matplotlib.pyplot as plt

# 1. 生成数据 (高斯函数)
# x: 波长范围 (例如 450nm 到 750nm)
x = np.linspace(1000,1060, 1000)

# 高斯参数: f(x) = a * exp(-(x-b)^2 / (2c^2))
center_wavelength = 1030  # 中心波长 600 nm
sigma = 5               # 宽度控制 (标准差)

# 计算归一化强度 (峰值为1)
y = np.exp(-0.5 * ((x - center_wavelength) / sigma) ** 2)

# 2. 设置科研绘图风格 (Publication Quality Style)
params = {
    'font.size': 14,
    'font.family': 'sans-serif',      # 使用无衬线字体 (如 Arial/Helvetica)，Nature/Science常用
    'axes.linewidth': 1.5,            # 边框加粗
    'xtick.major.width': 1.5,         # 刻度线加粗
    'ytick.major.width': 1.5,
    'xtick.direction': 'in',          # 刻度朝内
    'ytick.direction': 'in',
    'xtick.top': True,                # 顶部显示刻度
    'ytick.right': True,              # 右侧显示刻度
    'lines.linewidth': 2.5,           # 曲线加粗
    'legend.frameon': False,          # 图例去边框，更简洁
}
plt.rcParams.update(params)

# 创建画布
fig, ax = plt.subplots(figsize=(8, 6), dpi=300) # dpi=300 保证高分辨率

# 绘制曲线
# 颜色推荐：深蓝(#004C99), 深红(#990000), 或黑灰
ax.plot(x, y, color='#004C99', label='Gaussian Fit') 

# 设置标签
#通常SCI论文使用英文标签，如果需要中文可替换为 '波长 (nm)'
ax.set_xlabel('Wavelength (nm)', fontweight='bold')
ax.set_ylabel('Normalized Intensity (a.u.)', fontweight='bold')

# 设置坐标轴范围
ax.set_xlim(1000, 1060)
ax.set_ylim(0, 1.1)

# 添加注释 (例如显示峰值位置)
ax.text(0.05, 0.95, r'$\lambda_{peak} = 1030$ nm', transform=ax.transAxes, 
        verticalalignment='top', fontsize=14)

# 添加图例
ax.legend(loc='upper right')

# 紧凑布局并显示
plt.tight_layout()
plt.show()
# 如需保存，使用 plt.savefig('spectrum.png', dpi=300)