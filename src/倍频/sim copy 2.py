import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift, fftfreq

# 导入我们之前编写的两个硬核模块
from crystal_database import CrystalBBO
from solver import ultrafast_twm_solver

# ==========================================
# 1. 全局仿真环境设置
# ==========================================
c = 3e8
# 时间网格: 4096个点，覆盖 4000 飞秒窗口
N = 4096
T_window = 4000e-15
t_grid = np.linspace(-T_window/2, T_window/2, N, endpoint=False)
dt = t_grid[1] - t_grid[0]

# 频率网格 (用于绘制光谱)
f_grid = fftshift(fftfreq(N, d=dt))
nu_THz = f_grid * 1e-12

# 实例化 BBO 介质数据库
bbo = CrystalBBO()

# ==========================================
# 2. 初始脉冲定义 (1030 nm, 250 fs - 匹配飞秒光纤激光实验)
# ==========================================
lam_1w = 1030e-9
w_1 = 2 * np.pi * c / lam_1w
tau_fwhm = 250e-15  # 脉宽改为 250 飞秒
tau_0 = tau_fwhm / 1.665

# 峰值电场调至 2.0e8 V/m (约 5 GW/cm^2 峰值光强，激发强非线性)
E_peak = 2.0e8
A_1w_initial = E_peak * np.exp(-0.5 * (t_grid / tau_0)**2)

# ==========================================
# 3. 级联非线性光学张量预设 (唯象参数)
# ==========================================
# 克尔张量 (SPM & XPM)
g0 = 2e-16  
gamma_matrix = np.array([
    [g0, 2*g0, 2*g0],
    [2*g0, g0, 2*g0],
    [2*g0, 2*g0, g0]
])

# ==========================================
# 阶段 1：二倍频 SHG (1w + 1w -> 2w)
# ==========================================
print(">> 开始计算第一级：二倍频 (1030nm -> 515nm) ...")
lam_2w = lam_1w / 2
w_2 = 2 * w_1

theta_shg = bbo.get_phase_matching_angle(lam_1w, lam_1w, match_type='I')
deff_shg = bbo.calc_deff(theta_shg, match_type='I')
n_funcs_shg = bbo.get_solver_n_funcs(theta_shg, 'I')
beta_0 = np.zeros((3, 3))

A1_depleted_1, _, A_2w = ultrafast_twm_solver(
    A1_in=A_1w_initial, A2_in=A_1w_initial, A3_in=np.zeros_like(t_grid),
    w=(w_1, w_1, w_2), n_funcs=n_funcs_shg,
    L=1.5e-3, dz=10e-6, d_eff=deff_shg,  # 晶体厚度 1.5 mm
    gamma_matrix=gamma_matrix, beta_matrix=beta_0, alpha=(0, 0, 0), t_grid=t_grid
)

# ==========================================
# 阶段 2：四倍频 FHG (2w + 2w -> 4w)
# ==========================================
print(">> 开始计算第二级：四倍频 (515nm -> 257.5nm) ...")
lam_4w = lam_2w / 2
w_4 = 4 * w_1

theta_fhg = bbo.get_phase_matching_angle(lam_2w, lam_2w, match_type='I')
deff_fhg = bbo.calc_deff(theta_fhg, match_type='I')
n_funcs_fhg = bbo.get_solver_n_funcs(theta_fhg, 'I')

# 257.5nm 轻微吸收与 TPA
beta_fhg = np.zeros((3, 3))
beta_fhg[2, 2] = 1e-11 

_, _, A_4w = ultrafast_twm_solver(
    A1_in=A_2w, A2_in=A_2w, A3_in=np.zeros_like(t_grid),
    w=(w_2, w_2, w_4), n_funcs=n_funcs_fhg,
    L=0.2e-3, dz=2e-6, d_eff=deff_fhg, # 晶体厚度降至 0.2 mm
    gamma_matrix=gamma_matrix, beta_matrix=beta_fhg, alpha=(0, 0, 50), t_grid=t_grid
)

# ==========================================
# 阶段 3：五倍频 5HG (1w + 4w -> 5w) - 极紫外生死决战
# ==========================================
print(">> 开始计算第三级：五倍频 (1030nm + 257.5nm -> 206nm) ...")
lam_5w = lam_1w / 5
w_5 = 5 * w_1

theta_5hg = bbo.get_phase_matching_angle(lam_1w, lam_4w, match_type='I')
deff_5hg = bbo.calc_deff(theta_5hg, match_type='I')
n_funcs_5hg = bbo.get_solver_n_funcs(theta_5hg, 'I')

# 极强双光子吸收与交叉吸收张量
beta_5hg = np.zeros((3, 3))
beta_5hg[1, 1] = 2e-11  # 4w Self-TPA
beta_5hg[2, 2] = 8e-11  # 5w 极强 Self-TPA
beta_5hg[0, 1] = 3e-11  # 1w+4w Cross-TPA
beta_5hg[1, 0] = 3e-11  # 4w+1w Cross-TPA

# [光学延迟线 (Delay Line)] 将 4w 的峰值强制平移到 t=0，与 1w 对齐
shift_idx = (N // 2) - np.argmax(np.abs(A_4w))
A_4w_aligned = np.roll(A_4w, shift_idx)

A1_final, A4_final, A_5w = ultrafast_twm_solver(
    A1_in=A1_depleted_1, A2_in=A_4w_aligned, A3_in=np.zeros_like(t_grid),
    w=(w_1, w_4, w_5), n_funcs=n_funcs_5hg,
    L=0.05e-3, dz=1e-6, d_eff=deff_5hg, # 极限超薄晶体: 50 um
    gamma_matrix=gamma_matrix, beta_matrix=beta_5hg, 
    alpha=(0, 50, 1500), # 模拟 206 nm 强烈的紫外吸收墙
    t_grid=t_grid
)

print(">> 仿真计算完成！正在生成分析图像...")

# ==========================================
# 4. 可视化分析：多维数据绘图 (5子图架构)
# ==========================================
def get_spectrum(A_t):
    """计算对数光强谱 (dB)"""
    return 10 * np.log10(np.abs(fftshift(fft(A_t)))**2 + 1e-10)

def get_wavelength_spectrum(A_t, lam_center_m):
    """提取物理域波长光谱 (nm)"""
    spectrum_dB = get_spectrum(A_t)
    nu_abs = f_grid + (c / lam_center_m)
    
    # 物理安全锁：截断非物理的低频和负频，防止波长倒数爆炸
    valid_idx = nu_abs > 10e12 
    nu_valid = nu_abs[valid_idx]
    spectrum_valid = spectrum_dB[valid_idx]
    
    lam_nm = (c / nu_valid) * 1e9
    return lam_nm, spectrum_valid

# 设置画布
plt.figure(figsize=(18, 10))
plt.rcParams['font.sans-serif'] = ['SimHei'] # 解决中文显示
plt.rcParams['axes.unicode_minus'] = False

# [子图 1] 第一级：SHG 时域波形
plt.subplot(2, 3, 1)
plt.plot(t_grid*1e15, np.abs(A_1w_initial)**2, 'k--', label='初始 1w (1030 nm)')
plt.plot(t_grid*1e15, np.abs(A1_depleted_1)**2, 'r-', label='消耗后 1w')
plt.plot(t_grid*1e15, np.abs(A_2w)**2, 'g-', label='生成 2w (515 nm)')
plt.xlim(-400, 400)
plt.xlabel('时间 (fs)')
plt.ylabel('光强 $|A|^2$')
plt.title('Stage 1: SHG (时域)')
plt.legend()
plt.grid(True, alpha=0.3)

# [子图 2] 最终级：5HG 时域波形与脉冲形变 (双 Y 轴)
ax1 = plt.subplot(2, 3, 2)
ax1.plot(t_grid*1e15, np.abs(A1_depleted_1)**2, 'r--', alpha=0.5, label='注入的 1w')
ax1.plot(t_grid*1e15, np.abs(A_4w_aligned)**2, 'b--', alpha=0.5, label='对齐后的 4w')
ax1.set_xlabel('时间 (fs)')
ax1.set_ylabel('1w/4w 光强 $|A|^2$', color='k')
ax1.set_xlim(-300, 300)

ax2 = ax1.twinx()
ax2.plot(t_grid*1e15, np.abs(A_5w)**2, 'm-', linewidth=2, label='最终 5w (206 nm)')
ax2.set_ylabel('5w 光强 $|A|^2$ (放大显示)', color='m')

plt.title('Stage 3: 5HG 深紫外脉冲生成 (时域)')
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
ax1.grid(True, alpha=0.3)

# [子图 3] 全光路光谱演化 (频率域 THz)
plt.subplot(2, 3, 3)
plt.plot(nu_THz + c/lam_1w*1e-12, get_spectrum(A1_depleted_1), 'r', label='1w')
plt.plot(nu_THz + c/lam_2w*1e-12, get_spectrum(A_2w), 'g', label='2w')
plt.plot(nu_THz + c/lam_4w*1e-12, get_spectrum(A_4w), 'b', label='4w')
plt.plot(nu_THz + c/lam_5w*1e-12, get_spectrum(A_5w), 'm', label='5w')
plt.xlim(200, 1600)
plt.ylim(60, 200)
plt.xlabel('绝对频率 (THz)')
plt.ylabel('谱密度 (dB)')
plt.title('各阶谐波光谱 (频率域)')
plt.legend()
plt.grid(True, alpha=0.3)

# [子图 4] 新增：全光路光谱演化 (波长域 nm)
plt.subplot(2, 3, 4)
lam_nm_1, spec_1 = get_wavelength_spectrum(A1_depleted_1, lam_1w)
lam_nm_2, spec_2 = get_wavelength_spectrum(A_2w, lam_2w)
lam_nm_4, spec_4 = get_wavelength_spectrum(A_4w, lam_4w)
lam_nm_5, spec_5 = get_wavelength_spectrum(A_5w, lam_5w)

plt.plot(lam_nm_1, spec_1, 'r', label='1w (1030 nm)')
plt.plot(lam_nm_2, spec_2, 'g', label='2w (515 nm)')
plt.plot(lam_nm_4, spec_4, 'b', label='4w (257.5 nm)')
plt.plot(lam_nm_5, spec_5, 'm', label='5w (206 nm)')

# 反转 X 轴 (让短波长/高能在左侧，符合光学实验仪器的习惯)
plt.xlim(1200, 180) 
plt.ylim(60, 200)
plt.xlabel('波长 (nm)')
plt.ylabel('谱密度 (dB)')
plt.title('各阶谐波光谱 (波长域)')
plt.legend()
plt.grid(True, alpha=0.3)

# [子图 5] 脉冲形变与色散分析 (归一化对比)
plt.subplot(2, 3, 5)
plt.plot(t_grid*1e15, np.abs(A_1w_initial)/np.max(np.abs(A_1w_initial)), 'k--', label='初始 1w 包络')
plt.plot(t_grid*1e15, np.abs(A_5w)/np.max(np.abs(A_5w)), 'm-', label='最终 5w 包络')
plt.xlim(-200, 200)
plt.xlabel('时间 (fs)')
plt.ylabel('归一化振幅 $|A|$')
plt.title('脉宽展宽与畸变对比 (GVD & SPM 综合效应)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

# ==========================================
# 5. 单独输出：全光路波长域归一化光谱
# ==========================================
def get_normalized_spectrum(A_t, lam_center_m):
    """计算绝对波长坐标 (nm) 与 自身归一化的线性光谱强度"""
    # 1. 计算线性光谱强度 |A(w)|^2 (不再使用对数 dB)
    spectrum_linear = np.abs(fftshift(fft(A_t)))**2
    
    # 2. 转移到绝对频率轴
    nu_abs = f_grid + (c / lam_center_m)
    
    # 3. 物理安全锁：截断非物理频率
    valid_idx = nu_abs > 10e12 
    nu_valid = nu_abs[valid_idx]
    spectrum_valid = spectrum_linear[valid_idx]
    
    # 4. 按该波段自身的最高峰值进行归一化 -> 使得每个光谱的最高点都是 1.0
    if np.max(spectrum_valid) > 0:
        spectrum_norm = spectrum_valid / np.max(spectrum_valid)
    else:
        spectrum_norm = spectrum_valid
        
    # 5. 转换为波长 (nm)
    lam_nm = (c / nu_valid) * 1e9
    return lam_nm, spectrum_norm

# 创建一个全新的独立画布 (单独弹出一个窗口)
plt.figure(figsize=(10, 6))

# 提取各阶光的归一化波长光谱
lam_nm_1, spec_norm_1 = get_normalized_spectrum(A1_depleted_1, lam_1w)
lam_nm_2, spec_norm_2 = get_normalized_spectrum(A_2w, lam_2w)
lam_nm_4, spec_norm_4 = get_normalized_spectrum(A_4w, lam_4w)
lam_nm_5, spec_norm_5 = get_normalized_spectrum(A_5w, lam_5w)

# 绘制曲线
plt.plot(lam_nm_1, spec_norm_1, 'r', linewidth=1.5, label='1w (1030 nm)')
plt.plot(lam_nm_2, spec_norm_2, 'g', linewidth=1.5, label='2w (515 nm)')
plt.plot(lam_nm_4, spec_norm_4, 'b', linewidth=1.5, label='4w (257.5 nm)')
plt.plot(lam_nm_5, spec_norm_5, 'm', linewidth=2.0, label='5w (206 nm)')

# 反转 X 轴 (高能短波长在左，低能长波长在右)
plt.xlim(1150, 180)
# Y 轴限制在 0 到 1.05 之间，完美展示归一化峰值
plt.ylim(0, 1.05)

plt.xlabel('波长 (nm)', fontsize=12)
plt.ylabel('自身归一化光谱强度 (a.u.)', fontsize=12)
plt.title('各阶谐波独立归一化光谱演化 (直观对比光谱展宽)', fontsize=14)
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.show()