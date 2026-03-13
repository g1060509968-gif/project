import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftshift, fftfreq, ifft, ifftshift
from scipy.interpolate import interp1d

# 导入晶体数据库和求解器 (需确保 crystal_database.py 和 solver.py 在同目录下)
from crystal_database import CrystalBBO, CrystalLBO
from solver import ultrafast_twm_solver

# ==========================================
# 1. 全局仿真环境设置
# ==========================================
c = 299792458.0 #光速
# 时间网格: 扩大到 8192 个点，覆盖 8000 飞秒窗口，防止厚晶体走离导致脉冲溢出边界
N = 8192
T_window = 8000e-15
t_grid = np.linspace(-T_window/2, T_window/2, N, endpoint=False)
dt = t_grid[1] - t_grid[0]

# 频率网格 (用于绘制光谱)
f_grid = fftshift(fftfreq(N, d=dt))
nu_THz = f_grid * 1e-12

# 实例化晶体介质数据库
bbo = CrystalBBO()
lbo = CrystalLBO()

# ==========================================
# 2. 初始脉冲定义 (基于真实的 CSV 光谱与相位重建)
# ==========================================
# 假设你的 CSV 文件名为 spectrum_1030nm.csv，请根据实际文件名修改
csv_filename = 'd:\project\src\倍频\spectrum_1030nm.csv'

# 论文实际测量的基频中心波长为 1029.3 nm
lam_center = 1029.3e-9 
nu_0 = c / lam_center
w_1 = 2 * np.pi * nu_0

# 构建绝对频率坐标系 (对应于 solver 内部的 f_grid)
nu_abs = f_grid + nu_0

# --- 第 1 步：加载 CSV 光谱数据并插值 ---
# 读取 CSV 文件，delimiter=',' 表示逗号分隔。
# (注意: 如果你的第一行是表头字母，请在 loadtxt 中加上 skiprows=1)
data = np.genfromtxt(csv_filename, delimiter=',', skip_header=1, filling_values=0.0)

# 提取数据：假设第一列是波长 (单位 nm)，第二列是相对强度
lam_data = data[:, 0] * 1e-9  
intensity_data = data[:, 1]

# 将波长转换为频率
nu_data = c / lam_data

# 因为波长从小到大，对应的频率是从大到小，所以需要按频率从小到大对数组进行排序
# (插值函数要求 X 轴必须是严格递增的)
sort_idx = np.argsort(nu_data)
nu_data_sorted = nu_data[sort_idx]
intensity_data_sorted = intensity_data[sort_idx]

# 构建插值函数 (kind='cubic' 让曲线平滑，超出数据范围的频率强度设为 0)
interp_func = interp1d(nu_data_sorted, intensity_data_sorted, 
                       kind='cubic', bounds_error=False, fill_value=0)

# 将真实光谱映射到我们仿真的高精度频率网格上
S_nu = interp_func(nu_abs)
S_nu[S_nu < 0] = 0 # 物理安全锁：消除三次插值可能产生的微小负值

# --- 第 2 步：引入群延时色散 (GDD) 作为啁啾 ---
# 论文中光谱极宽但脉宽为 433fs。我们通过 GDD 引入真实的二阶色散相位来展宽脉冲。
# 你可以微调这个 GDD 的数值(比如在 2.0e-26 到 3.5e-26 之间)，直到第五子图里的初始 1w 脉宽刚好约等于 433fs。
GDD = 2.8e-26  
spectral_phase = 0.5 * GDD * (2 * np.pi * f_grid)**2

# --- 第 3 步：合成带啁啾的复数频域电场 ---
# E(w) = |E(w)| * exp(i * phi)  (振幅是强度的平方根)
A_1w_w = np.sqrt(S_nu) * np.exp(1j * spectral_phase)

# --- 第 4 步：转换回时域并定标能量 ---
A_1w_t = ifft(ifftshift(A_1w_w))

# 根据实验平均功率(2.4W, 1MHz, 258um)定标真实的峰值电场
E_peak = 1.3e8 # V/m
A_1w_initial = A_1w_t / np.max(np.abs(A_1w_t)) * E_peak

lam_1w = lam_center # 更新全局波长基准供后续代码使用

# ==========================================
# 3. 级联非线性光学张量预设 (唯象参数)
# ==========================================
# 克尔张量 (SPM & XPM) - 适当降低以匹配较长晶体，防止频谱过度爆炸
g0 = 5e-17  
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

# 论文参数：3mm LBO 晶体，theta=90°
phi_shg = lbo.get_phase_matching_angle(lam_1w, lam_1w, match_type='I')
deff_shg = lbo.calc_deff(phi_shg, match_type='I')
n_funcs_shg = lbo.get_solver_n_funcs(phi_shg, 'I')
beta_0 = np.zeros((3, 3))

A1_depleted_1, _, A_2w = ultrafast_twm_solver(
    A1_in=A_1w_initial, A2_in=A_1w_initial, A3_in=np.zeros_like(t_grid),
    w=(w_1, w_1, w_2), n_funcs=n_funcs_shg,
    L=3.0e-3, dz=10e-6, d_eff=deff_shg,  # 论文：晶体厚度 3 mm
    gamma_matrix=gamma_matrix, beta_matrix=beta_0, alpha=(0, 0, 0), t_grid=t_grid
)

# ==========================================
# 阶段 2：四倍频 FHG (2w + 2w -> 4w)
# ==========================================
print(">> 开始计算第二级：四倍频 (515nm -> 257.5nm) ...")
lam_4w = lam_2w / 2
w_4 = 4 * w_1

# 论文参数：1mm BBO 晶体，theta=50°
theta_fhg = bbo.get_phase_matching_angle(lam_2w, lam_2w, match_type='I')
deff_fhg = bbo.calc_deff(theta_fhg, match_type='I')
n_funcs_fhg = bbo.get_solver_n_funcs(theta_fhg, 'I')

# 257.5nm 轻微吸收与 TPA
beta_fhg = np.zeros((3, 3))
beta_fhg[2, 2] = 1e-11 

_, _, A_4w = ultrafast_twm_solver(
    A1_in=A_2w, A2_in=A_2w, A3_in=np.zeros_like(t_grid),
    w=(w_2, w_2, w_4), n_funcs=n_funcs_fhg,
    L=1.0e-3, dz=5e-6, d_eff=deff_fhg, # 论文：晶体厚度 1 mm
    gamma_matrix=gamma_matrix, beta_matrix=beta_fhg, alpha=(0, 0, 10), t_grid=t_grid
)

# ==========================================
# 阶段 3：五倍频 5HG (1w + 4w -> 5w)
# ==========================================
print(">> 开始计算第三级：五倍频 (1030nm + 257.5nm -> 206nm) ...")
lam_5w = lam_1w / 5
w_5 = 5 * w_1

# 论文参数：1mm BBO 晶体
theta_5hg = bbo.get_phase_matching_angle(lam_1w, lam_4w, match_type='I')
deff_5hg = bbo.calc_deff(theta_5hg, match_type='I')
n_funcs_5hg = bbo.get_solver_n_funcs(theta_5hg, 'I')

# 深紫外 TPA 吸收
beta_5hg = np.zeros((3, 3))
beta_5hg[1, 1] = 2e-11  
beta_5hg[2, 2] = 5e-11  
beta_5hg[0, 1] = 1e-11  
beta_5hg[1, 0] = 1e-11  

# [光学延迟线 (Delay Line)]：完美对应论文中补偿 1945.1 fs 时空走离的延迟线系统！
# 强制将 4w 的峰值平移到 t=0，与残余的 1w 重新在时域上重合
shift_idx = (N // 2) - np.argmax(np.abs(A_4w))
A_4w_aligned = np.roll(A_4w, shift_idx)

A1_final, A4_final, A_5w = ultrafast_twm_solver(
    A1_in=A1_depleted_1, A2_in=A_4w_aligned, A3_in=np.zeros_like(t_grid),
    w=(w_1, w_4, w_5), n_funcs=n_funcs_5hg,
    L=1.0e-3, dz=5e-6, d_eff=deff_5hg, # 论文：晶体厚度 1 mm
    gamma_matrix=gamma_matrix, beta_matrix=beta_5hg, 
    alpha=(0, 10, 500), # 模拟 206nm 吸收
    t_grid=t_grid
)

print(">> 仿真计算完成！正在生成分析图像...")

# ==========================================
# 4. 可视化分析：多维数据绘图 (5子图架构)
# ==========================================
def get_spectrum(A_t):
    return 10 * np.log10(np.abs(fftshift(fft(A_t)))**2 + 1e-10)

def get_wavelength_spectrum(A_t, lam_center_m):
    spectrum_dB = get_spectrum(A_t)
    nu_abs = f_grid + (c / lam_center_m)
    valid_idx = nu_abs > 10e12 
    nu_valid = nu_abs[valid_idx]
    spectrum_valid = spectrum_dB[valid_idx]
    lam_nm = (c / nu_valid) * 1e9
    return lam_nm, spectrum_valid

plt.figure(figsize=(18, 10))
plt.rcParams['font.sans-serif'] = ['SimHei'] 
plt.rcParams['axes.unicode_minus'] = False

# [子图 1] 第一级：SHG 时域波形
plt.subplot(2, 3, 1)
plt.plot(t_grid*1e15, np.abs(A_1w_initial)**2, 'k--', label='初始 1w (1030 nm)')
plt.plot(t_grid*1e15, np.abs(A1_depleted_1)**2, 'r-', label='消耗后 1w')
plt.plot(t_grid*1e15, np.abs(A_2w)**2, 'g-', label='生成 2w (515 nm)')
plt.xlim(-1000, 1000) # 根据脉宽放大坐标系
plt.xlabel('时间 (fs)')
plt.ylabel('光强 $|A|^2$')
plt.title('Stage 1: SHG (时域 - 3mm LBO)')
plt.legend()
plt.grid(True, alpha=0.3)

# [子图 2] 最终级：5HG 时域波形
ax1 = plt.subplot(2, 3, 2)
ax1.plot(t_grid*1e15, np.abs(A1_depleted_1)**2, 'r--', alpha=0.5, label='注入的 1w')
ax1.plot(t_grid*1e15, np.abs(A_4w_aligned)**2, 'b--', alpha=0.5, label='延迟线对齐后的 4w')
ax1.set_xlabel('时间 (fs)')
ax1.set_ylabel('1w/4w 光强 $|A|^2$', color='k')
ax1.set_xlim(-1000, 1000)

ax2 = ax1.twinx()
ax2.plot(t_grid*1e15, np.abs(A_5w)**2, 'm-', linewidth=2, label='最终 5w (206 nm)')
ax2.set_ylabel('5w 光强 $|A|^2$ (放大显示)', color='m')

plt.title('Stage 3: 5HG 深紫外脉冲生成 (时域 - 1mm BBO)')
lines_1, labels_1 = ax1.get_legend_handles_labels()
lines_2, labels_2 = ax2.get_legend_handles_labels()
ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
ax1.grid(True, alpha=0.3)

# [子图 3] 绝对频率光谱
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

# [子图 4] 波长光谱
plt.subplot(2, 3, 4)
lam_nm_1, spec_1 = get_wavelength_spectrum(A1_depleted_1, lam_1w)
lam_nm_2, spec_2 = get_wavelength_spectrum(A_2w, lam_2w)
lam_nm_4, spec_4 = get_wavelength_spectrum(A_4w, lam_4w)
lam_nm_5, spec_5 = get_wavelength_spectrum(A_5w, lam_5w)

plt.plot(lam_nm_1, spec_1, 'r', label='1w (1030 nm)')
plt.plot(lam_nm_2, spec_2, 'g', label='2w (515 nm)')
plt.plot(lam_nm_4, spec_4, 'b', label='4w (257.5 nm)')
plt.plot(lam_nm_5, spec_5, 'm', label='5w (206 nm)')

plt.xlim(1200, 180) 
plt.ylim(60, 200)
plt.xlabel('波长 (nm)')
plt.ylabel('谱密度 (dB)')
plt.title('各阶谐波光谱 (波长域)')
plt.legend()
plt.grid(True, alpha=0.3)

# [子图 5] 脉冲形变对比
plt.subplot(2, 3, 5)
plt.plot(t_grid*1e15, np.abs(A_1w_initial)/np.max(np.abs(A_1w_initial)), 'k--', label='初始 1w 包络')
plt.plot(t_grid*1e15, np.abs(A_5w)/np.max(np.abs(A_5w)), 'm-', label='最终 5w 包络 (含走离延迟)')

# 将视窗向右扩展，因为深紫外脉冲群速度极慢，会严重滞后！
plt.xlim(-500, 2500) 
plt.xlabel('时间 (fs)')
plt.ylabel('归一化振幅 $|A|$')
plt.title('脉宽展宽与严重走离 (GVD 效应)')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()

# ==========================================
# 5. 归一化光谱独立窗口
# ==========================================
def get_normalized_spectrum(A_t, lam_center_m):
    spectrum_linear = np.abs(fftshift(fft(A_t)))**2
    nu_abs = f_grid + (c / lam_center_m)
    valid_idx = nu_abs > 10e12 
    nu_valid = nu_abs[valid_idx]
    spectrum_valid = spectrum_linear[valid_idx]
    
    if np.max(spectrum_valid) > 0:
        spectrum_norm = spectrum_valid / np.max(spectrum_valid)
    else:
        spectrum_norm = spectrum_valid
        
    lam_nm = (c / nu_valid) * 1e9
    return lam_nm, spectrum_norm

plt.figure(figsize=(10, 6))

lam_nm_1, spec_norm_1 = get_normalized_spectrum(A1_depleted_1, lam_1w)
lam_nm_2, spec_norm_2 = get_normalized_spectrum(A_2w, lam_2w)
lam_nm_4, spec_norm_4 = get_normalized_spectrum(A_4w, lam_4w)
lam_nm_5, spec_norm_5 = get_normalized_spectrum(A_5w, lam_5w)

plt.plot(lam_nm_1, spec_norm_1, 'r', linewidth=1.5, label='1w (1030 nm)')
plt.plot(lam_nm_2, spec_norm_2, 'g', linewidth=1.5, label='2w (515 nm)')
plt.plot(lam_nm_4, spec_norm_4, 'b', linewidth=1.5, label='4w (257.5 nm)')
plt.plot(lam_nm_5, spec_norm_5, 'm', linewidth=2.0, label='5w (206 nm)')

plt.xlim(1150, 180)
plt.ylim(0, 1.05)
plt.xlabel('波长 (nm)', fontsize=12)
plt.ylabel('自身归一化光谱强度 (a.u.)', fontsize=12)
plt.title('各阶谐波独立归一化光谱演化', fontsize=14)
plt.legend(loc='upper right')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()