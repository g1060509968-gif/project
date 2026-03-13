import numpy as np
from scipy.fft import fft, ifft, fftshift, ifftshift, fftfreq

def get_nonlinear_derivatives(A1, A2, A3, kappa, gamma_matrix, beta_matrix, delta_k, z):
    """
    内部辅助函数：计算当前空间步长 z 处的非线性偏导数 (dA/dz)
    包含：
    1. 二阶耦合 (chi2) 及其宏观相位失配
    2. 完整的 Kerr 张量 (chi3): 自相位调制 (SPM) + 交叉相位调制 (XPM)
    3. 完整的 TPA 张量: 自双光子吸收 (Self-TPA) + 交叉双光子吸收 (Cross-TPA)
    """
    I1 = np.abs(A1)**2
    I2 = np.abs(A2)**2
    I3 = np.abs(A3)**2
    
    # 计算当前的宏观相位失配因子 (Phase Mismatch)
    # 对于 w3 = w1 + w2，定义 delta_k = k3 - k1 - k2
    phase_mismatch = np.exp(1j * delta_k * z)
    
    # 1. chi(2) 三波混频项 (能量转移)
    # 注意共轭和相位因子的符号对应关系
    dA1_dz_chi2 = -1j * kappa[0] * A3 * np.conj(A2) * np.conj(phase_mismatch)
    dA2_dz_chi2 = -1j * kappa[1] * A3 * np.conj(A1) * np.conj(phase_mismatch)
    dA3_dz_chi2 = -1j * kappa[2] * A1 * A2 * phase_mismatch
    
    # 2. chi(3) 克尔效应 (SPM 与 XPM)
    # gamma_matrix[i, j] 表示场 j 的光强对场 i 造成的相位调制系数
    dA1_dz_kerr = -1j * (gamma_matrix[0,0]*I1 + gamma_matrix[0,1]*I2 + gamma_matrix[0,2]*I3) * A1
    dA2_dz_kerr = -1j * (gamma_matrix[1,0]*I1 + gamma_matrix[1,1]*I2 + gamma_matrix[1,2]*I3) * A2
    dA3_dz_kerr = -1j * (gamma_matrix[2,0]*I1 + gamma_matrix[2,1]*I2 + gamma_matrix[2,2]*I3) * A3
    
    # 3. 双光子吸收 (Self-TPA 与 Cross-TPA)
    # beta_matrix[i, j] 表示场 j 的光强导致场 i 发生的吸收
    # 对角线系数需除以 2 (Self-TPA 物理惯例)，非对角线不除 (Cross-TPA 协同吸收)
    dA1_dz_tpa = - ( (beta_matrix[0,0]/2)*I1 + beta_matrix[0,1]*I2 + beta_matrix[0,2]*I3 ) * A1
    dA2_dz_tpa = - ( beta_matrix[1,0]*I1 + (beta_matrix[1,1]/2)*I2 + beta_matrix[1,2]*I3 ) * A2
    dA3_dz_tpa = - ( beta_matrix[2,0]*I1 + beta_matrix[2,1]*I2 + (beta_matrix[2,2]/2)*I3 ) * A3
    
    # 汇总各阶非线性导数
    dA1_dz = dA1_dz_chi2 + dA1_dz_kerr + dA1_dz_tpa
    dA2_dz = dA2_dz_chi2 + dA2_dz_kerr + dA2_dz_tpa
    dA3_dz = dA3_dz_chi2 + dA3_dz_kerr + dA3_dz_tpa
    
    return dA1_dz, dA2_dz, dA3_dz

def ultrafast_twm_solver(A1_in, A2_in, A3_in, w, n_funcs, L, dz, 
                         d_eff, gamma_matrix, beta_matrix, alpha, t_grid):
    """
    超快非线性三波混频 (TWM) 终极核心求解器
    架构：频域 SVEA (全光谱解析色散) + 分步傅里叶方法 (SSFM) + 四阶龙格-库塔 (RK4)
    
    参数:
    A1_in, A2_in, A3_in : 输入的时域复包络 numpy 数组 (若某场无初始输入则传 np.zeros_like(t_grid))
    w                   : 元组 (w1, w2, w3)，中心角频率 (rad/s)
    n_funcs             : 元组 (n1_func, n2_func, n3_func)，针对每个光场的精确折射率函数句柄
    L, dz               : 晶体总长度和空间计算步长 (m)
    d_eff               : 有效二阶非线性系数 (m/V)
    gamma_matrix        : 3x3 numpy 数组，克尔系数张量 (W^-1 m^-1)
    beta_matrix         : 3x3 numpy 数组，双光子吸收张量 (m/W)
    alpha               : 元组 (alpha1, alpha2, alpha3)，线性吸收系数 (1/m)
    t_grid              : 时间网格数组 (s)
    
    返回:
    A1_out, A2_out, A3_out : 传播距离 L 后的时域复包络
    """
    c = 3e8
    N = len(t_grid)
    dt = t_grid[1] - t_grid[0]
    
    # 频率网格构建
    # fftfreq 默认 0 频率偏移在索引 0，fftshift 后中心频率 (w_i) 移动到索引 N//2
    f_grid = fftshift(fftfreq(N, d=dt))
    omega_grid = 2 * np.pi * f_grid
    
    # 提取各波段中心折射率
    n1_0 = n_funcs[0](w[0])
    n2_0 = n_funcs[1](w[1])
    n3_0 = n_funcs[2](w[2])
    
    # 二阶非线性耦合系数 kappa
    kappa = (w[0] * d_eff / (n1_0 * c),
             w[1] * d_eff / (n2_0 * c),
             w[2] * d_eff / (n3_0 * c))
    
    # -----------------------------------------------------------------
    # 频域绝对色散与损耗算子构建 (全光谱，无泰勒级数近似)
    # -----------------------------------------------------------------
    k1_exact = (w[0] + omega_grid) * n_funcs[0](w[0] + omega_grid) / c
    k2_exact = (w[1] + omega_grid) * n_funcs[1](w[1] + omega_grid) / c
    k3_exact = (w[2] + omega_grid) * n_funcs[2](w[2] + omega_grid) / c
    
    # 计算中心角频率处的绝对相位失配
    delta_k = k3_exact[N//2] - k1_exact[N//2] - k2_exact[N//2]
    
    # 扣除载波波数，将观察坐标系转移至各自脉冲的群速度系，防止包络溢出网格边界
    idx = N // 2
    dw = omega_grid[idx + 1] - omega_grid[idx - 1]
    k1_prime = (k1_exact[idx + 1] - k1_exact[idx - 1]) / dw
    
    # 扣除载波波数的同时，必须扣除群速度带来的绝对时间漂移！
    # 这等效于让时间网格以 A1 脉冲的群速度跟着一起飞
    k1_shift = k1_exact - k1_exact[idx] - k1_prime * omega_grid
    k2_shift = k2_exact - k2_exact[idx] - k1_prime * omega_grid
    k3_shift = k3_exact - k3_exact[idx] - k1_prime * omega_grid
    
    # 预计算频域演化因子 (色散 + 线性吸收)
    linear_operator_1 = np.exp(-1j * k1_shift * dz - (alpha[0]/2) * dz)
    linear_operator_2 = np.exp(-1j * k2_shift * dz - (alpha[1]/2) * dz)
    linear_operator_3 = np.exp(-1j * k3_shift * dz - (alpha[2]/2) * dz)
    
    # 场初始化 (转换到频域作为计算起点)
    A1_w = fftshift(fft(A1_in))
    A2_w = fftshift(fft(A2_in))
    A3_w = fftshift(fft(A3_in))
    
    steps = int(np.ceil(L / dz))
    
    # SSFM 主循环
    for step in range(steps):
        # 当前所在的晶体宏观位置 (用于传递给相位失配因子)
        current_z = step * dz
        
        # 转换到时域计算非线性相互作用
        A1 = ifft(ifftshift(A1_w))
        A2 = ifft(ifftshift(A2_w))
        A3 = ifft(ifftshift(A3_w))
        
        # --- 四阶龙格-库塔 (RK4) 积分器计算非线性演化步 ---
        # k1 (处于 current_z)
        k1_A1, k1_A2, k1_A3 = get_nonlinear_derivatives(A1, A2, A3, kappa, gamma_matrix, beta_matrix, delta_k, current_z)
        
        # k2 (处于 current_z + dz/2)
        A1_k2 = A1 + 0.5 * dz * k1_A1
        A2_k2 = A2 + 0.5 * dz * k1_A2
        A3_k2 = A3 + 0.5 * dz * k1_A3
        k2_A1, k2_A2, k2_A3 = get_nonlinear_derivatives(A1_k2, A2_k2, A3_k2, kappa, gamma_matrix, beta_matrix, delta_k, current_z + 0.5 * dz)
        
        # k3 (处于 current_z + dz/2)
        A1_k3 = A1 + 0.5 * dz * k2_A1
        A2_k3 = A2 + 0.5 * dz * k2_A2
        A3_k3 = A3 + 0.5 * dz * k2_A3
        k3_A1, k3_A2, k3_A3 = get_nonlinear_derivatives(A1_k3, A2_k3, A3_k3, kappa, gamma_matrix, beta_matrix, delta_k, current_z + 0.5 * dz)
        
        # k4 (处于 current_z + dz)
        A1_k4 = A1 + dz * k3_A1
        A2_k4 = A2 + dz * k3_A2
        A3_k4 = A3 + dz * k3_A3
        k4_A1, k4_A2, k4_A3 = get_nonlinear_derivatives(A1_k4, A2_k4, A3_k4, kappa, gamma_matrix, beta_matrix, delta_k, current_z + dz)
        
        # 整合 RK4 结果并更新场
        A1_nl = A1 + (dz / 6.0) * (k1_A1 + 2*k2_A1 + 2*k3_A1 + k4_A1)
        A2_nl = A2 + (dz / 6.0) * (k1_A2 + 2*k2_A2 + 2*k3_A2 + k4_A2)
        A3_nl = A3 + (dz / 6.0) * (k1_A3 + 2*k2_A3 + 2*k3_A3 + k4_A3)
        
        # --- 频域线性步：施加全光谱色散与线性损耗 ---
        A1_w = fftshift(fft(A1_nl)) * linear_operator_1
        A2_w = fftshift(fft(A2_nl)) * linear_operator_2
        A3_w = fftshift(fft(A3_nl)) * linear_operator_3
        
    # 空间传播结束，转换回时域输出最终包络
    A1_out = ifft(ifftshift(A1_w))
    A2_out = ifft(ifftshift(A2_w))
    A3_out = ifft(ifftshift(A3_w))
    
    return A1_out, A2_out, A3_out