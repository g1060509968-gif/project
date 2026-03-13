import numpy as np
from scipy.optimize import fsolve

class CrystalBBO:
    """
    BBO (Beta-Barium Borate) 非线性晶体介质计算器
    负单轴晶体 (n_e < n_o)，广泛用于超快激光的 SHG, THG, FHG, 5HG (直至深紫外)。
    """
    def __init__(self):
        # 光速 (m/s)
        self.c = 3e8
        
        # BBO 的高精度 Sellmeier 系数 (Kato 1986 / Eimerl 1987，适用范围 0.22 - 1.06 um)
        # 公式: n^2 = A + B / (lambda^2 - C) - D * lambda^2 (lambda 单位: um)
        self.sell_o = {'A': 2.7359, 'B': 0.01878, 'C': 0.01822, 'D': 0.01354}
        self.sell_e = {'A': 2.3753, 'B': 0.01224, 'C': 0.01667, 'D': 0.01516}
        
        # BBO 的非线性张量元 (pm/V)
        self.d22 = 2.2
        self.d31 = 0.16

    def n_o(self, lam_m):
        """计算 o 光 (寻常光) 折射率"""
        # [安全锁 1] 限制波长不低于 180nm，避开方程在深紫外的数学奇点
        lam_um = np.maximum(lam_m * 1e6, 0.18)
        n2 = self.sell_o['A'] + self.sell_o['B'] / (lam_um**2 - self.sell_o['C']) - self.sell_o['D'] * lam_um**2
        # [安全锁 2] 确保 n^2 不为负数，发生异常时强制折射率最小为 1 (真空)
        return np.sqrt(np.maximum(n2, 1.0))

    def n_e_principal(self, lam_m):
        """计算 e 光 (非常光) 主折射率"""
        lam_um = np.maximum(lam_m * 1e6, 0.18)
        n2 = self.sell_e['A'] + self.sell_e['B'] / (lam_um**2 - self.sell_e['C']) - self.sell_e['D'] * lam_um**2
        return np.sqrt(np.maximum(n2, 1.0))

    def n_e_theta(self, lam_m, theta_rad):
        """计算沿光轴夹角 theta (弧度) 传播的 e 光折射率 (折射率椭球公式)"""
        no = self.n_o(lam_m)
        ne = self.n_e_principal(lam_m)
        return 1.0 / np.sqrt((np.cos(theta_rad)**2 / no**2) + (np.sin(theta_rad)**2 / ne**2))

    def get_phase_matching_angle(self, lam1_m, lam2_m, match_type='I'):
        """
        自动寻根：计算完美相位匹配角 theta_pm
        对于 BBO (负单轴)，和频通常使用:
        Type-I: o + o -> e
        Type-II: o + e -> e
        """
        # 能量守恒: 计算生成的第三波长
        lam3_m = 1.0 / (1.0/lam1_m + 1.0/lam2_m)
        
        def phase_mismatch_type1(theta):
            # Type-I: k3(e) - k1(o) - k2(o) = 0
            n1 = self.n_o(lam1_m)
            n2 = self.n_o(lam2_m)
            n3 = self.n_e_theta(lam3_m, theta)
            return (n3 / lam3_m) - (n1 / lam1_m) - (n2 / lam2_m)
            
        def phase_mismatch_type2(theta):
            # Type-II: k3(e) - k1(o) - k2(e) = 0  (假设 lam1 是较长的波长，通常设为 o 光)
            n1 = self.n_o(lam1_m)
            n2 = self.n_e_theta(lam2_m, theta)
            n3 = self.n_e_theta(lam3_m, theta)
            return (n3 / lam3_m) - (n1 / lam1_m) - (n2 / lam2_m)

        # 初始猜测角设为 45 度 (pi/4)
        guess = np.pi / 4
        if match_type == 'I':
            theta_pm, = fsolve(phase_mismatch_type1, guess)
        elif match_type == 'II':
            theta_pm, = fsolve(phase_mismatch_type2, guess)
        else:
            raise ValueError("匹配类型必须是 'I' 或 'II'")
            
        return theta_pm % (np.pi / 2) # 限制在 0 - 90 度之间

    def calc_deff(self, theta_rad, phi_rad=np.pi/2, match_type='I'):
        """计算特定匹配角下的有效非线性系数 d_eff (返回单位: m/V)"""
        if match_type == 'I':
            # Type-I BBO (3m 点群) 公式
            deff_pmV = self.d31 * np.sin(theta_rad) - self.d22 * np.cos(theta_rad) * np.sin(3 * phi_rad)
        elif match_type == 'II':
            # Type-II BBO (3m 点群) 公式
            deff_pmV = self.d22 * np.cos(theta_rad)**2 * np.cos(3 * phi_rad)
        return np.abs(deff_pmV * 1e-12)

    def get_solver_n_funcs(self, theta_rad, match_type='I'):
        """
        桥接函数：提供偏振绑定的折射率函数句柄
        """
        def safe_lam(w):
            # [安全锁 3] 防止 FFT 网格中的负频率或零频率，最低限制在 10 THz (30 um 远红外)
            w_safe = np.maximum(w, 2 * np.pi * 10e12)
            return 2 * np.pi * self.c / w_safe

        if match_type == 'I':
            n1_func = lambda w: self.n_o(safe_lam(w))
            n2_func = lambda w: self.n_o(safe_lam(w))
            n3_func = lambda w: self.n_e_theta(safe_lam(w), theta_rad)
        elif match_type == 'II':
            n1_func = lambda w: self.n_o(safe_lam(w))
            n2_func = lambda w: self.n_e_theta(safe_lam(w), theta_rad)
            n3_func = lambda w: self.n_e_theta(safe_lam(w), theta_rad)
            
        return (n1_func, n2_func, n3_func)

class CrystalLBO:
    """
    LBO (Lithium Triborate) 非线性晶体介质计算器
    双轴晶体。本类专门针对论文中使用的 XY 平面 (theta = 90°) Type-I 相位匹配进行简化实现。
    """
    def __init__(self):
        self.c = 3e8
        
        # LBO 的 Sellmeier 系数 (Kato 1994)
        # 公式: n^2 = A + B / (lambda^2 - C) - D * lambda^2 (lambda 单位: um)
        self.sell_x = {'A': 2.454140, 'B': 0.011249, 'C': 0.011350, 'D': 0.014591}
        self.sell_y = {'A': 2.539070, 'B': 0.012711, 'C': 0.012523, 'D': 0.018540}
        self.sell_z = {'A': 2.586179, 'B': 0.013099, 'C': 0.011893, 'D': 0.017968}
        
        # d32 = 0.85 pm/V (通常用于 XY 平面的 Type I 匹配)
        self.d32 = 0.85 

    def _calc_n(self, lam_m, sell_coeffs):
        lam_um = np.maximum(lam_m * 1e6, 0.16)
        n2 = sell_coeffs['A'] + sell_coeffs['B'] / (lam_um**2 - sell_coeffs['C']) - sell_coeffs['D'] * lam_um**2
        return np.sqrt(np.maximum(n2, 1.0))

    def n_x(self, lam_m): return self._calc_n(lam_m, self.sell_x)
    def n_y(self, lam_m): return self._calc_n(lam_m, self.sell_y)
    def n_z(self, lam_m): return self._calc_n(lam_m, self.sell_z)

    def n_xy(self, lam_m, phi_rad):
        """计算在 XY 平面内，与 X 轴夹角为 phi 的偏振光的折射率"""
        nx = self.n_x(lam_m)
        ny = self.n_y(lam_m)
        # 椭球方程在 XY 平面的截面
        return 1.0 / np.sqrt((np.cos(phi_rad)**2 / nx**2) + (np.sin(phi_rad)**2 / ny**2))

    def get_phase_matching_angle(self, lam1_m, lam2_m, match_type='I'):
        """
        计算 XY 平面内的完美相位匹配角 phi
        匹配条件: n_z(1w) = n_xy(2w, phi)
        """
        if match_type == 'I':
            lam3_m = 1.0 / (1.0/lam1_m + 1.0/lam2_m)
            
            def eq(phi):
                # 基频光 (Z轴偏振)，倍频光 (XY平面偏振)
                n1 = self.n_z(lam1_m)
                n2 = self.n_z(lam2_m)
                n3 = self.n_xy(lam3_m, phi)
                return (n1/lam1_m + n2/lam2_m) - n3/lam3_m
                
            # 论文中 phi = 12.9° (通常是与 Y 轴的夹角，即与 X 轴夹角 90-12.9 = 77.1°)
            phi_guess = np.deg2rad(77.1) 
            phi_pm = fsolve(eq, phi_guess)[0]
            return phi_pm
        else:
            raise NotImplementedError("目前仅实现了 LBO 在 XY 平面的 Type-I 匹配。")

    def calc_deff(self, phi_rad, match_type='I'):
        """计算有效非线性系数 d_eff"""
        if match_type == 'I':
            # XY 平面 Type I 的 d_eff = d32 * cos(phi)
            return self.d32 * np.cos(phi_rad) * 1e-12
        return 0

    def get_solver_n_funcs(self, phi_rad, match_type='I'):
        """返回供 solver 使用的折射率函数 (n1_func, n2_func, n3_func)"""
        if match_type == 'I':
            # 基频光 (1, 2) 偏振沿 Z 轴，倍频光 (3) 偏振在 XY 平面
            n1_func = lambda lam: self.n_z(lam)
            n2_func = lambda lam: self.n_z(lam)
            n3_func = lambda lam: self.n_xy(lam, phi_rad)
            return (n1_func, n2_func, n3_func)
