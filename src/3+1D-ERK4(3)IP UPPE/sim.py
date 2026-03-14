import matplotlib
matplotlib.use('Agg')  # 后台绘图模式

import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
import time
import os
from solver_3d import ERK43IP_UPPE_3D_Optimized

class MPC_Final_Runner:
    def __init__(self):
        self.output_dir = "mpc_results_fixed_zoom_1"
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # --- 1. 激光参数 ---
        self.wavelength = 1064e-9
        self.pulse_duration = 10.7e-12 

        self.avg_power = 50.0  
        self.rep_rate = 500e3
        self.pulse_energy = self.avg_power / self.rep_rate  

        self.fwhm_to_tau = 1.0 / (2 * np.sqrt(np.log(2)))
        self.tau = self.pulse_duration * self.fwhm_to_tau

        # --- 2. 腔体几何 ---
        self.L_cavity = 212.5e-3
        self.R_concave = 300e-3

        self.w0_waist = self.calculate_eigenmode(self.L_cavity, self.R_concave, self.wavelength)
        print(f"[Physics] Calculated Eigenmode Waist w0 = {self.w0_waist*1e6:.2f} um")

        self.d_fs = 12e-3
        self.d_air_1 = 30e-3
        self.d_air_mid = 25e-3
        occupied = self.d_air_1 + 4 * self.d_fs + 3 * self.d_air_mid
        self.d_air_last = self.L_cavity - occupied

        self.cycles = 33
        self.total_trans = 53.9 / 100.0
        self.T_per_round = self.total_trans ** (1 / self.cycles)

        # --- 3. 网格与求解器 ---
        self.window_width = 2.0e-3  
        self.Nxy = 256
        self.Nt = 2046
        self.time_window = 80e-12

        x = np.linspace(-self.window_width / 2, self.window_width / 2, self.Nxy)
        y = np.linspace(-self.window_width / 2, self.window_width / 2, self.Nxy)
        t = np.linspace(-self.time_window / 2, self.time_window / 2, self.Nt)

        print(f"[Init] Allocating Solvers (Grid: {self.Nxy}x{self.Nxy}x{self.Nt}, Window: 2mm)...")

        # [修复材料] 数值问题解决后，恢复 FS 材料的真实物理属性 (n2=3.0e-20)
        self.solver_fs = ERK43IP_UPPE_3D_Optimized(
            x, y, t, material='fused_silica', n2=3.0e-20, center_wavelength=self.wavelength,
            use_shock=False, use_raman=False, precision='single'
        )
        self.solver_air = ERK43IP_UPPE_3D_Optimized(
            x, y, t, material='air', n2=0.0, center_wavelength=self.wavelength,
            use_shock=False, use_raman=False, precision='single'
        )

        self._precompute_mirror_phase()

        self.viz_wl_min_nm = 1050.0
        self.viz_wl_max_nm = 1080.0
        self.viz_time_ps_min = -30.0
        self.viz_time_ps_max = 30.0
        
        # 归一化缩放因子 (初始化为 1.0)
        self.A_scale = 1.0

    def calculate_eigenmode(self, L, R, wl):
        if L >= R: raise ValueError("Unstable Cavity!")
        w0_sq = (wl / np.pi) * np.sqrt(L * (R - L))
        return np.sqrt(w0_sq)

    def _precompute_mirror_phase(self):
        X, Y = cp.meshgrid(self.solver_fs.x, self.solver_fs.y, indexing='ij')
        R2 = X**2 + Y**2
        c0 = 299792458.0
        omega0 = 2 * np.pi * c0 / self.wavelength
        omega_full = omega0 + self.solver_fs.omega
        k_omega = omega_full / c0
        phase_3d = -1j * k_omega[cp.newaxis, cp.newaxis, :] * R2[:, :, cp.newaxis] / self.R_concave
        self.mirror_phase = cp.exp(phase_3d).astype(self.solver_fs.complex_dtype)

    def apply_curved_mirror(self, field_norm):
        # 纯相位操作，直接对归一化场进行
        field_w = cp.fft.fft(field_norm, axis=-1)
        field_w_reflected = field_w * self.mirror_phase
        return cp.fft.ifft(field_w_reflected, axis=-1)

    def generate_initial_field(self):
        cp.get_default_memory_pool().free_all_blocks()
        X, Y = np.meshgrid(self.solver_fs.x.get(), self.solver_fs.y.get(), indexing='ij')
        T = self.solver_fs.t.get()

        spatial = np.exp(-(X**2 + Y**2) / (self.w0_waist**2))
        temporal = np.exp(-(T**2) / (self.tau**2))
        field = spatial[:, :, np.newaxis] * temporal[np.newaxis, np.newaxis, :]

        # 使用双精度计算初始能量，避免求和截断
        current_E = np.sum(np.abs(field).astype(np.float64)**2) * self.solver_fs.dx * self.solver_fs.dy * self.solver_fs.dt
        norm_factor = np.sqrt(self.pulse_energy / current_E)
        
        # 计算物理场并记录最大振幅作为全局缩放因子
        field_physical = field * norm_factor
        self.A_scale = float(np.max(np.abs(field_physical)))
        if self.A_scale < 1e-30: self.A_scale = 1.0
        
        # 返回归一化场 (量级 O(1))
        field_norm = field_physical / self.A_scale
        return cp.asarray(field_norm, dtype=self.solver_fs.complex_dtype)

    def propagate_segment(self, field_norm, distance, medium_type):
        if cp.any(cp.isnan(field_norm)):
            field_norm = cp.nan_to_num(field_norm)

        # 解包回物理场，以激发正确的非线性 Kerr 效应强度
        field_physical = field_norm * self.A_scale

        if medium_type == 'air':
            step = min(distance, 5e-3)
            _, _, A_spec = self.solver_air.propagate(field_physical, distance, max_step=step, tol=1e-4)
        else:
            _, _, A_spec = self.solver_fs.propagate(field_physical, distance, max_step=20e-6, tol=1e-5)

        # 变回时域，并再次封装回避浮点极限
        A_out_physical = cp.fft.ifft(cp.fft.ifftshift(cp.asarray(A_spec), axes=-1), axis=-1)
        return A_out_physical / self.A_scale

    # =========================
    # [强化版] 双精度诊断工具函数
    # =========================
    def _calc_energy(self, field_norm):
        # 强制将海量元素平方操作转为 float64，防止 GPU 上严重的大数吞小数(Swamping)现象
        integral = cp.sum(cp.abs(field_norm).astype(cp.float64)**2)
        return float(integral * self.solver_fs.dx * self.solver_fs.dy * self.solver_fs.dt)

    def _calc_beam_moments(self, fluence):
        x = self.solver_fs.x
        y = self.solver_fs.y
        X, Y = cp.meshgrid(x, y, indexing='ij')

        total = cp.sum(fluence.astype(cp.float64)) * self.solver_fs.dx * self.solver_fs.dy + 1e-30
        x_mean = cp.sum(X * fluence) * self.solver_fs.dx * self.solver_fs.dy / total
        y_mean = cp.sum(Y * fluence) * self.solver_fs.dx * self.solver_fs.dy / total

        x2 = cp.sum((X - x_mean)**2 * fluence) * self.solver_fs.dx * self.solver_fs.dy / total
        y2 = cp.sum((Y - y_mean)**2 * fluence) * self.solver_fs.dx * self.solver_fs.dy / total

        return float(x_mean), float(y_mean), float(cp.sqrt(x2)), float(cp.sqrt(y2))

    def _fwhm_from_intensity(self, t_cpu, I_cpu):
        I, t = np.asarray(I_cpu).astype(np.float64), np.asarray(t_cpu).astype(np.float64)
        if I.size < 4: return np.nan
        imax = np.max(I)
        if not np.isfinite(imax) or imax <= 0: return np.nan
        half = 0.5 * imax
        i_peak = int(np.argmax(I))

        il = i_peak
        while il > 0 and I[il] >= half: il -= 1
        ir = i_peak
        while ir < I.size - 1 and I[ir] >= half: ir += 1
        if ir <= il + 1: return np.nan

        def interp(i1, i2):
            y1, y2 = I[i1], I[i2]
            return t[i1] + (half - y1) * (t[i2] - t[i1]) / (y2 - y1 + 1e-30)

        t_left = interp(il, il + 1) if il + 1 < I.size else t[il]
        t_right = interp(ir - 1, ir) if ir - 1 >= 0 else t[ir]
        fwhm = t_right - t_left
        return float(fwhm) if (fwhm > 0 and np.isfinite(fwhm)) else np.nan

    def _norm(self, a):
        a = np.asarray(a)
        m = np.max(a) if np.max(a) > 0 else 1.0
        return a / (m + 1e-30)

    def save_snapshot(self, cycle, field_norm, label):
        field_clean = cp.nan_to_num(field_norm)
        scale_I = self.A_scale ** 2  # 还原物理单位的强度乘子

        # =========================
        # 1) 基本量：物理单位恢复与双精度求和
        # =========================
        E_total = self._calc_energy(field_clean) * scale_I
        t_cpu = self.solver_fs.t.get()
        t_ps = t_cpu * 1e12

        P_t = cp.sum(cp.abs(field_clean).astype(cp.float64)**2, axis=(0, 1)) * self.solver_fs.dx * self.solver_fs.dy * scale_I
        P_t_cpu = cp.asnumpy(P_t)

        fluence = cp.sum(cp.abs(field_clean).astype(cp.float64)**2, axis=-1) * self.solver_fs.dt * scale_I
        x_mean, y_mean, wx_rms, wy_rms = self._calc_beam_moments(fluence)
        flu_cpu = cp.asnumpy(fluence)

        it_peak = int(cp.argmax(P_t).get())
        A_xy_physical = field_clean[:, :, it_peak] * self.A_scale
        A_xy_amp = cp.abs(A_xy_physical)
        A_xy_phase = cp.angle(A_xy_physical)

        # =========================
        # 2) 光谱计算
        # =========================
        field_w = cp.fft.fftshift(cp.fft.fft(field_clean, axis=-1), axes=-1)
        X, Y = cp.meshgrid(self.solver_fs.x, self.solver_fs.y, indexing='ij')
        fiber_mode = cp.exp(-(X**2 + Y**2) / (self.w0_waist**2)).astype(self.solver_fs.float_dtype)[:, :, cp.newaxis]

        coupled_field_w = cp.sum(field_w * fiber_mode, axis=(0, 1))
        spec_smf = cp.abs(coupled_field_w)**2 * scale_I
        spec_full_incoh = cp.sum(cp.abs(field_w).astype(cp.float64)**2, axis=(0, 1)) * scale_I
        ix0, iy0 = self.Nxy // 2, self.Nxy // 2
        spec_center = cp.abs(field_w[ix0, iy0, :])**2 * scale_I

        freq_rel = cp.asnumpy(self.solver_fs.freq_axis_hz)
        c0 = 2.99792458e8
        freq_abs = freq_rel + (c0 / self.wavelength)
        valid = freq_abs > 1e9
        wl_nm = np.zeros_like(freq_abs)
        wl_nm[valid] = (c0 / freq_abs[valid]) * 1e9

        spec_smf_cpu, spec_full_cpu, spec_center_cpu = cp.asnumpy(spec_smf), cp.asnumpy(spec_full_incoh), cp.asnumpy(spec_center)

        # =========================
        # 3) 时域诊断
        # =========================
        I_center_t = cp.abs(field_clean[ix0, iy0, :] * self.A_scale)**2
        coupled_field_t = cp.sum(field_clean * fiber_mode, axis=(0, 1)) * self.A_scale
        I_smf_t = cp.abs(coupled_field_t)**2

        I_center_t_cpu, I_smf_t_cpu = cp.asnumpy(I_center_t), cp.asnumpy(I_smf_t)
        fwhm_center, fwhm_smf = self._fwhm_from_intensity(t_cpu, I_center_t_cpu), self._fwhm_from_intensity(t_cpu, I_smf_t_cpu)

        # =========================
        # 4) 保存数据与绘图 (逻辑保持不变)
        # =========================
        diag_npz_path = f"{self.output_dir}/cycle_{cycle:02d}_{label}_DIAG.npz"
        np.savez(diag_npz_path, cycle=cycle, label=label, energy_total=E_total, x_mean_m=x_mean, y_mean_m=y_mean,
                 wx_rms_m=wx_rms, wy_rms_m=wy_rms, t_s=t_cpu, P_t=P_t_cpu, I_center_t=I_center_t_cpu, I_smf_t=I_smf_t_cpu,
                 wl_nm=wl_nm, freq_abs_Hz=freq_abs, spec_smf=spec_smf_cpu, spec_full_incoh=spec_full_cpu,
                 spec_center=spec_center_cpu, fluence=flu_cpu, it_peak=it_peak)

        mask_wl = (wl_nm > self.viz_wl_min_nm) & (wl_nm < self.viz_wl_max_nm) & valid
        mask_t = (t_ps >= self.viz_time_ps_min) & (t_ps <= self.viz_time_ps_max)
        extent_mm = [-self.window_width*500, self.window_width*500, -self.window_width*500, self.window_width*500]

        fig = plt.figure(figsize=(18, 10), constrained_layout=True)
        gs = fig.add_gridspec(2, 3)

        # (1) 光谱
        ax = fig.add_subplot(gs[0, 0])
        ax.plot(wl_nm[mask_wl], self._norm(spec_smf_cpu)[mask_wl], lw=1.8, label="SMF")
        ax.plot(wl_nm[mask_wl], self._norm(spec_full_cpu)[mask_wl], lw=1.0, alpha=0.65, label="Full Beam")
        ax.plot(wl_nm[mask_wl], self._norm(spec_center_cpu)[mask_wl], lw=1.0, alpha=0.65, label="Center")
        ax.set_xlim(self.viz_wl_min_nm, self.viz_wl_max_nm)
        ax.set_title(f"Spectrum - {label} Cycle {cycle}"); ax.legend()

        # (2) 时域
        ax = fig.add_subplot(gs[0, 1])
        ax.plot(t_ps[mask_t], self._norm(I_center_t_cpu)[mask_t], label="Center")
        ax.plot(t_ps[mask_t], self._norm(I_smf_t_cpu)[mask_t], label="SMF")
        ax.set_xlim(self.viz_time_ps_min, self.viz_time_ps_max)
        title = f"Temporal - Peak@t={t_ps[it_peak]:.2f}ps\nFWHM(c)={fwhm_center*1e12:.2f}ps, FWHM(S)={fwhm_smf*1e12:.2f}ps"
        ax.set_title(title); ax.legend()

        # (3) 功率
        ax = fig.add_subplot(gs[0, 2])
        ax.plot(t_ps[mask_t], self._norm(P_t_cpu)[mask_t], color="k")
        ax.set_xlim(self.viz_time_ps_min, self.viz_time_ps_max)
        ax.set_title("Total Power vs Time")

        # (4) Fluence
        ax = fig.add_subplot(gs[1, 0])
        im = ax.imshow(flu_cpu, cmap="inferno", origin="lower", extent=extent_mm)
        ax.set_title(f"Near-field Fluence\nE={E_total:.3e}J, wrms={wx_rms*1e6:.1f}µm")
        fig.colorbar(im, ax=ax)

        # (5) 振幅
        ax = fig.add_subplot(gs[1, 1])
        A_amp_cpu = cp.asnumpy(A_xy_amp)
        im = ax.imshow(A_amp_cpu, cmap="viridis", origin="lower", extent=extent_mm)
        ax.set_title(f"Amplitude |A| at t_peak={t_ps[it_peak]:.2f}ps")
        fig.colorbar(im, ax=ax)

        # (6) 相位
        ax = fig.add_subplot(gs[1, 2])
        thr = 0.10 * (np.max(A_amp_cpu) + 1e-30)
        phase_masked = np.where(A_amp_cpu >= thr, cp.asnumpy(A_xy_phase), np.nan)
        im = ax.imshow(phase_masked, cmap="twilight", origin="lower", extent=extent_mm, vmin=-np.pi, vmax=np.pi)
        ax.set_title("Phase arg(A)")
        fig.colorbar(im, ax=ax)

        plt.savefig(f"{self.output_dir}/cycle_{cycle:02d}_{label}_MULTI_DIAG.png", dpi=140)
        plt.close(fig)

    def run(self):
        A_norm = self.generate_initial_field()
        print(f"\n[Run] ZOOM-IN Simulation (100W, Window=2mm)...")

        seq_forward = [
            (self.d_air_1, 'air'), (self.d_fs, 'fs'),
            (self.d_air_mid, 'air'), (self.d_fs, 'fs'),
            (self.d_air_mid, 'air'), (self.d_fs, 'fs'),
            (self.d_air_mid, 'air'), (self.d_fs, 'fs'),
            (self.d_air_last, 'air')
        ]
        seq_backward = seq_forward[::-1]

        self.save_snapshot(0, A_norm, "Init")

        for i in range(1, self.cycles + 1):
            t0 = time.time()

            # 1. Forward
            for dist, mat in seq_forward:
                A_norm = self.propagate_segment(A_norm, dist, mat)

            # 2. Curved Mirror (线性操作，直接传归一化场)
            A_norm = self.apply_curved_mirror(A_norm)

            # 3. Backward
            for dist, mat in seq_backward:
                A_norm = self.propagate_segment(A_norm, dist, mat)

            # 4. Flat Mirror Loss
            A_norm *= np.sqrt(self.T_per_round)

            cp.get_default_memory_pool().free_all_blocks()
            elapsed = time.time() - t0
            
            # 记录时换算回物理单位的双精度求和
            max_flu = float(cp.max(cp.sum(cp.abs(A_norm).astype(cp.float64)**2, axis=-1))) * self.solver_fs.dt * (self.A_scale**2)
            has_nan = cp.any(cp.isnan(A_norm))
            status_str = "WARNING: NaN" if has_nan else "OK"

            print(f"Cycle {i:<2} | Time: {elapsed:.1f}s | MaxFluence: {max_flu:.2e} J/m^2 | Status: {status_str}")

            if has_nan:
                print("!!! Simulation Stopped due to NaN !!!")
                break

            if i == 1 or i % 2 == 0:
                self.save_snapshot(i, A_norm, "Flat")


if __name__ == "__main__":
    runner = MPC_Final_Runner()
    runner.run()