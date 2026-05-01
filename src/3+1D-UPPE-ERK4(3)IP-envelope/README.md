# 3+1D UPPE 包络求解器（成丝优化版）

## 技术路径

**SVEA（慢变包络近似）+ 复包络 + UPPE 单向传播方程**

- 电场分解：E(t) = Re[A(t)·exp(−iω₀t)]，只求解复包络 A(t)
- 适用脉宽 ≥ 50fs，坐标变换到群速度参考系
- 频域全部使用 FFT 原生顺序（DC 在 index 0），仅输出时 fftshift

## 数值方法

| 组件 | 方法 |
|------|------|
| 线性步 | `exp(Δz·D_op)` 频域精确乘法，平方差公式消除 `kz−k_ref` 灾难性相消 |
| 非线性步 | 4 阶 RK 子步，3D 实空间计算 |
| 框架 | 相互作用绘景 (IP)：`L_step + N_func` 交错，避免算子分裂误差 |
| 误差估计 | ERK4(3) 嵌入式对，`err = max|A4−A3|/max|A4|` (L∞ 范数) |
| 步长控制 | `h_new = h × 0.95 × (tol/err)^0.2`，fail_count 上限安全阀 |
| 阶数验证 | 4 阶收敛：`err(2h)/err(h/2) = 16.00` |

## 物理模型

### 非线性极化
- χ⁽³⁾ Kerr：`γ_kerr = i(ω₀/c)n₂`，`Kerr_term = γ_kerr·I_eff·A`
- **HOKE (n₄)**：`γ_hoke = i(ω₀/c)n₄`，`Kerr_term += γ_hoke·I_eff²·A`（分步乘避免 float32 溢出）
- 拉曼响应：2×Nt 零填充法，因果性严格保持
- 自陡峭算子 `(1+ω/ω₀)` 仅作用于 Kerr 项，等离子体不蓝移

### 等离子体引擎（CUDA RawKernel）
- **输入/输出**：float32 输入（W_ion, W_ava），double 内部计算，float64 输出
- 电离率：Keldysh 公式 + MPI 查表（CPU 预计算, float64）
- 等离子体密度：精确指数积分 `ρ(t+dt) = ρ(t)e^(Γdt) + S·(e^(Γdt)−1)/Γ`
- 中性分子耗尽 + 能量守恒
- gdt 对称截断 ±10，`fabs()` 替代 `abs()`

### 抗混叠与吸收
- 16 阶超高斯频域滤波器（0.85×f_Nyq 处 −1/e）
- 空间边界吸收层（85%–98% 窗口范围，多项式平滑）

## 介质列表

| 介质 | 色散 | n₂ (m²/W) | n₄ (m⁴/W²) | 电离 | 拉曼 | 雪崩 |
|------|------|-----------|-------------|------|------|------|
| FusedSilica | Sellmeier | 3.0×10⁻²⁰ | 0 | Keldysh K=8 | ✅ | ✅ |
| Air | 常数 n₀ | 3.0×10⁻²³·P | 0 | Keldysh K=11 | — | 0 |
| Helium | Cauchy | 4.0×10⁻²⁵·P | −1.0×10⁻³⁸·P² | Keldysh K=17 | — | 0 |
| Neon | Cauchy | 2.0×10⁻²⁴·P | −5.0×10⁻³⁸·P² | Keldysh K=15 | — | 0 |
| Argon | Cauchy | 1.0×10⁻²³·P | −1.0×10⁻³⁶·P² | Keldysh K=11 | — | 0 |
| Krypton | Cauchy | 2.8×10⁻²³·P | −2.5×10⁻³⁶·P² | Keldysh K=10 | — | 0 |
| Xenon | Cauchy | 6.5×10⁻²³·P | −8.0×10⁻³⁶·P² | Keldysh K=8 | — | 0 |

> P = 气压 (atm)。Cauchy 色散：n(λ) = 1 + (A + B/λ² + C/λ⁴)×10⁻⁵×(P/P₀)

## 已知问题

- Helium LUT 含 ~23% 的 NaN 值（σ_K=0 × I¹⁷ float64 溢出），实际仿真中隧穿项兜底，He 的 K=17 极高，MPI 贡献可忽略
- Keldysh 模型在 ~100 GW/cm² 以上 LUT 值接近饱和，定量电离率精度有限

## 适用场景

- 飞秒激光成丝（filamentation），脉宽 ≥ 50fs
- 多通池（MPC）非线性脉冲压缩
- 超连续谱产生
- 稀有气体成丝（Ar, Kr, Xe）

## 文件结构

- `solver_3d.py` — ERK4(3)IP 求解器
- `media_physics.py` — 7 种光学介质 + CUDA 等离子体 Kernel
