# ERK4(3)IP UPPE 非线性光学仿真库

## 概述

ERK4(3)IP UPPE (Unidirectional Pulse Propagation Equation with Embedded Runge-Kutta 4(3) Integrating Factor) 是一个用于模拟非线性光学现象的高级数值求解器库。该库实现了基于Balac & Mahé (2013) 和 Goorjian (1992) 论文的UPPE/FME (Forward Maxwell Equation) 形式，特别适用于超短脉冲传播仿真。

## 核心特性

### 1. 算法优势
- **ERK4(3)-IP方法**: 嵌入式Runge-Kutta 4(3)积分因子法，具有自适应步长控制
- **UPPE/FME形式**: 全波长色散处理，适用于极宽光谱（如5fs脉冲）
- **精确非线性项**: 包含折射率色散修正 n(ω₀)/n(ω)
- **自适应步长**: PID控制器实现智能步长调整，防止步长崩溃

### 2. 物理模型
- **色散处理**: 基于Sellmeier方程的精确材料色散（支持熔融石英、蓝宝石、YAG等）
- **非线性效应**:
  - 瞬时克尔效应
  - 拉曼散射（延迟响应）
  - 自陡峭效应（精确频率依赖）
  - 折射率色散修正
- **损耗模型**: 线性吸收损耗

### 3. 材料支持
- 熔融石英 (fused_silica)
- 蓝宝石 (sapphire)
- YAG晶体
- 自定义材料（通过Sellmeier系数）
- 空气介质（扩展支持）

## 文件结构

### 核心文件
```
src/1D-ERK4(3)IP UPPE/
├── erk43ip_method.py          # 主求解器类
├── visualization.py           # 可视化工具
├── gas-beamchange.py          # 多通腔气体求解器
├── test_*.py                  # 各种测试脚本
├── *.csv                      # 光谱数据文件
└── __pycache__/              # Python缓存
```

### 主要类说明

#### 1. `ERK43IP_UPPE` 类
核心求解器，提供以下功能：

**初始化参数**:
```python
solver = ERK43IP_UPPE(
    material='fused_silica',    # 材料类型
    gamma=None,                 # 非线性系数（可选）
    n2=None,                    # 非线性折射率（可选）
    beam_radius=None,           # 光束半径（可选）
    alpha=0.0,                  # 损耗系数
    center_wavelength=1064e-9,  # 中心波长
    use_raman=True,             # 启用拉曼效应
    f_R=0.18,                   # 拉曼分数
    tau1=12.2e-15,              # 拉曼响应时间常数1
    tau2=32e-15,                # 拉曼响应时间常数2
    use_self_steepening=True    # 启用自陡峭效应
)
```

**主要方法**:
- `propagate()`: 脉冲传播主函数
- `generate_gaussian_pulse()`: 生成高斯脉冲
- `generate_dispersed_pulse()`: 生成带色散的高斯脉冲
- `generate_pulse_from_spectrum()`: 从光谱重建脉冲
- `analyze_spectral_phase()`: 分析光谱相位
- `get_beta2()`: 计算二阶色散系数

#### 2. `SimResultAdapter` 类
结果适配器，用于数据标准化和可视化兼容。

#### 3. `MultipassGasSolver` 类（在gas-beamchange.py中）
扩展求解器，支持：
- 多通腔（MPC）几何
- 空间变化的Gamma系数
- 氮气折射率计算（Peck & Khanna公式）
- 压力-温度修正

#### 4. `NitrogenRefractiveIndex` 类
氮气折射率计算工具。

#### 5. `CavityGeometry` 类
腔体几何参数计算。

## 测试脚本说明

### 基础测试
1. **test_gaussian_pulse.py**: 基础高斯脉冲传播测试
2. **test_gaussian_pulse GDD TOD.py**: 色散脉冲生成与传播测试
3. **test_3d-1d.py**: 纯SPM（自相位调制）测试

### 高级应用
4. **test_gaussian_pulse copy.py**: 级联变光斑传播测试（3段式）
5. **test_gaussian_pulse GDD TOD 1mj.py**: 1mJ能量级色散脉冲测试
6. **test_gaussian_pulse GDD TOD 2mj.py**: 2mJ能量级色散脉冲测试

### 多通腔仿真
7. **test_gaussian_pulse GDD TOD 1mj copy.py**: 双凹型多通腔非线性压缩仿真
8. **test_gaussian_pulse GDD TOD 2mj copy .py**: 2mJ多通腔仿真

### 光谱重建
9. **test_spectrum_pulse-out.py**: 从CSV光谱数据重建脉冲（33次循环变光斑）
10. **test_spectrum_pulse-Origin-out.py**: 基于Origin拟合参数的光谱重建
11. **test_spectrum_pulse-coupling_efficiency-out.py**: 考虑耦合效率的光谱重建

### 气体介质
12. **gas-beamchange.py**: 多通腔气体求解器主程序

## 可视化模块

`visualization.py` 提供丰富的绘图功能：

### 主要绘图函数
1. **plot_results()**: 综合结果绘图（时域、频域演化对比）
2. **plot_supercontinuum()**: 超连续谱产生可视化
3. **plot_pulse_compression()**: 脉冲压缩可视化
4. **plot_modulation_instability()**: 调制不稳定性可视化
5. **plot_soliton_collision()**: 孤子碰撞可视化
6. **draw_parameter_table()**: 仿真参数表生成

### 辅助函数
- `pulse_width()`: 计算脉冲宽度（FWHM）
- `instantaneous_freq()`: 计算瞬时频率
- `analyze_results()`: 分析仿真结果
- `detect_spectral_range()`: 自动检测频谱有效范围

## 物理模型详解

### 1. 色散算子
```python
D(ω) = i[k(ω) - k₀ - k₁·ω] - α/2
```
其中：
- k(ω) = n(ω)·ω/c
- k₀, k₁ 为参考系参数
- α 为损耗系数

### 2. 非线性算子
UPPE增强版非线性项：
```python
N(A) = i·γ·(ω/ω₀)·(n₀/n(ω))·P_NL
```
包含：
- 瞬时克尔效应
- 拉曼卷积响应
- 自陡峭效应修正
- 折射率色散修正

### 3. 拉曼响应
时域响应函数：
```python
h(t) = (τ₁² + τ₂²)/(τ₁·τ₂²)·exp(-t/τ₂)·sin(t/τ₁)
```
频域响应通过FFT计算。

## 使用示例

### 基础使用
```python
from erk43ip_method import ERK43IP_UPPE, SimResultAdapter
from visualization import plot_results

# 初始化求解器
solver = ERK43IP_UPPE(
    material='fused_silica',
    n2=2.7e-20,
    beam_radius=0.4e-3,
    center_wavelength=1064e-9
)

# 生成时间网格
t = np.linspace(-10e-12, 10e-12, 4096)

# 生成高斯脉冲
A_initial = solver.generate_gaussian_pulse(
    t=t,
    pulse_energy=200e-6,
    pulse_fwhm=200e-15
)

# 传播仿真
z_array, A_evolution, omega = solver.propagate(
    A_initial, t, L=0.12,
    tol=1e-5, max_step=1e-2
)

# 可视化
sim_adapter = SimResultAdapter(solver, z_array, t, A_evolution)
plot_results(sim_adapter, A_evolution, save_path="results.png")
```

### 从光谱重建脉冲
```python
# 加载光谱数据
wl_data, int_data = load_spectrum_csv("input_spectrum.csv")

# 重建脉冲
A_initial = solver.generate_pulse_from_spectrum(
    t=t,
    wavelengths_nm=wl_data,
    intensities=int_data,
    pulse_energy=200e-6,
    GDD=23600000 * 1e-30,
    use_jacobian=True,
    interp_kind='cubic'
)
```

### 多通腔仿真
```python
from gas-beamchange import MultipassGasSolver, CavityGeometry

# 定义腔体几何
cavity_geom = CavityGeometry(
    cavity_length=0.4,
    mirror_roc=0.25,
    wavelength=1030e-9
)

# 初始化多通腔求解器
solver = MultipassGasSolver(
    pressure_bar=10.0,
    cavity_geometry=cavity_geom,
    center_wavelength=1030e-9,
    use_raman=False
)

# 运行仿真
total_distance = cavity_params['length'] * cavity_params['passes']
z_hist, A_hist, omega, w_hist = solver.propagate(A0, t, total_distance)
```

## 数据文件

### 光谱数据格式
CSV文件包含两列：
1. 波长（nm）
2. 强度（任意单位，自动归一化）

### 示例数据文件
1. **ClipboardImage1.csv**: 实验测量光谱数据
2. **input_spectrum.csv**: 输入光谱数据
3. **image.csv**: 图像提取的光谱数据

## 性能特点

### 自适应步长控制
- PID控制器：`h_new = h * 0.95 * (tol/error)^0.25`
- 步长崩溃保护机制
- 几何步长限制（基于瑞利长度）

### 数值稳定性
- 精确的色散算子缓存
- 拉曼响应预计算
- 边界条件处理

### 内存优化
- 选择性数据记录
- 数组预分配
- 缓存机制

## 应用领域

1. **超连续谱产生**
2. **脉冲压缩与展宽**
3. **调制不稳定性研究**
4. **孤子动力学**
5. **多通腔非线性压缩**
6. **气体介质非线性光学**
7. **光谱整形与优化**

## 参考文献

1. Balac, S., & Mahé, F. (2013). Embedded Runge-Kutta scheme for step-size control in the interaction picture method.
2. Goorjian, P. M. (1992). Numerical simulation of ultrashort optical pulse propagation.
3. Peck, E. R., & Khanna, B. N. (1966). Dispersion of nitrogen.

## 许可证

本项目代码用于学术和研究目的。具体使用请参考相关论文和引用要求。

---

*最后更新: 2026年3月4日*
