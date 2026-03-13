# 代码差异分析报告

## 概述
本报告对比分析 `src/aa` 和 `src/bb` 两个文件夹中的代码差异。两个文件夹都包含 `sim.py` 和 `solver_3d.py` 文件，但存在一些关键差异。

## 文件结构对比
```
src/aa/                    src/bb/
├── sim.py                 ├── sim.py
└── solver_3d.py          └── solver_3d.py
```

## 1. sim.py 文件差异

### 1.1 窗口宽度设置
- **src/aa/sim.py**: `self.window_width = 6.0e-3` (6mm)
- **src/bb/sim.py**: `self.window_width = 2.0e-3` (2mm)

### 1.2 求解器材料参数
- **src/aa/sim.py**: `material='air'` (空气材料)
- **src/bb/sim.py**: `material='fused_silica'` (熔融石英材料)

### 1.3 镜面相位计算方法
**src/aa/sim.py** 使用更复杂的频域依赖方法：
```python
# 引入完整的频域依赖，解决脉冲前沿弯曲和时间漂移
c0 = 299792458.0
omega0 = 2 * np.pi * c0 / self.wavelength
omega_full = omega0 + self.solver_fs.omega
k_omega = omega_full / c0
phase_3d = -1j * k_omega[cp.newaxis, cp.newaxis, :] * R2[:, :, cp.newaxis] / self.R_concave
self.mirror_phase = cp.exp(phase_3d).astype(self.solver_fs.complex_dtype)
```

**src/bb/sim.py** 使用简单的单频近似：
```python
k0 = 2 * np.pi / self.wavelength
self.mirror_phase = cp.exp(-1j * k0 * R2 / self.R_concave).astype(self.solver_fs.complex_dtype)
```

### 1.4 镜面反射应用方法
**src/aa/sim.py** 在频域应用镜面相位：
```python
def apply_curved_mirror(self, field):
    field_w = cp.fft.fft(field, axis=-1)
    field_w_reflected = field_w * self.mirror_phase
    return cp.fft.ifft(field_w_reflected, axis=-1)
```

**src/bb/sim.py** 在时域直接应用：
```python
def apply_curved_mirror(self, field):
    return field * self.mirror_phase[:, :, cp.newaxis]
```

### 1.5 传播序列
**src/aa/sim.py** 使用简化的传播序列：
```python
seq_forward = [
    (self.L_cavity, 'air')  # 整个腔体长度作为空气传播
]
```

**src/bb/sim.py** 使用详细的传播序列：
```python
seq_forward = [
    (self.d_air_1, 'air'), (self.d_fs, 'fs'),
    (self.d_air_mid, 'air'), (self.d_fs, 'fs'),
    (self.d_air_mid, 'air'), (self.d_fs, 'fs'),
    (self.d_air_mid, 'air'), (self.d_fs, 'fs'),
    (self.d_air_last, 'air')
]
```

## 2. solver_3d.py 文件差异

### 2.1 色散算子计算方法
**src/aa/solver_3d.py** 使用数值稳定的计算方法：
```python
# [修复 1: 避免单精度浮点数下的灾难性相消]
# 稳定计算公式: (k_z^2 - k_ref^2) / (k_z + k_ref)
numerator = (k_sq - k_ref_3d**2) - k_perp_sq
kz = cp.sqrt((k_sq - k_perp_sq).astype(self.complex_dtype))
denominator = kz + k_ref_3d
self.D_operator = 1j * (numerator / denominator)
```

**src/bb/solver_3d.py** 使用直接计算方法：
```python
kz_sq = k_sq - k_perp_sq
kz = cp.sqrt(kz_sq.astype(self.complex_dtype))
self.D_operator = 1j * (kz - k_ref[cp.newaxis, cp.newaxis, :])
```

### 2.2 吸收边界条件
**src/aa/solver_3d.py** 使用软吸收边界：
```python
# Space absorber (soft)
s = (R - r_start) / (r_edge - r_start + 1e-30)
s = cp.clip(s, 0.0, 1.0)
k_space = 120.0
damp_s = k_space * (s ** p_space)
self.damp_space = damp_s[:, :, cp.newaxis].astype(self.float_dtype)

# Time absorber (soft)
u = (cp.abs(T) - t_start) / (t_edge - t_start + 1e-30)
u = cp.clip(u, 0.0, 1.0)
k_time = 120.0
damp_t = k_time * (u ** p_time)
self.damp_time = damp_t[cp.newaxis, cp.newaxis, :].astype(self.float_dtype)
```

**src/bb/solver_3d.py** 使用超高斯吸收边界：
```python
X, Y = cp.meshgrid(self.x, self.y, indexing='ij')
R = cp.sqrt(X**2 + Y**2)
R_max = min(float(self.x[-1]), float(self.y[-1])) * 0.90
self.absorber_space = cp.exp(-(R/R_max)**20).astype(self.float_dtype)

T = self.t
T_boundary = float(cp.max(cp.abs(T))) * 0.90
self.absorber_time = cp.exp(-(T/T_boundary)**60).astype(self.float_dtype)
```

### 2.3 非线性算子
**src/aa/solver_3d.py** 关闭了去混叠以节省显存：
```python
# 模式 B: 纯 Kerr 效应 (关闭去混叠，极大节省显存)
else:
    # 直接在原网格上计算，不使用 pad_factor
    Intensity = cp.abs(A_t)**2
    P_nl_t = A_t * Intensity
    P_nl_w = fft(P_nl_t, axis=-1)
```

**src/bb/solver_3d.py** 包含2倍过采样的去混叠：
```python
# 模式 B: 纯 Kerr 效应 (启用 2x 去混叠)
else:
    # 1. 准备工作：获取维度
    Nt = A_t.shape[-1]     # 原始时间点数
    pad_factor = 2         # 2倍过采样 (足以消除三阶非线性混叠)
    Nt_pad = Nt * pad_factor
    
    # 2. 升采样 (Upsampling)
    # ... 详细的频域补零和降采样代码
```

### 2.4 ERK43步进中的吸收应用
**src/aa/solver_3d.py**：
```python
A4_t *= cp.exp(-self.damp_space * h)
#A4_t *= cp.exp(-self.damp_time * h)  # 时间吸收被注释掉
```

**src/bb/solver_3d.py**：
```python
A4_t *= self.absorber_space
A4_t *= self.absorber_time
```

## 3. 关键差异总结

| 差异类别 | src/aa (版本A) | src/bb (版本B) | 影响 |
|---------|---------------|---------------|------|
| **窗口宽度** | 6mm | 2mm | 计算区域大小不同 |
| **材料设置** | 空气 (n2=0.0) | 熔融石英 | 非线性效应不同 |
| **镜面相位** | 频域依赖，包含色散 | 单频近似 | 脉冲前沿精度 |
| **传播序列** | 简化 (单段空气) | 详细 (多段空气+石英) | 物理模型精度 |
| **色散算子** | 数值稳定计算 | 直接计算 | 数值稳定性 |
| **吸收边界** | 软吸收 (渐变) | 超高斯吸收 | 边界反射控制 |
| **非线性算子** | 无去混叠 (节省显存) | 2x去混叠 (抗混叠) | 数值精度与计算成本 |
| **时间吸收** | 注释掉 | 启用 | 时间边界处理 |

## 4. 物理意义分析

### 4.1 版本A (src/aa) 特点：
- **简化模型**：将整个腔体视为空气传播，忽略材料界面
- **数值优化**：采用更稳定的数值计算方法
- **计算效率**：关闭去混叠以节省显存
- **频域精度**：镜面反射考虑频域依赖，提高时间精度

### 4.2 版本B (src/bb) 特点：
- **详细模型**：考虑空气和熔融石英的交替传播
- **抗混叠**：使用2倍过采样消除非线性混叠
- **强吸收边界**：使用超高斯函数作为吸收边界
- **传统方法**：使用更传统的数值计算方法

## 5. 建议

1. **精度要求高时**：建议使用版本B的详细模型和抗混叠技术
2. **计算资源有限时**：建议使用版本A的简化模型和优化算法
3. **脉冲前沿精度重要时**：建议使用版本A的频域依赖镜面相位
4. **数值稳定性重要时**：建议使用版本A的稳定色散算子

两个版本各有优劣，选择取决于具体的仿真需求和计算资源限制。