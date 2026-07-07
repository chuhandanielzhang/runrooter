# STANCE Phase 速度收敛机制 + Baseline 参数

> NOTE: 当前 SRB-QP-only 版本已移除 MPC，本文件仅作历史参考。

## 速度收敛机制（两层）

### 1) **MPC 路径**（主要，当 MPC 求解成功时）

**参考速度生成**（混合策略）：
- **早期 STANCE**（compression）：保持当前速度 `v_hat_w[0:2]`（软着陆，不立即刹车）
- **后期 STANCE**（push）：平滑过渡到 `desired_v_xy_w`（速度收敛目标）

```python
# 混合权重（smoothstep）
wv = smoothstep((t - t_comp) / (stance_T - t_comp))
vx_ref = (1 - wv) * v_hat[0] + wv * desired_v_xy_w[0]
vy_ref = (1 - wv) * v_hat[1] + wv * desired_v_xy_w[1]
```

**MPC 求解**：
- 目标：最小化 `||v - v_ref||^2`（权重 `mpc_w_vxy`）
- 约束：摩擦锥 `|f_xy| <= mu * fz`，竖直力 `fz_min <= fz <= fz_max`
- 输出：`f_ref = [fx, fy, fz]`（GRF 参考）

### 2) **Fallback 路径**（MPC 不可行时）

**PD + PI 控制器**：
```python
err_xy = desired_v_xy_w - v_hat_w[0:2]
fx = m * axy_damp * err_x + m * ki_xy * v_int_xy[0]
fy = m * axy_damp * err_y + m * ki_xy * v_int_xy[1]
# 然后限制在摩擦锥内：|f_xy| <= mu * fz
```

**PI 积分器**（slip-gated）：
- 只在 STANCE 且无滑动时积分（`gate` 基于预测的脚端速度）
- 积分上限：`v_int_max`

### 3) **速度估计融合**（STANCE 时）

**腿运动学测量**：
```python
v_meas_b = -(foot_vrel_b + omega × foot_b)  # body frame
v_meas_w = R_wb @ v_meas_b                  # world frame
```

**融合到 v_hat**（complementary filter）：
```python
# XY: 受 v_fuse_vx_scale 控制（默认 1.0 = 完全信任测量）
v_hat[0] = (1 - a_eff_xy) * v_pred[0] + a_eff_xy * v_meas_w[0]
v_hat[1] = (1 - a_eff_xy) * v_pred[1] + a_eff_xy * v_meas_w[1]
# Z: 完全融合（a_eff）
v_hat[2] = (1 - a_eff) * v_pred[2] + a_eff * v_meas_w[2]
```

**滑动检测（slip gate）**：
- 如果预测的脚端速度 `||v_foot_w_pred||` 很大 → `gate` 变小 → 减少融合权重（避免滑移时错误修正）

---

## 所有 Baseline 参数（默认值）

### **速度收敛控制**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `mpc_w_vxy` | `5.0` | MPC 速度跟踪权重（vx/vy 对称） |
| `axy_damp` | `2.0` | Fallback PD 阻尼增益 (N/(m/s)) |
| `ki_xy` | `3.0` | 速度误差积分增益（对称 XY） |
| `v_int_max` | `0.50` | 积分器上限 (m/s) |
| `v_fuse_vx_scale` | `1.0` | 速度融合缩放（XY 对称，1.0 = 完全信任腿运动学测量） |

### **STANCE 时间/参考**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `stance_T` | `0.38` | STANCE 总时长 (s) |
| `stance_min_T` | `0.10` | 最小 STANCE 时长 (s) |
| `soft_land_tc_max_ratio` | `0.60` | 压缩时间上限 = `ratio * stance_T` |

### **摩擦/力约束**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `mu` | `0.3` | 摩擦系数（用于 `|f_xy| <= mu * fz`） |
| `mpc_fz_min` | `40.0` | MPC 最小竖直 GRF (N) |
| `mpc_fz_max` | `220.0` | MPC 最大竖直 GRF (N)（硬编码） |
| `mpc_w_roll` | `20.0` | MPC roll 姿态跟踪权重 |
| `mpc_w_pitch` | `20.0` | MPC pitch 姿态跟踪权重 |

### **MPC 求解器**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `mpc_dt` | `0.02` | MPC 时间步长 (s) |
| `mpc_N` | `12` | MPC 预测步数 |

### **速度估计器**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `use_hopper4_vxy_estimator` | `False` | 启用 Hopper4 式 XY 速度估计（STANCE 用腿运动学，FLIGHT 保持） |
| `_v_hat_lpf_tau` | `0.05` | 速度融合 LPF 时间常数 (s)（内部变量） |

### **高度/起跳**

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `hop_peak_z` | `0.7` | 目标 apex 高度 (m) |
| `hop_z0` | `0.55` | 初始高度 (m) |
| `stance_push_v_to_catch` | `True` | PUSH 阶段强制竖直力下限（确保起跳 vz） |

---

## 速度收敛流程图

```
STANCE 开始
    ↓
[速度估计融合]
v_hat_w ← 融合(IMU预测, 腿运动学测量)
    ↓
[MPC 求解]
v_ref = 混合(v_hat, desired_v)  # 早期保持，后期跟踪
f_ref = MPC.solve(v_ref, ...)   # 最小化 ||v - v_ref||
    ↓
MPC 成功？
    ├─ YES → f_ref = MPC.f0
    └─ NO  → f_ref = PD+PI(desired_v - v_hat)
                ↓
            [摩擦锥限制]
            |f_xy| <= mu * fz
    ↓
[WBC-QP]
GRF = WBC.solve(f_ref, ...)
    ↓
[腿扭矩映射]
tau = J^T * f_leg
    ↓
输出到硬件
```

---

## 调参建议

### **如果速度收敛太慢**：
- 增大 `mpc_w_vxy`（例如 8.0~12.0）
- 增大 `axy_damp`（例如 3.0~5.0）
- 增大 `ki_xy`（例如 5.0~8.0）

### **如果速度收敛振荡**：
- 减小 `ki_xy`（例如 1.0~2.0）
- 减小 `v_int_max`（例如 0.30）
- 检查 `mu` 是否太小（摩擦不够）

### **如果速度估计漂移**：
- 启用 `use_hopper4_vxy_estimator = True`（STANCE 用腿运动学，FLIGHT 保持）
- 检查 `v_fuse_vx_scale`（确保 STANCE 时腿运动学测量被充分信任）

---

## 命令行参数（run_modee.py）

```bash
--mpc-w-vxy 5.0          # MPC 速度跟踪权重
--axy-damp 2.0           # Fallback PD 阻尼
--ki-xy 3.0              # 速度积分增益
--v-int-max 0.50         # 积分器上限
--hopper4-vxy             # 启用 Hopper4 式 XY 速度估计
```




