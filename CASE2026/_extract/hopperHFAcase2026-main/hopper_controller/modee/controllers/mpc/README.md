# MIT-style Model Predictive Control

基于MIT Cheetah论文的MPC控制器实现。

## 架构

```
┌─────────────────────────────────────────────────┐
│                   MPC Controller                │
├─────────────────────────────────────────────────┤
│                                                 │
│  状态: x = [px, py, pz, vx, vy, vz,            │
│             roll, pitch, ωx, ωy]   (10D)       │
│                                                 │
│  输入: u = [fx, fy, fz, T]         (4D)        │
│         接触力(世界系) + 总推力                 │
│                                                 │
├─────────────────────────────────────────────────┤
│                                                 │
│  优化问题:                                      │
│  min  Σ ||x_k - x_ref||²_Q + ||u_k||²_R        │
│                                                 │
│  s.t. x_{k+1} = A_k x_k + B_k u_k + c_k        │
│       |f_xy| ≤ μ·f_z      (摩擦锥)             │
│       f_z ∈ [f_min, f_max] (接触时)            │
│       f = 0               (飞行时)              │
│       T ∈ [T_min, T_max]  (推力限制)           │
│                                                 │
└─────────────────────────────────────────────────┘
```

## 文件结构

```
mpc/
├── __init__.py          # 模块导出
├── mit_mpc.py           # MPC控制器 (QP求解)
├── srb_dynamics.py      # 单刚体动力学模型
└── README.md            # 本文档
```

## 使用方法

### 基本使用

```python
from controllers.mpc import MITMPC, MITMPCConfig

# 创建MPC
cfg = MITMPCConfig(
    dt=0.02,       # 20ms 时间步
    N=15,          # 15步预测 (0.3s)
    mu=1.0,        # 摩擦系数
    fz_max=200.0,  # 最大接触力
    T_max=15.0,    # 最大推力
)
mpc = MITMPC(cfg)

# 当前状态
x0 = np.array([
    0, 0, 0.5,     # 位置 [m]
    0, 0, 0,       # 速度 [m/s]
    0, 0,          # roll, pitch [rad]
    0, 0,          # 角速度 [rad/s]
])

# 参考状态
x_ref = np.array([0, 0, 0.6, 0.3, 0, 0, 0, 0, 0, 0])

# 足端位置 (相对质心, 世界系)
r_foot = np.array([0, 0, -0.5])

# 接触调度 (1=支撑, 0=飞行)
contact_schedule = np.array([1]*5 + [0]*10)

# 求解
sol = mpc.solve(x0, x_ref, r_foot, contact_schedule)

# 获取第一个输入
f_contact = mpc.get_contact_force(sol)  # [fx, fy, fz]
thrust = mpc.get_thrust(sol)            # T
```

### 权重调节

```python
cfg = MITMPCConfig(
    # 状态跟踪权重
    w_pz=50.0,      # 高度跟踪
    w_vx=10.0,      # 前向速度
    w_vy=10.0,      # 侧向速度
    w_vz=20.0,      # 垂直速度
    w_roll=100.0,   # 横滚姿态
    w_pitch=100.0,  # 俯仰姿态
    
    # 输入正则化
    w_f=1e-4,       # 接触力
    w_T=1e-3,       # 推力
)
```

## 动力学模型

### 单刚体 (SRB) 假设

1. **小角度近似**: roll/pitch 接近零
2. **忽略yaw**: 三旋翼无法主动控制yaw
3. **点接触**: 足端单点接触
4. **推力沿体轴**: 总推力沿body z轴

### 状态方程

```
位置: dp/dt = v
速度: dv/dt = (f + T·z_body) / m + g
姿态: dθ/dt = ω
角速度: dω/dt = I⁻¹ · (r × f)
```

## 参考文献

1. MIT Cheetah 3: "Highly Dynamic Quadruped Locomotion via Whole-Body Impulse Control"
2. Convex MPC: "Dynamic Locomotion in the MIT Cheetah 3 Through Convex Model-Predictive Control"
3. Mini Cheetah: "MIT Cheetah 3: Design and Control of a Robust, Dynamic Quadruped Robot"

