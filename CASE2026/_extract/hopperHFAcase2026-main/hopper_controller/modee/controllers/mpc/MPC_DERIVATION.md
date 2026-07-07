# MPC 数学推导文档

## 1. SRB (Single Rigid Body) 动力学模型

### 1.1 状态变量

```
x = [p(3), v(3), rpy(3), omega(3), yaw_ref(1)]  (13维)
```

- `p`: COM 位置 (world frame, +Z up)
- `v`: COM 速度 (world frame)
- `rpy`: 姿态角 (roll, pitch, yaw, 在 yaw 线性化后的 heading frame)
- `omega`: 角速度 (world frame)
- `yaw_ref`: 参考 yaw（用于线性化）

### 1.2 控制变量

```
u = [f_foot_w(3), t_rotors(3)]  (6维)
```

- `f_foot_w`: 足端 GRF (ground reaction force, world frame, +Z up)
- `t_rotors`: 三个螺旋桨推力 (每个 arm 一个，标量)

### 1.3 动力学方程（离散化，Euler 积分）

#### 位置更新
```
p[k+1] = p[k] + v[k] * dt + 0.5 * a[k] * dt^2
```

#### 速度更新（牛顿第二定律）
```
v[k+1] = v[k] + a[k] * dt

其中：
a[k] = (1/m) * (f_foot_w[k] + z_w * sum(t_rotors[k])) + g_world
```

- `m`: 机器人质量
- `z_w`: body +Z 轴在 world frame 的方向（从当前姿态得到，horizon 内视为常数）
- `g_world = [0, 0, -g]`: 重力加速度（world frame，+Z up）

#### 姿态更新（小角度线性化，yaw 固定）
```
rpy[k+1] = rpy[k] + Rz(-yaw_ref) * omega[k] * dt
```

其中 `Rz(-yaw_ref)` 是 yaw 旋转矩阵的逆，用于将 world frame 的角速度转换到 heading frame。

#### 角速度更新（欧拉方程）
```
omega[k+1] = omega[k] + I_w_inv * tau[k] * dt
```

其中：
- `I_w = Rz(yaw_ref) * I_body * Rz(yaw_ref)^T`: body 惯性矩阵在 world frame 的表示
- `tau[k]`: 总力矩（world frame）

力矩来源：
1. **GRF 力矩**：`tau_grf = r_foot_w × f_foot_w`
   - `r_foot_w`: 足端位置相对于 COM（world frame）

2. **螺旋桨力矩**：`tau_props = sum_i (r_i_w × (z_w * t_i))`
   - `r_i_w`: 第 i 个螺旋桨位置相对于 COM（world frame）
   - 每个螺旋桨推力沿 body +Z 方向，在 world frame 为 `z_w * t_i`

#### yaw_ref 更新
```
yaw_ref[k+1] = yaw_ref[k] + yaw_rate_ref * dt
```

### 1.4 线性化后的状态空间形式

将所有更新写成矩阵形式：

```
x[k+1] = A * x[k] + B * u[k] + b
```

其中：
- `A`: (13×13) 状态转移矩阵
- `B`: (13×6) 控制输入矩阵
- `b`: (13×1) 常数项（主要是重力）

**A 矩阵结构**：
```
A = [I3    dt*I3   0      0       0    ]  (位置)
    [0     I3      0      0       0    ]  (速度)
    [0     0       I3    dt*Rz_m 0    ]  (姿态)
    [0     0       0     I3      0    ]  (角速度)
    [0     0       0     0       1    ]  (yaw_ref)
```

**B 矩阵结构**：
```
B = [0.5*dt^2/m * I3    0.5*dt^2/m * Z]  (位置)
    [dt/m * I3          dt/m * Z      ]  (速度)
    [0                  0              ]  (姿态)
    [dt * I_w_inv * S   dt * I_w_inv * C] (角速度)
    [0                  0              ]  (yaw_ref)
```

其中：
- `Z = [z_w, z_w, z_w]`: (3×3) 每列都是 `z_w`
- `S = skew(r_foot_w)`: (3×3) GRF 力矩的叉乘矩阵
- `C`: (3×3) 螺旋桨力矩的系数矩阵，`C[:,i] = cross(r_i_w, z_w)`

**b 向量**：
```
b = [0.5*dt^2 * g_world, dt * g_world, 0, 0, 0, dt * yaw_rate_ref]^T
```

---

## 2. Condensed QP 形式推导

### 2.1 Full-space MPC 问题

原始 MPC 问题（full-space）：

```
minimize: sum_{k=0}^{N-1} ||x[k+1] - x_ref[k+1]||_W^2 + ||u[k]||_K^2

subject to:
  x[k+1] = A * x[k] + B * u[k] + b  (k=0..N-1)
  u[k] in U_k  (控制约束：摩擦锥、推力上下限等)
```

其中：
- `W`: 状态跟踪权重（对角矩阵）
- `K`: 控制正则化权重（对角矩阵）
- `U_k`: 第 k 步的控制约束集

### 2.2 状态消除（Condensing）

将状态序列用初始状态和控制序列表示：

```
X = [x[1], x[2], ..., x[N]]^T  (N*nx 维)
U = [u[0], u[1], ..., u[N-1]]^T  (N*nu 维)
```

通过递归代入动力学方程，得到：

```
X = A_qp * x[0] + B_qp * U + xbar
```

**推导过程**：

1. **A_qp 矩阵**（状态传播）：
   ```
   x[1] = A * x[0] + B * u[0] + b
   x[2] = A * x[1] + B * u[1] + b
        = A^2 * x[0] + A * B * u[0] + B * u[1] + (A*b + b)
   ...
   x[k] = A^k * x[0] + sum_{j=0}^{k-1} A^{k-1-j} * B * u[j] + (A^{k-1}*b + ... + b)
   ```

   因此：
   ```
   A_qp = [A; A^2; ...; A^N]  (N*nx × nx)
   ```

2. **B_qp 矩阵**（控制影响）：
   ```
   B_qp = [B,     0,     0,   ..., 0    ]
          [A*B,   B,     0,   ..., 0    ]
          [A^2*B, A*B,   B,   ..., 0    ]
          ...
          [A^{N-1}*B, ..., A*B, B]  (N*nx × N*nu)
   ```

   这是一个下三角块矩阵，因为 `u[j]` 只影响 `x[j+1]` 及之后的状态。

3. **xbar 向量**（常数项传播）：
   ```
   xbar = [b; A*b+b; A^2*b+A*b+b; ...]  (N*nx 维)
   ```

### 2.3 Condensed QP 问题

将状态消除后，问题变成只优化控制序列：

```
minimize: (1/2) * U^T * H * U + g^T * U

subject to:
  lU <= U <= uU  (控制上下限)
  A_fr * U <= b_fr  (摩擦锥约束)
  A_sum * U in [l_sum, u_sum]  (总推力约束)
  (可选) A_tau * U in [l_tau, u_tau]  (关节力矩约束)
  (可选) A_rp * (A_qp*x0 + B_qp*U + xbar) in [l_rp, u_rp]  (姿态约束)
```

**Hessian 矩阵 H**：
```
H = 2 * (B_qp^T * W_block * B_qp + K_block)
```

其中：
- `W_block = kron(I_N, W)`: 块对角状态权重矩阵
- `K_block = kron(I_N, K)`: 块对角控制权重矩阵

**梯度向量 g**：
```
g = 2 * B_qp^T * W_block * (A_qp * x[0] + xbar - X_ref)
```

其中 `X_ref` 是参考状态序列堆叠。

---

## 3. 约束处理

### 3.1 摩擦锥约束（Friction Pyramid）

真实摩擦锥是圆锥：
```
|f_xy| <= mu * fz
```

为了在 QP 中表示，用**内接菱形（diamond）**近似：
```
|fx| + |fy| <= mu * fz
```

这可以写成 4 个线性约束：
```
+fx +fy - mu*fz <= 0
+fx -fy - mu*fz <= 0
-fx +fy - mu*fz <= 0
-fx -fy - mu*fz <= 0
```

### 3.2 总推力约束

```
sum(t_rotors) in [thrust_sum_min, thrust_sum_max]
```

这可以写成：
```
A_sum * U in [l_sum, u_sum]
```

其中 `A_sum` 的每一行在对应步的推力变量位置为 1，其他为 0。

### 3.3 关节力矩约束（可选）

如果提供 `A_tau_f`（雅可比转置映射），可以约束：
```
|A_tau_f * f_foot_w| <= tau_cmd_max
```

这确保 MPC 生成的 GRF 可以通过腿关节实现。

---

## 4. 代码实现关键点

### 4.1 `_build_dynamics()` 函数

构建 `A, B, b` 矩阵，对应上述数学推导。

### 4.2 `_condense()` 函数

实现状态消除，计算 `A_qp, B_qp, xbar`。

### 4.3 `solve()` 函数

1. 调用 `_build_dynamics()` 和 `_condense()` 得到 condensed 形式
2. 构建 QP 的 `H, g`（Hessian 和梯度）
3. 构建约束矩阵和上下限
4. 调用 OSQP 求解器求解
5. 返回第一步的控制 `u[0]`（MPC 只执行第一步，然后重新规划）

---

## 5. 为什么用 Condensed 形式？

**优点**：
- 变量数从 `(N*nx + N*nu)` 降到 `N*nu`（状态被消除）
- 约束数也减少（状态约束变成控制约束）
- 求解更快

**缺点**：
- `B_qp` 矩阵是稠密的（下三角块），内存占用 `O(N^2)`
- 当 `r_foot_w` 变化时，需要重新计算 `B_qp`（代码支持 LTV 版本 `_condense_ltv()`）

---

## 6. 与 Hopper4 的区别

- **Hopper4**: 只用 GRF，没有螺旋桨推力作为决策变量
- **ModeE**: GRF + 3 个螺旋桨推力，形成 "wrench MPC"
- 这允许 MPC 同时优化接触力和推力分配，实现更好的姿态/速度权衡

