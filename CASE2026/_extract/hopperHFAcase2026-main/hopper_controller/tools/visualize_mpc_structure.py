#!/usr/bin/env python3
"""
可视化 MPC 矩阵结构
展示 A, B, B_qp 等关键矩阵的稀疏性和结构
"""

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys

# Add parent directory to path to import MPC
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from modee.controllers.mpc import MITCondensedWrenchMPC, MITCondensedWrenchMPCConfig


def main():
    print("=" * 70)
    print("MPC 矩阵结构可视化")
    print("=" * 70)

    # 创建 MPC 实例
    cfg = MITCondensedWrenchMPCConfig(dt=0.02, N=5)  # 用小的 N 便于可视化
    mpc = MITCondensedWrenchMPC(cfg)

    # 示例参数
    m = 3.75  # kg
    g = 9.81
    I_body = np.diag([0.07, 0.07, 0.058])
    r_foot_w = np.array([0.0, 0.0, -0.5])  # 足端在 COM 下方 0.5m
    prop_r_w = np.array([
        [-0.5 * 0.569, np.sqrt(3) * 0.5 * 0.569, 0.0],  # RED arm
        [1.0 * 0.569, 0.0, 0.0],  # GREEN arm
        [-0.5 * 0.569, -np.sqrt(3) * 0.5 * 0.569, 0.0],  # BLUE arm
    ])
    z_w = np.array([0.0, 0.0, 1.0])  # body +Z 在 world 中（假设水平）
    yaw_ref = 0.0
    yaw_rate_ref = 0.0

    # 构建动力学矩阵
    A, B, b = mpc._build_dynamics(
        m=m,
        g=g,
        I_body=I_body,
        r_foot_w=r_foot_w,
        prop_r_w=prop_r_w,
        z_w=z_w,
        yaw_ref=yaw_ref,
        yaw_rate_ref=yaw_rate_ref,
    )

    # Condense
    A_qp, B_qp, xbar = mpc._condense(A=A, B=B, b=b)

    print(f"\n状态维度: nx = {mpc.nx}")
    print(f"控制维度: nu = {mpc.nu}")
    print(f"预测步数: N = {cfg.N}")
    print(f"\nA 矩阵: {A.shape}")
    print(f"B 矩阵: {B.shape}")
    print(f"b 向量: {b.shape}")
    print(f"\nCondensed:")
    print(f"A_qp: {A_qp.shape}  (N*nx × nx)")
    print(f"B_qp: {B_qp.shape}  (N*nx × N*nu)")
    print(f"xbar: {xbar.shape}  (N*nx)")

    # 可视化
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))

    # A 矩阵
    im = axes[0, 0].imshow(A, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
    axes[0, 0].set_title('A 矩阵 (13×13)\n状态转移矩阵')
    axes[0, 0].set_xlabel('x[k]')
    axes[0, 0].set_ylabel('x[k+1]')
    plt.colorbar(im, ax=axes[0, 0])

    # B 矩阵
    im = axes[0, 1].imshow(B, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
    axes[0, 1].set_title('B 矩阵 (13×6)\n控制输入矩阵')
    axes[0, 1].set_xlabel('u[k] = [f(3), t(3)]')
    axes[0, 1].set_ylabel('x[k+1]')
    plt.colorbar(im, ax=axes[0, 1])

    # B_qp 矩阵（下三角块结构）
    im = axes[0, 2].imshow(B_qp, cmap='RdBu', aspect='auto', vmin=-0.1, vmax=0.1)
    axes[0, 2].set_title(f'B_qp 矩阵 ({B_qp.shape[0]}×{B_qp.shape[1]})\nCondensed 控制影响矩阵')
    axes[0, 2].set_xlabel('U = [u[0], u[1], ..., u[N-1]]')
    axes[0, 2].set_ylabel('X = [x[1], x[2], ..., x[N]]')
    plt.colorbar(im, ax=axes[0, 2])

    # B 矩阵的块结构（放大看）
    axes[1, 0].spy(B, markersize=8)
    axes[1, 0].set_title('B 矩阵稀疏结构')
    axes[1, 0].set_xlabel('u[k]')
    axes[1, 0].set_ylabel('x[k+1]')

    # B_qp 矩阵的块结构
    axes[1, 1].spy(B_qp, markersize=2)
    axes[1, 1].set_title('B_qp 矩阵稀疏结构\n(下三角块矩阵)')
    axes[1, 1].set_xlabel('U')
    axes[1, 1].set_ylabel('X')

    # 状态变量说明
    state_names = [
        'px', 'py', 'pz',
        'vx', 'vy', 'vz',
        'roll', 'pitch', 'yaw',
        'wx', 'wy', 'wz',
        'yaw_ref'
    ]
    control_names = ['fx', 'fy', 'fz', 't0', 't1', 't2']

    # 打印矩阵数值（部分）
    print("\n" + "=" * 70)
    print("A 矩阵（部分，前 6×6 块）:")
    print(A[:6, :6])
    print("\nB 矩阵（前 6 行，对应位置/速度）:")
    print(B[:6, :])
    print("\nB_qp 矩阵（前 13 行，对应第一步）:")
    print(B_qp[:13, :6])

    # 状态变量标签图
    ax = axes[1, 2]
    ax.axis('off')
    ax.text(0.1, 0.9, '状态变量 (x):', fontsize=12, weight='bold', transform=ax.transAxes)
    for i, name in enumerate(state_names):
        ax.text(0.1, 0.8 - i*0.05, f'  [{i:2d}] {name}', fontsize=10, transform=ax.transAxes)
    ax.text(0.1, 0.2, '控制变量 (u):', fontsize=12, weight='bold', transform=ax.transAxes)
    for i, name in enumerate(control_names):
        ax.text(0.1, 0.1 - i*0.05, f'  [{i:2d}] {name}', fontsize=10, transform=ax.transAxes)

    plt.tight_layout()
    out_path = Path(__file__).parent.parent / "docs" / "mpc_structure.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=150, bbox_inches='tight')
    print(f"\n📈 可视化图表已保存: {out_path}")

    # 打印推导要点
    print("\n" + "=" * 70)
    print("推导要点总结:")
    print("=" * 70)
    print("""
1. **SRB 模型**: 将机器人简化为单刚体，状态 13 维（位置3 + 速度3 + 姿态3 + 角速度3 + yaw_ref1）

2. **线性化**: 
   - 姿态用 rpy 小角度线性化（yaw 固定为 yaw_ref）
   - 动力学用 Euler 积分离散化

3. **Condensing**:
   - 通过递归代入，将状态序列 X 用初始状态 x[0] 和控制序列 U 表示
   - X = A_qp * x[0] + B_qp * U + xbar
   - B_qp 是下三角块矩阵（u[j] 只影响 x[j+1] 及之后）

4. **QP 形式**:
   - 目标函数: (1/2) * U^T * H * U + g^T * U
   - 约束: 控制上下限、摩擦锥、总推力等
   - 求解后只执行第一步 u[0]，然后重新规划（receding horizon）

5. **与 Hopper4 的区别**:
   - Hopper4: 只优化 GRF (3维)
   - ModeE: 优化 GRF + 3个螺旋桨推力 (6维)，形成 "wrench MPC"
    """)


if __name__ == "__main__":
    try:
        main()
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure you're running from the correct directory")
        sys.exit(1)

