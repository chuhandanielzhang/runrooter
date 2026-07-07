# PAC-NMPC Hopping Demo (isolated)

Joseph Moore (JHU) 风格的 **PAC-NMPC**(采样式随机 NMPC + PAC 置信上界,
Polevoy/Kobilarov/Moore, RA-L 2023, arXiv:2210.08092)应用到 Pro-OMEGA2 跳跃机器人,
在 MuJoCo 中闭环运行。

**本文件夹与 Jetson/PC 运行栈完全隔离**:不 import、不修改
`robot_runtime/upper_controller_pc`、`lower_driver_jetson` 等任何在跑的文件。
需要的约定/几何/电机表全部以"快照拷贝"方式放在 `hopper_pac/` 内。

## 接口与坐标 = Jetson 栈

`hopper_pac/sim.py` 复刻了 `mujoco_lcm_fake_robot.py`(CASE 仿真 parity 层)的
IO 语义,serial plant 路径:

| 信号 | 约定 |
|---|---|
| `q/qd` (3,) | LCM 关节 [roll, pitch, shift],serial: `q_sign=1, q_offset=0` |
| `tau_ff` (3,) | 关节前馈:roll/pitch 力矩 (Nm, ±27),shift 为直线力 (N, ±2500) |
| `quat` | wxyz, body→world |
| `gyro` | 体坐标角速度 |
| `acc` | −(比力):静止 `[0,0,-9.81]`,自由落体 `[0,0,0]` |
| `pwm_us` (6,) | ESC PWM;实测 1950KV 电机表 → 推力,沿体 +Z 施加在臂端 |
| 桨臂映射 | RED/GREEN/BLUE,GREEN 朝 +X;PWM 通道 per arm = (2,1,3) |
| 机身系 | +X 前, +Y 左, +Z 上;世界 +Z 上 |

模型 = `model/hopper_serial.xml`(CASE 同款,连 mesh 一起拷贝)。

## 结构

```
hopper_pac/
  conventions.py   # 所有与运行栈一致的常量/坐标(快照,注明出处行号)
  leg.py           # serial 腿 FK/IK/Jacobian/GRF→tau (核对 core.py _serial_leg_fk_jac)
  motor_table.py   # PWM<->推力表 (快照自 motor_utils.py)
  sim.py           # MuJoCo plant, step(tau, pwm)->SensorData;真实不确定性注入在这里
  srb_rollout.py   # 向量化随机 SRB rollout:每个样本自己触发 touchdown(多峰接触时序)
  pac_nmpc.py      # PAC-NMPC:采样分布 -> 多世界评估 -> PAC 上置信界 -> elite 更新
  controller.py    # 50Hz PAC-NMPC 外环 + 500Hz 内环(GRF→tau / Raibert / 桨姿态PD)
run_demo.py        # 闭环 demo / Monte-Carlo / push / 录像
plot_results.py    # 两张主图:episode 时序 + certificate 校核
```

分层与真机一致:外环出 wrench(50 Hz,= 运行栈 `_mpc_f_ref_cache` 那条 MPC→QP 通路的位置),
内环 500 Hz 执行(对应 WBC-QP 层,含摩擦锥 + fz 下限的可行性保护)。

## 跑法

```bash
cd pac_nmpc_demo
python3 run_demo.py                      # PAC-NMPC,名义 plant
python3 run_demo.py --deterministic      # 基线:同一采样器,关掉不确定性
python3 run_demo.py --hard               # 恶劣真机参数(低摩擦/地面偏移/载荷/弱桨)
python3 run_demo.py --trials 10          # Monte-Carlo 随机 plant -> 成功率
python3 run_demo.py --push 25            # 每 2.5s 一次 25N×0.1s 随机方向推
python3 run_demo.py --video out/x.mp4    # 录像
python3 plot_results.py --trials 8       # 生成 out/timeseries.png, out/bound_check.png
```

## 当前状态(2026-07-04)

- 闭环稳定跳:名义 plant 4/4 seed,10 s 内 ~16 hops,|roll|,|pitch| < 2°。
- Monte-Carlo 随机 plant(μ∈[0.1,0.9]、地面高度 ±3cm、载荷 ≤1kg、推力 ±10%、陀螺噪声):
  PAC 10/10,确定性基线也 10/10 —— **当前扰动分布下两者尚未拉开差距**,
  要展示 PAC 的差异化需要更极端的多峰分布(如双峰地面高度/冰面摩擦)+ push 组合。
- push:25 N 可恢复;40 N 会摔(内环姿态权限所限)。
- 求解:纯 NumPy CPU,外环单次 ~50 ms(=20Hz 实时);500Hz tick 均值 5ms。
  搬 GPU/torch 采样是下一步(Moore 原文即 GPU 并行)。

## 与"投名状"的对应关系

- 方法主体 = PROPS 式采样策略分布优化 + 逐次最小化 PAC 上置信界(`pac_nmpc.py`),
  证书 = 每次 solve 输出的 `bound`(见 `out/timeseries.png` 第三栏)。
- 新场景 = 混合接触:rollout 里每个采样世界自己触发 touchdown/liftoff,
  接触时序不确定性天然多峰 —— 这是 Moore 原文(光滑固定翼/车)没有的疆域。
- 下一步(按优先级):
  1. torch/GPU 化 rollout(3070 Ti),把 S×M 提到 10^4 量级,bound 收紧;
  2. 用真机 CSV log 拟合 `RolloutUncertainty`(接触时序方差/滑移/推力误差);
  3. 设计"多峰专属"场景(双峰地面、冰面)拉开 PAC vs 确定性的消融差距;
  4. 真机部署:外环替换 `f_ref` 通路,内环沿用现有 WBC-QP(不改现有文件,新开分支)。
