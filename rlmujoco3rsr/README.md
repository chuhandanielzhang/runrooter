# 3-RSR 三旋翼并联跳跃机器人 — 仿真 / RL / Sim2Real 工作包

> **位置**: `robot_runtime/rlmujoco3rsr/`（从 `Hopper-mujoco-standalone/3RSR_package_2` 打包）
> **上层依赖**: 默认用同级 `../upper_controller_pc/`（可用 `CASE_REPO` 覆盖）
> **策略**: `policies/hop_policy_hwcal.params` 等 | **示例 GIF**: `demos/`

闭环并联腿(3-RSR)+ 三旋翼 的 aerial-legged hopper。本包包含 MuJoCo 模型、
MJX+PPO 强化学习训练、CPU 评估、以及通过 LCM 与 Cao(hopperHFAcase2026)上层
控制器互联做 sim2sim 交叉验证 / sim2real 部署的全部脚本。

---

## 1. 系统总览

```
                         ┌─────────────────────────────────────────┐
   高层 (model-based)     │  Cao ModeE SRB/质心 QP 控制器              │
   或 RL 任务策略          │  run_cao_on_our_model.py  /  RL 任务策略    │
                         └───────────────┬─────────────────────────┘
                                         │  LCM:hopper_cmd_lcmt (q_des,kp,kd)
                                         │      motor_pwm_lcmt   (6ch PWM)
                         ┌───────────────▼─────────────────────────┐
   被控对象 (plant)       │  cao_fake_robot.py  (我们的 MuJoCo 模型,     │
   仿真 或 真机           │  说 case 的 LCM 协议)  /  真实机器人 CAN桥     │
                         └───────────────┬─────────────────────────┘
                                         │  LCM:hopper_data_lcmt (q,qd)
                                         │      hopper_imu_lcmt   (quat,gyro)
                                         ▼
                              lcm-spy 可实时监看所有通道
```

两套控制可互换,都说同一套 LCM 协议:
- **Cao SRB QP**(model-based,已验证):`run_cao_on_our_model.py`
- **RL 策略**(本包训练):`rl_lcm_runner.py`

控制架构(分层 RL):
- **端到端策略** `train_hop.py` → `hop_policy_hwcal.params`(会跳,论文托底)
- **分层底层** `train_leg.py` → `leg_policy.params`(纯腿力/速度/腿长跟踪,配 SRB 高层)
- **空翻** `train_flip.py`(实验)

---

## 2. 模型 (`three_leg_3rsr_closed.xml`)

- **3-RSR 并联腿**:3 个髋电机 `ctrl_joint_1/2/3`,万向节被动副 `cross_pin+lower_pin`,
  足端经 `equality connect` 软约束闭环(MJX 原生闭环,非串联近似)。
- **三旋翼**:`prop1/2/3` 120° 对称;3 舵机绕外向 Z 倾转(矢量推力);反扭矩 `gear=±0.018`,
  自旋方向 +/+/−(三旋翼偏航)。
- **执行器(9)**:`ctrl_motor_1/2/3`(位置,kp=100 kv=3,**力矩限幅 ±25 N·m**,
  ctrlrange `[-90°,+60°]`)+ `servo_motor_1/2/3` + `thrust_1/2/3`。
- **传感器**:髋 q/dq、舵机 q、`foot_touch_sensor`(足端法向力,`sensordata[-1]`)。
- 质量:整机 3.75 kg,腿 0.35 kg。

### 腿部运动学(实测,髋角 → 腿长)
| 髋角 | 足端 z (m) | 腿长 base−foot (m) | 状态 |
|---|---|---|---|
| **−90°**(ctrl 下限) | 0.040 | **0.560** | **最长 / 最伸展 / 足最低** |
| −45° | 0.109 | 0.491 | |
|   0° | 0.241 | 0.359 | home 附近(home=−3.5°) |
| +60°(ctrl 上限) | 0.348 | 0.253 | 最短 / 最蜷缩 |

> 单调关系:**髋角越负 → 腿越长**。−90° 即腿最长、足端最低。

### qpos/qvel 布局 (nq=26, nv=24)
```
base free : qpos[0:7]   qvel[0:6]
leg joints: qpos[7:16]  (hip 在 7,10,13)
servos    : qpos[16:19]
foot free : qpos[19:26] (foot_z = qpos[21]) qvel[18:24]
```

---

## 3. 硬件 (sim2real 标定依据)

| 部件 | 型号 | 仿真建模 |
|---|---|---|
| 关节电机 | AK70 级(±25 N·m) | 位置执行器 + 力矩限幅 |
| 旋翼电机 | T-Motor F40 Pro V KV1950 | 一阶滞后 `Tm=0.04s` |
| 桨叶 | Gemfan 1050 三叶 (10×5) | 反扭系数 `gear=0.018` |
| 舵机 | DS3218MG | slew-rate 6.5 rad/s |
| ESC | 4in1 50A (Holybro Kakute H7) | 限流 → 推力封顶,DR ±15% |

---

## 4. 环境配置

### 4.1 RL / 仿真 (Ubuntu + NVIDIA GPU)
```bash
conda create -n rsr python=3.11 -y && conda activate rsr
pip install mujoco mujoco-mjx "jax[cuda12]" brax imageio
python check_mjx.py          # 验证 MJX/GPU
```

### 4.2 LCM 运行时 + Cao 上层依赖
```bash
# 1) LCM (含 lcm-spy / lcm-logger)
pip install lcm           # 或源码编译;lcm-spy 在 ~/.local/bin/
# 2) case 仓库 (Cao 控制器 + LCM 类型),与本包同级:
git clone git@github.com:chuhandanielzhang/hopperHFAcase2026.git \
    /home/abc/Hopper/hopperHFAcase2026
# 本包脚本通过 sys.path 引用:
#   <case>/hopper_lcm_types/lcm_types   (python LCM 消息类型)
#   <case>/hopper_controller            (modee.* 控制器)
```

> Windows 仅查看模型:`py -m mujoco.viewer --mjcf=three_leg_3rsr_closed.xml`

---

## 5. 怎么启动程序

### 5.1 训练 (GPU)
```bash
# 端到端跳跃策略 (9 维: 3髋+3桨+3舵机, 硬件可观测+DR)
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 \
  python -u train_hop.py --hw --dr --timesteps 80000000 \
  --episode_len 1000 --out hop_policy_hwcal.params

# 分层纯腿底层 (3 维: 纯髋; 指令=竖直力+水平速度+腿长)
XLA_PYTHON_CLIENT_PREALLOCATE=false XLA_PYTHON_CLIENT_MEM_FRACTION=0.6 \
  python -u train_leg.py --hw --dr --timesteps 60000000 \
  --episode_len 500 --out leg_policy.params
```

### 5.2 评估 + GIF (CPU)
```bash
# 端到端: <params> <秒> <vx> <vy> <跳高m> [摩擦]
JAX_PLATFORM_NAME=cpu CUDA_VISIBLE_DEVICES="" \
  python _eval_cpu.py hop_policy_hwcal.params 10 0 0 0.08
# 纯腿底层: <params> <秒> <f_cmd 0-1> <腿长m> <vx> <vy>
JAX_PLATFORM_NAME=cpu CUDA_VISIBLE_DEVICES="" \
  python _eval_leg.py leg_policy.params 8 0.6 0.40 0 0
# 输出 *_10s.gif / *_leg.gif + 力跟踪 / GRF / 漂移统计
```

### 5.3 LCM 运行时管线 (sim2sim 验证 / sim2real 部署)

**三个独立终端**(同一台 PC,LCM 走本机组播):

```bash
# 终端 A — 被控对象:我们的模型当 fake robot
cd Hopper-mujoco-standalone/3RSR_package_2
FAKE_TAU=25 python cao_fake_robot.py --duration-s 30 --drop-start-s 1.5

# 终端 B — 上层控制器 (二选一)
#   (B1) Cao SRB QP:
CAO_L0=0.42 CAO_HOP_H=0.20 CAO_MODE=3 CAO_TAU=25 \
  python run_cao_on_our_model.py
#   (B2) RL 策略:
JAX_PLATFORMS=cpu python rl_lcm_runner.py \
  --params hop_policy_hwcal.params --vx 0 --vy 0 --hop-h 0.08

# 终端 C — 实时监看 LCM 通道
~/.local/bin/lcm-spy
#   关注: hopper_data_lcmt(q,qd) hopper_imu_lcmt(quat,gyro)
#         hopper_cmd_lcmt(q_des) motor_pwm_lcmt(PWM)
```

**启动顺序**:先 A(plant)→ 再 B(controller)→ C(lcm-spy)随时开。
Cao 控制器需 gamepad 事件触发:`cao_fake_robot.py` 已模拟 `Y`(站立)→ `X@0.5s`(PD 腿跳)
→ `A@1.0s`(PWMPD 开桨)。

---

## 6. 关键标定值 (sim ↔ 真机坐标对齐)

`deploy_map.py` / `cao_fake_robot.py`(由 `_cao_calib.py` 数值搜索得到,3 电机偏转矢量对齐到 mm):
```
YAW_SIM_FROM_IMU = -120°      # R_sim<-imu = Rz(-120);  v_imu = Rz(+120) v_sim
JOINT_SIGN       = -1         # sim +q 收腿, 控制器 +q 伸腿
Q0_LCM           = 0.0940     # 控制器侧 home 角
HOME_SIM         = -0.060299  # 仿真 home 髋角
q_lcm  = Q0_LCM - (q_sim - HOME_SIM)
tau_sim= -tau_ff_lcm   (±25 N·m)
```
**陀螺符号修正**:真机 Pixhawk 上 `gyro_y` 取反,仿真侧发布时同样 `gyro_pub[1] = -gyro_pub[1]`,
否则 3D 姿态阻尼被反号。

---

## 7. 文件索引

| 文件 | 说明 |
|---|---|
| `three_leg_3rsr_closed.xml` | 主模型(闭环并联腿 + 三旋翼) |
| `train_hop.py` | 端到端跳跃 RL(9 动作) |
| `train_leg.py` | 分层纯腿底层 RL(3 动作,力/速度/腿长跟踪) |
| `train_flip.py` | 空翻 RL(实验,CAV 奖励 + RSI) |
| `_eval_cpu.py` / `_eval_leg.py` / `_eval_flip.py` | CPU 评估 + GIF |
| `cao_fake_robot.py` | 我们的模型当 LCM fake robot(plant) |
| `run_cao_on_our_model.py` | 启动 Cao ModeE QP 控制器(高层) |
| `rl_lcm_runner.py` | RL 策略的 LCM runner(部署) |
| `deploy_map.py` | sim↔真机坐标 / 动作映射 + 足端正运动学 |
| `_cao_calib.py` / `_align_check.py` / `_att_check.py` | 标定 / 对齐 / 姿态诊断 |
| `check_mjx.py` / `sim2sim_check.py` | 环境 / sim2sim 自检 |
| `*.STL` | 机体 / 腿件 / 足 / 桨叶网格 |

---

## 8. 当前状态

- ✅ `hop_policy_hwcal.params`:端到端,原地跳/定速/跳高/软着陆,TWR<1,真机迁移就绪。
- 🔄 `leg_policy.params`:分层底层 v2(竖直力+水平速度+腿长),训练中。
- ⏳ 接 Cao SRB 高层 → 纯腿底层,联调分层栈。
- ⏳ 真机 sim2real 验证(RA-L 命门)。
