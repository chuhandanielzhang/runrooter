# Hopper 跳跃控制框架（Mode1 + 三旋翼）

> 对应代码：`upper_controller_pc/hopper_controller/modee/core.py`
> 基座：2026-07-20 `2aa050e` 的 Mode1 控制器（虚拟弹簧腿 + hip-torque SRB 姿态 + Raibert 落点），
> 2026-08-01 重建，加入：解耦三旋翼分配器、桨能量补充（PUSH→apex）、PogoX 式飞行速度收敛。
> 2026-08-01 (v2)：stance 能量律默认切换为 **NRC 连续律**（§2b，Lo/Chu/Au ACC 2020），
> Mode1 两段弹簧保留为 `stance_energy_law = "mode1"` 可选回退。

---

## 0. 设计原则

- **优先级**：① 跳跃高度/能量 ② 姿态稳定 ③ 速度收敛。饱和时按此顺序让路（分配器的
  lexicographic `s` 缩放）。
- **正交通道**：腿轴向力（能量）、腿侧向力/hip torque（stance 姿态）、桨 collective（Fz 能量）、
  桨差分（姿态力矩）四个通道物理上互不污染。
- **不要硬切**：所有过渡都是物理量驱动的连续函数——PUSH 弹簧 blend、桨补能的 vz 淡出、
  倾斜的斜坡限速、落地回平的连续预算。唯一的离散事件是物理接触（TD/LO）本身。
- **耦合走物理测量**：腿和桨两个能量源不在 tick 内解代数环，而是通过**apex 回程映射**
  （每跳实测顶点高度 → `E_loss` 自适应）逐跳耦合收敛。

---

## 1. 相位机（事件驱动，无时间表）

```
TD (q_shift ≤ -2cm) ──► COMPRESSION ──► PUSH latch (vz_f 过零去抖) ──► LO (腿伸展阈值)
      ▲                                                                    │
      └───────── descent ◄── apex (vz_up=0) ◄── ascent ◄──────────────────┘
```

- TD/LO 由腿长/关节位移的物理阈值触发，带最小相位时间去抖。
- PUSH latch：世界系竖直速度（LPF 后）过零 + 连续 tick 确认 —— 即"到底了"。
- apex 不是事件，是 `vz_up = 0` 这个物理点，所有随 apex 消失的量都用 vz 连续淡出。

## 2. 腿：能量注入（Hopper4 两段弹簧）

**COMPRESSION**（落地承接）：固定增益世界高度阻抗
\[ f_z = k_z (h_{des} - h_{com}) - b_z v_z, \quad h_{des} = l_0 + h_{hop} \]

**PUSH**（一次性重解弹簧，从底部释放）：在 latch 时刻解
\[ \tfrac12 k_{push} x_0^2 = \underbrace{\tfrac12 m v_{to}^2}_{顶点动能} + \underbrace{m g_{st} x_0}_{提升} + \underbrace{E_{loss}}_{逐跳学习}, \quad v_{to} = \sqrt{2 g_{up} h_{tgt}} \]

- \(x_0 = l_0 - h_{com}\)：底部的世界系高度亏欠（不是腿长压缩量）。
- 力从压缩力连续 blend 过去（`stance_push_blend_tau_s`），不跳变。
- **腿力预算**：\(k_{push} \le f_{z,cap}/x_0\)。封顶后腿存不满 \(E_{need}\) → 差额交给桨（§3）。
- **apex 回程映射**：每跳飞行时间实测顶点高度，误差以增益 γ 折进 \(E_{loss}\)
  （Koditschek–Buehler 离散能量调节，1-D 收缩）。这是腿/桨双能量源的耦合闭环。
- RB 手柄大跳：下一次 PUSH 用 \(h_{tgt} \times\) `big_jump_height_gain`，一跳后自动恢复。

## 2b. NRC 连续能量律（新默认，`stance_energy_law = "nrc"`）

> 来源：Lo, Chu, Au, *A Norm-Regulation-Based Limit Cycle Control of Vertical Hoppers*, ACC 2020。
> 动机：Mode1 的 PUSH latch + 一次性重解弹簧在 latch 时刻产生 \(f_z\) 跳变（log 里 10 ms 内
> 130→430 N），是 tau 抽搐的主因之一。NRC 用**一条连续力律覆盖整个 stance**，能量误差在
> **stance 内部**单调收敛，不需要 latch、不需要 blend、不需要逐跳的 \(E_{loss}\) 映射。

**归一化相平面**（世界系竖直、向上为正）：
\[ \omega = \sqrt{k/m}, \qquad x_1 = (h_{com} - l_0) + \frac{m g_{st}}{k}, \qquad x_2 = \frac{v_z}{\omega} \]

极限环是相平面上的圆 \(\|x\| = r^\*\)。**高度耦合**就在目标半径里：
\[ v_{to} = \sqrt{2 g_{up} h_{hop}}, \qquad r^\* = \Big\| \big( \tfrac{m g_{st}}{k},\; \tfrac{v_{to}}{\omega} \big) \Big\| \]
（\(g_{st}, g_{up}\) 是桨怠速 collective 折算后的有效重力——桨的存在直接改写能量目标。）

**控制律**（论文 NRC-2 平滑版）：
\[ F_{pump} = -2 m k_R \omega\, x_2 (\|x\| - r^\*), \qquad f_z = \mathrm{clip}\big(k(l_0-h_{com}) - b_z v_z + F_{pump},\; 0,\; f_{z,cap}\big) \]

- \(\tfrac{d}{dt}\|x\| = -2 k_R x_2^2 (1 - r^\*/\|x\|)\)：半径误差在 stance 内单调衰减 →
  **一个 stance 内收敛**（1D 仿真：静止零能量起步，第一跳即 6.84 cm / 目标 7 cm，之后死平）。
- \(x_2=0\)（底部）时 \(F_{pump}=0\)：**底部无力尖峰**，能量摊在整个行程注入 → 无 chatter。
- 收敛后 \(F_{pump} \to 0\)，剩纯弹簧——极限环上控制器"消失"。
- TD 时刻若上一跳能量正确则 \(\|x\| \approx r^\*\) → 力从 0 连续起步，TD 也无跳变。
- **桨融合（stance）**：每 tick 腿被 \(f_{z,cap}\) 削掉的需求残差实时进桨 collective：
  \(F_{prop} = \mathrm{clip}(f_{des} - f_{z,cap},\, 0,\, \rho_{max} m g)\)。连续量、纯 Fz 通道。
- **桨融合（flight 上升段）**：对**当前弹道**的 apex 预测误差做纯反馈：
  \[ h_{pred} = h + \frac{v_z^2}{2g}, \qquad F = \mathrm{clip}\Big(k_h\, m g\, \frac{(l_0 + h_{hop}) - h_{pred}}{h_{hop}},\, 0,\, \rho_{max} m g\Big) \cdot \mathrm{clip}\big(\tfrac{v_z}{v_{fade}}, 0, 1\big) \]
  弧线已够高 → 力为 0；不够 → 连续泵，apex 处淡出归零。自限幅，无锁存。
- 大跳：TD 时消费 RB 请求，整个 stance 以 \(h_{hop} \times\) gain 为目标，LO 恢复。

**调参表（就这几个）**：

| 参数 | 默认 | 物理含义 | 怎么调 |
|---|---|---|---|
| `hop_height_m` | 0.07 | apex 目标（唯一高度源头） | 想跳多高改这个 |
| `nrc_k_n_m` | 1400 | 虚拟弹簧刚度 | 触底太深/stance 太长 → 调大；落地太硬 → 调小（深度≈\(v_{td}\sqrt{m/k}\)，时长≈\(\pi\sqrt{m/k}\)≈0.22 s） |
| `nrc_kR` | 400 | 能量泵增益 | 高度收敛慢/跳不到 → 调大；中段力峰太大 → 调小 |
| `nrc_bz` | 8 | 小阻尼（防振动） | 腿振铃 → 调大一点；它耗的能 kR 会自动补回 |
| `prop_energy_max_ratio` | 0.35 | 桨补能上限（×mg） | 桨介入太猛 → 调小 |
| `prop_height_kh` | 1.0 | 飞行段 apex 误差增益 | 0 = 纯腿；跳不够高且腿已饱和 → 调大 |
| `prop_energy_apex_fade_vz` | 0.30 | apex 前淡出窗口 (m/s) | 一般不动 |

日志新列：`nrc_r` / `nrc_r_star`（相平面半径 vs 目标，二者贴合 = 能量在环上）、`prop_energy_fz`
（stance = 腿残差进桨的力；flight = apex 误差反馈力）。

## 3. 桨：能量补充（PUSH → apex，解耦的 Fz 通道）—— Mode1 回退路径

腿被力预算封顶时的能量缺口：
\[ E_{def} = \max\!\big(0,\; E_{need} - \tfrac12 k_{boost} x_0^2\big) \]

桨没有行程限制，力作用在 **PUSH 行程 + 整个上升段**：
\[ F_{prop} = \min\!\Big( \frac{E_{def}}{x_0 + h_{tgt}},\; \rho_{max}\, m g \Big) \]

- **stance PUSH**：collective = 怠速基座 + \(F_{prop}\)。
- **flight 上升**：继续出力，用物理变量 vz 连续淡出：
  \[ F(t) = F_{prop} \cdot \mathrm{clip}\!\big(v_z^{up} / v_{fade},\, 0,\, 1\big) \]
  \(v_z^{up}=0\)（apex）时力恰好归零——滚降跟着剩余上升走，无开关无定时器。
- **下降段**：回到怠速，不加 Fz。
- 只进分配器的 **collective 通道**（纯 collective 零力矩），姿态差分完全不受影响。
- 上升段的额外做功如造成顶点超调，由 §2 的回程映射逐跳吸收。

## 4. 姿态：SO(3) 几何 PD，双执行器分工

误差（Lee et al. CDC'10）：\(e_R = \tfrac12 (R_{des}^T R - R^T R_{des})^\vee\)，参考保测量 yaw。

- **stance**：\(\tau_b = -k_R e_R - k_W \omega\)（可选转速观测器），投影掉点足腿轴不可发力方向，
  由 SRB 闭式解转成世界系接触力：固定 \(f_z\) 竖直支撑，反解水平 \(f_{xy}\)，摩擦锥按比例缩
  （方向保持）。桨差分只补**腿分配后剩下的残差力矩**（HFA Eq.12）。
- **flight**：桨差分是唯一姿态执行器，同一个 \(k_R/k_W\) PD 跟踪 §6 的倾斜参考。
  腿此时做 Raibert 摆腿（§5），互不争抢。

## 5. 速度：估计 + Raibert 落点

**估计（不锁存）**：
- stance：着地足里程计 \(v = -J\dot q\)（世界系），推进段末 N 拍均值作 LO 初值；
- flight：从 LO 初值起 **IMU 世界系加速度逐 tick 外推** \(v \mathrel{+}= (R a_b + g)\,dt\)
  ——桨的刹车会真实反映在速度里，Raibert 和倾斜环消费同一个活的估计（三旋翼串级的标准假设）。

**Raibert 落点**（世界系 FRD）：
\[ p_{xy} = K_v v_{xy} + K_r v_{des},\quad p_z = \sqrt{l_0^2 - \|p_{xy}\|^2},\quad p_b = R^T p_w \]
摆腿是足空间笛卡尔 PD（侧向力正交于腿轴 + 轴向弹簧），限步长。

## 6. 三旋翼飞行速度收敛（PogoX 式，连续）

外环每 tick 闭在活的速度上（Salazar-Cruz CEP'09 / Lee CDC'10 / PogoX ICRA'24）：
\[ a_{des} = k_v (v_{des} - v(t)) \;\rightarrow\; \text{期望推力方向} \;\rightarrow\; R_{des}(\text{保 yaw 倾斜}) \]

- **倾斜参考斜坡限速**（`flight_vel_tilt_slew_dps`）：R_des 全程连续。
- **连续落地预算**：倾角 θ 需要 θ/slew 秒还清 + settle 余量，弹道剩余时间只买得起
  \[ \theta_{budget} = \mathrm{slew} \cdot \max(0,\; t_{td} - t_{settle}) \]
  预算以 slew 速率本身线性收零 → 斜坡参考精确跟得上，落地前 `settle` 秒身体已回平。
  没有 on/off 门。
- **collective 不跟随倾斜**（2026-08-01）：倾斜只改姿态参考，刹车力矩由差分通道给，
  Fz 死守怠速基座（+ §3 的补能）。误差收敛 → 倾角自己收零 → 精确退化回水平参考。

## 7. 分配器：怠速基座差分混合（有界抬升）

每桨命令 \(t_i = T/3 + c + s\,\Delta_i\)：

- **[Fz] collective** \(T\)：全周期 = PWM 1100 怠速（每桨 0.225 N ≈ 0.0103 mg 合计），
  唯一计划内加成是 §3 的能量补充。
- **[τ] 差分** \(\Delta\)：\(M_{[:2]}\Delta = \tau_{xy}\) 最小范数解 + 零和投影（纯力矩）。
  怠速基座下行余量只有 ~0.1 N，所以按三旋翼低怠速惯例：最低桨压在地板 \(t_{min}\)，
  需要力矩的桨从基座**往上抬**（最小抬升 \(c\)），每桨硬顶
  \(t_{ceil} = T/3 + \) `prop_att_thrust_max_each_n`（3 N ≈ PWM 1380）——**不叠 PWM**。
- **饱和**：单标量 \(s\in[0,1]\) 缩整个差分（力矩方向精确），遥测列 `prop_att_scale`
  记录实际交付比例（<1 说明姿态需求被削）。

## 8. "无硬切"清单

| 过渡 | 连续机制 |
|---|---|
| COMP → PUSH 力 | `stance_push_blend_tau_s` 一阶 blend |
| 桨补能 → apex 结束 | \(F \propto \mathrm{clip}(v_z^{up}/v_{fade})\)，vz=0 时自然归零 |
| 倾斜建立/撤销 | 角度空间斜坡限速 |
| 落地前回平 | 连续预算 \(\theta_{budget}(t_{td})\)，以 slew 速率收零 |
| 低 collective 下的速度环 | 侧向力 \(f_z\tan\theta\) 随 \(f_z\to0\) 物理淡出，无门 |
| 姿态饱和 | 单标量 s 缩放，方向不变 |
| 反转地板进出 | 可行性触发 + 迟滞（"auto"），非角度阈值 |

## 9. 调参表

| 参数 | 现值 | 作用 |
|---|---|---|
| `hop_height_m` | 0.07 | 顶点高度目标（能量源头） |
| `stance_kp_z / kd_z` | 1400 / 10 | 落地承接阻抗 |
| `stance_push_blend_tau_s` | 0.01 | PUSH 力上升时间（抖→调大） |
| `mode1_apex_adapt_gamma` | 0.4 | 逐跳能量学习速率 |
| `prop_energy_max_ratio` | 0.35 | 桨补能上限（×mg） |
| `prop_energy_apex_fade_vz` | 0.30 | apex 前滚降的 vz 窗口 |
| `prop_base/stance_base_thrust_ratio` | 0.0103 | = PWM 1100 怠速基座 |
| `prop_att_thrust_max_each_n` | 3.0 | 姿态差分每桨抬升上限（≈PWM 1380） |
| `flight_kv / flight_kr` | 0.14 / 0.09 | Raibert 落点增益 |
| `flight_vel_kv` | 4.0 | 空中速度环带宽（刹车力度） |
| `flight_vel_tilt_max_deg` | 12 | 倾角上限（每度都要落地前还清） |
| `flight_vel_tilt_slew_dps` | 120 | 倾斜斜坡速率（建立与回平同一速率） |
| `flight_level_settle_s` | 0.14 | 落地前回平余量 |
| `flight_kR / flight_kW` | 40 / 6 | 飞行姿态 PD（跟踪倾斜参考） |

## 10. 关键遥测列

| 列 | 含义 |
|---|---|
| `energy_comp_fz` | 腿 PUSH 弹簧超出普通阻抗的能量注入力 (N) |
| `prop_energy_fz` | 桨补能 collective（含飞行淡出后实际值, N） |
| `prop_att_scale` | 姿态差分交付比例（<1 = 被抬升上限削了） |
| `fl_tilt_cmd_deg` / `fl_zb_des_x/y` | PogoX 倾斜命令角与方向 |
| `flight_vel_w` | IMU 外推的飞行速度（不锁存） |
| `tau_out_scale_applied` | 腿力矩比例限幅（持续 <1 = 腿饱和） |
