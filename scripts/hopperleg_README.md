# Hopper 腿 + 桨 操作手册（PC ↔ Jetson ↔ Pixhawk）

> 这是「不用每次问」的速查手册。按顺序照着做即可。
> 角色：**PC**（你现在敲命令的电脑，有线 `enp44s0` = `192.168.1.2`）跑上层 `run_modee.py`；
> **Jetson** 跑底层驱动 + Pixhawk 桥；**Pixhawk** 出 IMU、驱动 4 合一 ESC 转桨。

---

## 0. 正常开机流程（无故障，按 ①→⑥ 顺序照做）

> 一切正常时就这 6 步。任何一步对不上，去对应的 §排错，修好再继续。

```bash
# ───────────────── 在 PC 上 ─────────────────
# ① 修组播路由（Mihomo 每次重启/重连都会抢，每次开机必做一遍）
sudo ip route replace 224.0.0.0/4 dev enp44s0
ip route get 239.255.76.67 | head -1          # 校验：应显示 dev enp44s0 src 192.168.1.2（不对 → §1）

# ② 找 Jetson 当前有线 IP（DHCP 会变，别死记）。复制下面命令拿到 IP：
#    （完整脚本见 §2；拿到后把 <JETSON_IP> 全部替换成它，最近一次是 192.168.1.100）

# ③ 确认 Jetson 三个底层服务都 active
ssh nvidia@<JETSON_IP> 'systemctl is-active canable.service hopper-driver.service px4-bridge.service'
#    期望输出三行 active。任一不是 → 重启它： ssh nvidia@<JETSON_IP> 'sudo systemctl restart <服务名>'（见 §3）

# ④ 体检：PC 4 秒内能收到各通道（imu/hopper_data 在涨即正常）
#    （脚本见 §6；看到 hopper_imu_lcmt + hopper_data_lcmt 有计数就过）

# ⑤ 开上层（新开一个终端，前台常驻，Ctrl+C 退出时会自动归零所有发送数据）
cd /home/abc/Hopper/hopperHFAcase2026/hopper_controller && python3 run_modee.py
#    终端会刷 mode=<...> foot=[x y z] m。再开一个终端看 spy：
#    cd /home/abc/Hopper/hopperHFAcase2026 && bash hopper_lcm_types/scripts/launch_lcm_spy.sh   （见 §5）

# ───────────────── 在 Jetson 手柄上 ─────────────────
# ⑥ 按 X 进 PD 模式 → 再按 A(开桨) / B(全停) / LB / RB（按键含义见 §7）
```

---

## 1. 网络架构 & 坑

- PC 有线网卡 `enp44s0` = `192.168.1.2/24`，直连 Jetson。
- **坑 1：Mihomo 代理抢组播路由。** 每次 Mihomo 重启/重连，`224.0.0.0/4` 会被改到 `dev Mihomo`，于是 PC 收不到任何 LCM。
  - 修复（PC 上跑，需要密码）：
    ```bash
    sudo ip route replace 224.0.0.0/4 dev enp44s0
    ```
  - 检查当前指向哪：
    ```bash
    ip route get 239.255.76.67 | head -1
    # 正确应是: ... dev enp44s0 src 192.168.1.2 ...
    # 被抢时是:  ... dev Mihomo  src 198.18.0.1 ...
    ```
- **坑 2：Jetson 有线 IP 会变（DHCP）。** 重启后可能从 `.1` 变成 `.100` 等。而且 WiFi 路由器也叫 `192.168.1.1`，所以 SSH 到 `.1` 经常连的是 WiFi 那台、ARP 失败。**别死记 IP，用 §2 的方法现查。**

---

## 2. 找 Jetson 当前有线 IP（万能办法）

即使 SSH/ping 不通，只要还能收到组播，就能从组播包里读出 Jetson 真实源 IP：

```bash
cd /home/abc/Hopper/hopperHFAcase2026/hopper_controller && timeout 6 python3 - <<'PY'
import socket, struct, time
s=socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
s.bind(('', 7667))
mreq=struct.pack('4s4s', socket.inet_aton('239.255.76.67'), socket.inet_aton('192.168.1.2'))
s.setsockopt(socket.IPPROTO_IP, socket.IP_ADD_MEMBERSHIP, mreq)
s.settimeout(3); srcs=set(); t0=time.time()
while time.time()-t0<3:
    try: d,a=s.recvfrom(2048); srcs.add(a[0])
    except socket.timeout: break
print('Jetson 组播源 IP:', sorted(srcs) or '(收不到 -> 先修 §1 路由 / 查网线)')
PY
```

拿到 IP 后 SSH：

```bash
ssh nvidia@<JETSON_IP>          # 例: ssh nvidia@192.168.1.100
```

> 收不到任何组播 → 先做 §1 的路由修复；还不行查网线 / Jetson 是否上电。

---

## 3. Jetson 底层服务（systemd）

三个服务，开机相关：

| 服务 | 作用 | 备注 |
|---|---|---|
| `canable.service` | 起 CAN 接口（slcand） | 开机自启 |
| `hopper-driver.service` | AK60 腿电机驱动，发 `hopper_data_lcmt` / `gamepad_lcmt` | `Restart=always`，掉了自动 2s 拉回 |
| `px4-bridge.service` | Pixhawk→`hopper_imu_lcmt`，并把 `motor_pwm_lcmt`→桨 | `Restart=always`，自动找 `/dev/serial/by-id` 稳定串口 |

常用命令（在 Jetson 上，或 `ssh nvidia@<IP> '...'`）：

```bash
# 状态
systemctl is-active canable.service hopper-driver.service px4-bridge.service

# 重启底层腿驱动（最常用：没收到 hopper_data 时）
sudo systemctl restart hopper-driver.service

# 重启桨桥（没收到 hopper_imu / 桨不转时）
sudo systemctl restart px4-bridge.service

# 看日志
sudo journalctl -u hopper-driver.service -n 30 --no-pager
```

> `gamepad_lcmt` **只在你拨动手柄/按键时才发**，静止时看不到是正常的。

---

## 4. PC 上层 `run_modee.py`

**新开一个终端**（前台常驻，看输出，`Ctrl+C` 退出；退出时会自动把所有发送数据归零）：

```bash
cd /home/abc/Hopper/hopperHFAcase2026/hopper_controller
python3 run_modee.py
```

常用参数（按需）：

```bash
python3 run_modee.py --print-hz 5                  # 降打印频率到 5Hz
python3 run_modee.py --pwm-max 1100                 # 正常模式(A键)桨 PWM 上限 1100
python3 run_modee.py --pwm-max 1100 --tau-out-max 1 --thrust-ratio 0.03   # 首次上电更保守
python3 run_modee.py --lcm-url "udpm://239.255.76.67:7667?ttl=255"        # 显式指定 LCM
```

> **`--pwm-max` 只夹「正常模式 / A 键」的桨**（在 `core.step` 里限幅）。
> **LB/RB 切换 loop 里的桨不受 `--pwm-max` 限制**：切换分支直接把 PWM 写成
> `switch_prop_pwm_us = 1600`，不经过限幅。RB 会先把桨斜坡升到该值、腿保持不动，桨到位后腿才动。
> 想改切换桨的值，改 `ModeELCMConfig.switch_prop_pwm_us`，不是 `--pwm-max`。

---

## 5. lcm-spy（GUI，看通道/解码字段）

```bash
cd /home/abc/Hopper/hopperHFAcase2026
bash hopper_lcm_types/scripts/launch_lcm_spy.sh
```

> 该脚本自动 `export LCM_DEFAULT_URL=udpm://239.255.76.67:7667?ttl=255` 并挂上 Java 类型（能解码 q/qd/tau 等字段）。
> 前提：§1 的组播路由已修好，否则 spy 里啥都不出。

---

## 6. 快速体检（命令行，不用 GUI）

PC 上跑，看 4 秒内各通道计数：

```bash
cd /home/abc/Hopper/hopperHFAcase2026/hopper_controller && timeout 6 python3 - <<'PY'
import lcm, time, collections
seen=collections.Counter()
lc=lcm.LCM('udpm://239.255.76.67:7667?ttl=255')
lc.subscribe('.*', lambda ch,data: seen.update([ch]))
t0=time.time()
while time.time()-t0<4.0: lc.handle_timeout(300)
[print(f'  {k}: {v}') for k,v in seen.most_common()] if seen else print('  (空 -> 先修 §1 路由)')
PY
```

**健康基线（大致频率）：**

| 通道 | 来源 | 4s 计数 ≈ | 没有时怎么办 |
|---|---|---|---|
| `hopper_imu_lcmt` | Jetson px4-bridge | ~800 (200Hz) | 重启 `px4-bridge.service` |
| `hopper_data_lcmt` | Jetson hopper-driver | ~4000 (1000Hz) | 重启 `hopper-driver.service` |
| `gamepad_lcmt` | Jetson hopper-driver | 只在动手柄时有 | 拨一下摇杆再看 |
| `hopper_cmd_lcmt` | PC run_modee | ~1700 | 上层没跑 → §4 |
| `motor_pwm_lcmt` | PC run_modee | ~1700 | 上层没跑 → §4 |

---

## 7. 手柄操作（在 Jetson 端的 Xbox 手柄上）

> 前提：上层 `run_modee.py` 在跑，底层 `hopper-driver` active。

| 按键 | 行为 |
|---|---|
| **X** | 进 **PD 模式**。不按则底层只反馈、不施加 `tau_ff`，腿不动（桨走 px4-bridge 不受影响）。 |
| **A** | **桨总开关（control_mode 门控）**。这是唯一能把桨 `control_mode` 切到 ON 的键，px4-bridge 只在 `control_mode==3` 时才真转桨。同时把 PD→PWMPD 用于 SAFE 门控。`pwm_values` 始终是真实指令值。 |
| **B** | **全停**：腿 DAMP + 桨 OFF。 |
| **LB** | **收腿三阶段（不开桨）**：① 1N 恒力收到 -35° → ② 3N 恒力 0.5s → ③ 位置控制到 -30°（kp=12, kd=0.4, 力矩峰值 0.5N）。**整个 LB 过程禁用 SAFE。** |
| **RB** | 同 LB 的三阶段收腿，**额外**：按下 RB **先把桨阶梯斜坡升到 `switch_prop_pwm_us`(=1600)，腿在此期间保持不动(零力矩)**；桨升到 1600 后**腿才开始**阶段 1 收腿。桨在阶段 1/2 + 进阶段 3 的头 1 秒维持 **1600**；**1 秒后交接给「正常 hopper 旋翼参数」(ModeE 的 pwm),不再归零**——从固定 spin-up 平滑过渡到正常桨控制。**但 RB 不会自己切 `control_mode`** —— 桨真转与否由 **A 键**决定:想让 RB 的桨转,必须**先按 A**;A 没开时 `pwm_values` 显示真实值但 `control_mode` 仍 OFF、桨不转。**整个 RB 过程禁用 SAFE。** |

> 注意：阶段 1/2 是力矩控制（不受位置 limit）；阶段 3 位置控制时**也不受 limit**，只有 0.5N 力矩封顶。
> RB 的「先升桨到 1600 再动腿」靠 `_switch_prop_spun_up` 门控:只有当实际(限速后)桨 PWM `prev_pwm_us` 达到 `switch_prop_pwm_us` 才放行腿部阶段 1。
> **LB/RB 切换里的桨也不受上层 `--pwm-max` 限制**——切换分支直接写 `switch_prop_pwm_us`(=1600)，
> 走独立路径，不经过 `core.step` 的 PWM 限幅。`--pwm-max` 只影响正常模式(A 键)的桨。

---

## 8. 故障速查

| 现象 | 原因 | 解决 |
|---|---|---|
| PC 收不到任何 LCM（体检全空） | Mihomo 抢了组播路由 | `sudo ip route replace 224.0.0.0/4 dev enp44s0` |
| SSH `192.168.1.1` 连不上 / `No route to host` | Jetson DHCP 换 IP 了 + WiFi 也占 .1 | §2 现查真实 IP，用新 IP SSH |
| 收得到 `hopper_imu` 但没 `hopper_data` | hopper-driver 掉了 | `sudo systemctl restart hopper-driver.service` |
| 按 RB/LB 腿不动 | 没按 X 进 PD 模式 / 上层没跑 | 按 X；确认 `run_modee.py` 在跑 |
| 桨不转 | px4-bridge 掉了 / 没按 A（普通模式） / Pixhawk 串口变了 | 重启 `px4-bridge.service`（脚本自动找 by-id 串口） |
| 看不到 `gamepad_lcmt` | 手柄静止时本来就不发 | 拨一下摇杆 |
| **桨/姿态"很割裂、卡顿"，`motor_pwm` 数值长时间一模一样不变** | **`hopper_imu` 静默掉线**：FC 的 ATTITUDE 流停了，控制器拿**冻结的旧姿态**算 → 桨输出冻结、偶尔跳。常发生在桥重启后那次 `SET_MESSAGE_INTERVAL` 请求丢了 | 体检看 `hopper_imu_lcmt` 是否=0；现已加**IMU 看门狗**（>1.5s 没 ATTITUDE 自动重发请求，自愈）。仍为 0 就 `sudo systemctl restart px4-bridge.service` |
| 腿方向/角度反 | 关节符号/偏置 | `hopper_hardware.cpp` offset = `+1.4835f`（+85° 全伸） |
| **腿数据在发但 q 冻结 / q 恒为 ≈85°** | **电机 CAN 不回话**（motor_pos=0 → q=offset=85°）。常见于 CANable **bus-off** 锁死（见 §8.2） | 见 §8.2 |
| **桨没信号 / ESC 反复奏乐(开机自检音)** | **跑了多个 `run_modee.py`** 抢同一通道，桨指令开关乱跳 → ESC 反复重启；进程互相打架还会卡死。或 motor_pwm 没到 Jetson（路由/单实例问题） | 见 §8.3 |
| **桨完全无信号 / 不转，连 ESC 自检音都没有** | **飞控 `PWM_AUX_*` 参数被写坏**：AUX1/2/3 的 min/max/disarmed 变成 0 → 飞控恒输出 0us（无脉冲）。常因用 pymavlink 设 **INT32 参数时编码错误**（直接传浮点值而非整数位） | 见 §8.4 |

### §8.4 桨完全无信号 / 不转 —— 飞控 PWM_AUX 参数被写坏（INT32 编码陷阱）

**症状**：桨完全不转，**连 ESC 自检音都没有**（不同于 §8.3 的"反复奏乐"）。`actuator_test` 命令被接受、无报错，但桨纹丝不动。

**根因**：飞控 `pwm_out` 里 AUX1/2/3 的 `min/max/disarmed` 变成 **0** → 飞控给桨**恒输出 0us（无脉冲）** → ESC 收不到任何信号。最隐蔽的来源是**用 pymavlink 设 INT32 参数时编码错误**：

- PX4 的 `PWM_AUX_FUNC* / MIN* / MAX* / DIS*` 都是 **INT32** 参数。
- MAVLink `PARAM_SET` 把值放在一个 4 字节 `param_value`（float）字段里；INT32 参数要求把**整数的字节**塞进去，PX4 收到后按 INT32 `memcpy` 解释。
- 若直接 `param_set_send(name, 1000.0, INT32)`，pymavlink 把 `1000.0` 的**浮点位**（`0x447A0000`）当字节发出 → PX4 当 int 存成 `1148846080`（天文数字）→ mixer 认为超界/无效 → min/max 当 0 → 0us。

**诊断**：
```bash
# 看飞控实际输出范围（关键）；若 AUX 通道 min:0 max:0 → 命中本故障
#   nsh: pwm_out status   →  Channel 0: func:101 ... min:0  max:0   ← 坏
#   正常应是                Channel 0: func:101 ... min:1000 max:2000 value:1000
# 看参数存储的原始整数；param show 把存储 int 直接打印
#   nsh: param show PWM_AUX_MIN1   →  1148846080(坏) / 1000(正常)
```

**修复**：用**正确的 INT32 编码**重写参数（把整数位打包进 float 字段），再 `param save`：
```python
import struct
def set_int(m, name, ival):
    fval = struct.unpack('<f', struct.pack('<i', int(ival)))[0]  # int bits -> float field
    m.mav.param_set_send(m.target_system, m.target_component,
                         name.encode(), fval, mavutil.mavlink.MAV_PARAM_TYPE_INT32)
# PWM_AUX_FUNC1/2/3=101/102/103, MIN/DIS=1000, MAX=2000 ; 之后 nsh: param save
```
修好后 `pwm_out status` 的 AUX 通道应显示 `min:1000 max:2000 value:1000`，ESC 自检音停、桨可被 `actuator_test` 驱动。

### §8.3 桨没信号 / ESC 反复奏乐 —— 只能跑一个 run_modee

**症状**：桨不转或时转时停，ESC 反复播放开机自检音乐；`px4-bridge` 收不到稳定指令。

**根因**：PC 上**同时跑了多个 `run_modee.py`**，它们往同一个 `motor_pwm_lcmt` 抢着发 → `control_mode` 在 ON/OFF 间乱跳 → ESC 反复解除武装/重新初始化（奏乐）。进程互相打架后还可能**卡死**（`kill` 杀不掉，要 `kill -9`），导致干脆不发数据、桨彻底没信号。

**铁律：任何时候只能有一个 `run_modee.py`。**

**排查 / 修复**：
```bash
# PC：看有几个 run_modee（应只有 1 个）
pgrep -af "[r]un_modee"
# 多于 1 个 → 全部杀掉重开一个
pkill -9 -f "[r]un_modee.py"        # 或对每个 PID: kill -9 <PID>
cd /home/abc/Hopper/hopperHFAcase2026/hopper_controller && python3 run_modee.py --pwm-max 1100
```

**验证 Jetson 收到了 PC 的指令**（路由也要先修好，§1）：
```bash
ssh nvidia@<JETSON_IP> 'timeout 5 python3 -c "
import lcm,time,collections; seen=collections.Counter()
lc=lcm.LCM(\"udpm://239.255.76.67:7667?ttl=255\"); lc.subscribe(\".*\", lambda c,d: seen.update([c]))
t0=time.time()
while time.time()-t0<3: lc.handle_timeout(300)
print(dict(seen))"'
# 应看到 hopper_cmd_lcmt 和 motor_pwm_lcmt（来自 PC）。没有 → run_modee 没发/路由没修/启动时路由还被 Mihomo 抢着(重启 run_modee)。
```

> 注意：`run_modee.py` 的组播发送网卡在**启动时**按路由决定。若它在路由被 Mihomo 抢时启动，套接字会绑错网卡，即使之后修了路由也发不到 Jetson —— **先修路由(§1)再启动 run_modee**。

### §8.2 腿"没连接" / q 冻结在 85° —— CAN bus-off 排查

**症状**：`hopper_data_lcmt` 照常 1000Hz 在发，但 `q` 数值**一动不动**，且常常恒为 **≈85°**（=offset，说明 motor_pos 读不到、取了默认 0）。

**诊断**（SSH 到 Jetson）：看 CAN 收包是否在涨——

```bash
ssh nvidia@<JETSON_IP> 'r1=$(cat /sys/class/net/can0/statistics/rx_packets); sleep 1; \
r2=$(cat /sys/class/net/can0/statistics/rx_packets); echo "rx 1s 差 $((r2-r1))  (=0 表示电机零回复)"'
```

- `rx` 差 **=0** 但 `tx` 在涨 → 驱动在喊话、电机不回 → CAN 物理/总线问题。
- 抓帧确认只有发没有回：`candump -n 10 can0`（只看到 ID 001/002/003 的恒定命令帧 = 都是自己发的回显）。

**最常见根因 & 修复（按顺序试）**：
1. **CANable bus-off 锁死**（最常见，尤其改过供电/地之后）：失去共地或总线异常时 CANable 狂发不被 ACK → 进 bus-off，**重启驱动清不掉**，必须**重置 CAN 接口**：
   ```bash
   ssh nvidia@<JETSON_IP> 'sudo systemctl restart canable.service && sleep 2 && sudo systemctl restart hopper-driver.service'
   ```
   重置后 `rx` 立刻开始涨 ≈ `tx`、`q` 变活。
2. **共地**：Jetson 与电机不同电源时，CANable GND 必须和电机/6S 负极实接（万用表量 ≈ 0Ω）。
3. **CANH/CANL 接线松动 / 终端电阻**：动过线后检查 H↔H、L↔L 是否都通，两端 120Ω。
4. **电机没全上电**：确认每个电机控制器指示灯都亮。
| **桨一启动整个就"断连"** | **Jetson 掉电重启（brownout）**：ESC 浪涌电流把供电拽到欠压 → Jetson 复位。重启后 hopper-driver（disabled）不自启，故 `hopper_data` 消失、腿不动；imu 还在因为 px4-bridge 自启。 | 见下方「§8.1 桨 brownout」 |

### §8.1 桨启动导致 Jetson 重启（brownout）— 重要

**诊断**：怀疑掉电重启时，SSH 到 Jetson 看 uptime / last reboot：

```bash
ssh nvidia@<JETSON_IP> 'uptime -p; uptime -s; last reboot | head -3'
```

若 `uptime` 只有几分钟、且 `last reboot` 时间正好是你启动桨的时刻 → 就是 brownout 重启（不是网络/软件问题）。
dmesg 里会看到一整串 USB 设备（手柄 F710、HDMI 音频、蓝牙…）在同一秒重新枚举（冷启动指纹）：

```bash
ssh nvidia@<JETSON_IP> 'sudo dmesg -T | grep -iE "new .* device|input:" | tail -20'
```

**根因**：桨/ESC 与 Jetson 共用供电时，桨启动浪涌把电压拽到 Jetson 欠压阈值以下 → 复位。

**修复（按优先级）**：
1. **电源隔离/扩容**：ESC 与 Jetson 不共用同一路供电，或给 Jetson 单独一路足电流余量的 5V/9V BEC。
2. **加缓冲电容**：Jetson 供电输入端 + ESC 输入端各并大容量电解电容（几百~上千 µF）吸收浪涌。
3. **桨 soft-start**：启动别直接 1200，斜坡缓升（可在 `px4_bridge.py` 给桨加上升斜坡）。

**重启后恢复**：`hopper-driver` 默认 `disabled`（开机不自启，电机安全考虑），brownout 重启后需手动拉起：

```bash
ssh nvidia@<JETSON_IP> 'sudo systemctl restart hopper-driver.service'
```

**桨 soft-start（已实现，降浪涌）**：桨 PWM 硬跳变（1000→1200 瞬间）会一次性拉满 ESC 启动电流 → 拽塌电压。
`_publish_motor_pwm` 现在对**上升**的桨 PWM 限速 `prop_slew_up_us_per_s`（默认 400 us/s，即 1000→1200 约 0.5s 爬完）；
**下降/停止/disarm 立即生效**。想更猛地降浪涌就把这个值调小；设 `1e12` 关掉斜坡。改后重启 `run_modee.py` 生效。

---

## 9. 关键路径备忘

- PC 上层：`/home/abc/Hopper/hopperHFAcase2026/hopper_controller/run_modee.py`
- 控制逻辑：`/home/abc/Hopper/hopperHFAcase2026/hopper_controller/modee/lcm_controller.py`
- spy 脚本：`/home/abc/Hopper/hopperHFAcase2026/hopper_lcm_types/scripts/launch_lcm_spy.sh`
- Jetson 底层：`/home/nvidia/Hopper_srbRL/build/hopper_driver`
- Jetson 桨桥：`/home/nvidia/Hopper_srbRL/pixhawk/px4_bridge.py`
- systemd unit：`/etc/systemd/system/{canable,hopper-driver,px4-bridge}.service`（在 Jetson 上）
- LCM URL：`udpm://239.255.76.67:7667?ttl=255`

---

## 10. 桨配置（Y 120° 三旋翼，2026-06 最终标定）

**实机标定**（2026-06-25）：控制器/机体系是 **FRD（x 前 / y 右 / z 下）**。电机1 在 **−Y**；电机2=**右前**；电机3=**右后**。

控制器帧（FRD）下的几何：

| 电机 (PWM 通道) | 位置 | 方位 | 几何角 |
|---|---|---|---|
| 电机 1（pwm[1]） | (0, −Y) | **−Y 侧** | - |
| 电机 2（pwm[2]） | (+X, +Y) | **右前** | - |
| 电机 3（pwm[3]） | (−X, +Y) | **右后** | - |

> 分配是 `core.py` 里用 `prop_positions_b` 现算的 **r × ẑ 力矩映射**（无硬编码矩阵；不建模偏航 yaw）。
> 映射 `prop_pwm_idx_per_arm`：arm i → 电机 i+1。
>
> **上电前手扶验证**：桨低权限，根据右侧三桨几何检查 `r×ẑ` 力矩方向；如果 roll/pitch 反向，优先检查 `prop_positions_b`，不要再换 PWM 通道。

### 改桨转向（正转/反转）

**软件 PWM 改不了转向**，只能改转速。转向要在 **ESC（DShot）** 层改：

| 方法 | 说明 |
|---|---|
| **PX4 NSH（推荐）** | 停 `px4-bridge` 后：`dshot reverse -m 1` + `dshot save -m 1`（电机2 把 `1` 换成 `2`）。脚本：`hopperMOBILEmanipulation/pixhawk/fc_reverse_motors.py DEV --motors 1,2` |
| **QGroundControl** | Actuators → Testing → **Set Spin Direction**（DShot ESC） |
| **换线** | ESC 上任意两根电机线对调（定距桨常用办法） |

> **注意**：定距桨反转后 **升力仍朝上**，只是反扭矩（偏航 yaw）反了。ModeE **不建模**桨 yaw 反扭矩，所以改转向**不会**修 roll/pitch 姿态分配问题；那是 `prop_positions_b` 几何的事。
