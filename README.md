# Hopper robot runtime (collected runnable files)

This folder gathers the **actually-running** files of the Hopper stack (lower layer +
upper layer + IMU bridge + bring-up scripts). It is a curated copy of the live sources;
build artifacts, logs, caches, and analysis/plot scripts were left out.

## Architecture

```
   PC (this machine)                         Jetson (nvidia@192.168.1.100)
 ┌─────────────────────┐   LCM (multicast)  ┌──────────────────────────────┐
 │ run_modee.py         │ <───────────────> │ hopper_driver (legs/CAN/pad)  │  hopper-driver.service
 │  ModeECore + WBC-QP  │  hopper_data_lcmt  │   src/hardware/*.cpp          │
 │  propeller/leg cmd    │  hopper_imu_lcmt   │                              │
 │                       │  gamepad_lcmt      │ px4_bridge.py (USB MAVLink)  │  px4-bridge.service
 │                       │  motor_pwm_lcmt    │   props=USB  IMU=DDS/TELEM2  │
 └─────────────────────┘                    │  (split: IMU 500, prop 150)  │
                                             │ canable (can0 bring-up)      │  canable.service
                                             └──────────────────────────────┘
```

- **Legs**: 3× AK60 over SocketCAN (`can0`).
- **IMU + propellers**: on the Pixhawk (PX4). As of 2026-07-01 this uses the **SPLIT path**:
  - **IMU** over **DDS/TELEM2** @ **500 Hz**: `xrce-agent.service` (Micro-XRCE-DDS Agent on
    `/dev/ttyTHS1`) + `px4-dds-bridge.service` (`px4_dds_bridge.py --mount-rpy 0,90,0
    --publish-hz 500`) publishes `hopper_imu_lcmt` from PX4 DDS topics. Dedicated UART =
    contention-free. (IMU mount/coordinate unchanged from before — only comms changed.)
  - **Props** over **USB MAVLink**: `px4-bridge.service` runs `px4_bridge.py --no-imu`
    (props ONLY), streaming `MAV_CMD_DO_SET_ACTUATOR` from `motor_pwm_lcmt`. The PC still
    publishes `motor_pwm_lcmt` at the full **500 Hz** control rate, but the MAVLink
    re-stream is capped at **150 Hz** (`--prop-rate 150`): PX4 answers EVERY
    `DO_SET_ACTUATOR` with a `COMMAND_ACK` on this same USB link, so 400 Hz+ floods the link
    with ACKs, PX4 throttles/drops, and the actuator outputs FREEZE ("pwm stuck"). This cap
    is set by the ACK return traffic, NOT by whether the IMU shares the link — 150 Hz
    (6.7 ms) is still far faster than the prop time constant (~40 ms).
  - **Important**: `px4-bridge` MUST keep `--no-imu` so it does NOT also publish
    `hopper_imu_lcmt` — two publishers on one channel fight. To collapse back onto a single
    USB link (props+IMU), drop `--no-imu` and stop `px4-dds-bridge`/`xrce-agent` (keep
    `--prop-rate 150`).
- **Prop output wiring note (2026-06-30)**: the three props now live on the **AUX/FMU**
  group `AUX1/AUX2/AUX3` (moved off MAIN). PX4 output functions: `PWM_AUX_FUNC1=301`
  (Actuator Set 1 -> AUX1), `PWM_AUX_FUNC2=303` (Set3 -> AUX2), `PWM_AUX_FUNC3=302`
  (Set2 -> AUX3) — Set2/Set3 kept swapped so logical Set2/Set3 still drive the same
  physical props as before. `PWM_AUX_MIN/MAX/DIS 1/2/3 = 1000/2000/1000`. `PWM_MAIN_FUNC1/2/3`
  are now `0` (disabled). Driven by the controller via `MAV_CMD_DO_SET_ACTUATOR` (PWM, not
  DShot). NOTE: this Auterion PX4 1.14.3 build has **no `DSHOT_CONFIG` param** so DShot
  is not enable-able without reflashing stock PX4. Re-verify mapping with a per-Set
  low-power spin test after any rewire.

## Folder layout

| Folder | Runs on | What it is |
|---|---|---|
| `lower_driver_jetson/` | Jetson | C++ leg driver (`hopper_driver`). Build with `rebuild.sh`. Source = `Hopper_srbRL`. |
| `upper_controller_pc/` | PC | ModeE controller. `hopper_controller/run_modee.py` + `modee/`. `hopper_lcm_types/` kept as sibling so the `../../hopper_lcm_types/lcm_types` import resolves. |
| `imu_bridge_jetson/` | Jetson | SPLIT path: `px4_dds_bridge.py` (DDS/TELEM2 → `hopper_imu_lcmt` @500Hz) for IMU, `px4_bridge.py --no-imu` (USB MAVLink) for props (re-stream capped @150Hz). Launchers: `run_px4_dds_bridge.sh`, `run_xrce_agent.sh`, `run_px4_bridge.sh`. |
| `services/` | Jetson | systemd units: `hopper-driver`, `xrce-agent`, `px4-dds-bridge` (IMU/DDS), `px4-bridge` (props/USB, `--no-imu`). |
| `scripts/` | PC | `connect_and_start.sh` (one-shot lower-layer bring-up + LCM check) + `hopperleg_README.md`. |
| **`rlmujoco3rsr/`** | PC | **3-RSR 闭环 MuJoCo + MJX/Brax RL**（训练/评估/sim2sim/RL 部署）。见 `rlmujoco3rsr/README.md`。 |

## Bring-up

1. **Lower layer (Jetson)** — usually already installed as services. To (re)build the driver:
   ```bash
   cd lower_driver_jetson && ./rebuild.sh        # -> build/hopper_driver
   ```
   The services run `/home/nvidia/Hopper_srbRL/build/hopper_driver` etc.

2. **Bring everything up + verify (from PC)**:
   ```bash
   bash scripts/connect_and_start.sh             # SSHes to Jetson, starts services, checks CAN + LCM
   ```

3. **Upper layer (PC)**:
   ```bash
   cd upper_controller_pc/hopper_controller
   unset LD_LIBRARY_PATH
   python3 run_modee.py
   ```

## Gamepad map (current logic)

- **X** = enter PD (legs).  **A** = props ON (control_mode).  **B** = stop all (DAMP + props off).
- **LB** (only from the normal hopping cycle): arms a legs-only retraction. It actually
  starts on the **next flight phase, once the leg is back to its original length**
  (`|q_shift| <= switch_arm_l0_tol_m`), runs P1→P2→P3 and settles. A 2nd LB does nothing
  (one-way; use RB to return to hopping).
- **RB** (only after LB has settled at P3): position-controls the legs **up to
  `switch_rb_target_rad`** (P4), holds `switch_rb_settle_s`, then **enters the hopping
  cycle** (ModeE resumes). During the **first hop cycle** after RB (until the 2nd liftoff)
  the propeller PWM is floored at `hop_prop_base_pwm_us` (= 1200; props still spin only if A is ON).

Status line tags: `FLIGHT/STANCE` (normal), `+LB-ARM` (LB armed/waiting), `SW-LB-P1..3`
(retracting), `SW-LB-P3-DONE(RB?)` (settled, RB valid), `SW-RB-P4` (RB pull-up),
`+BASE1200` (post-RB prop base window).

## Key tunables

- Lower layer `qd`: published from a **2-sample central difference of the CAN position**
  (no heavy filtering), see `lower_driver_jetson/src/hardware/hopper_hardware.cpp`.
- Switch-loop / LB / RB / prop-base params: `ModeELCMConfig` at the top of
  `upper_controller_pc/hopper_controller/modee/lcm_controller.py`.

> Note: this is a **collected snapshot**, not a wired-up workspace. The Jetson-side pieces
> still build/run from `~/Hopper_srbRL` (and ROS `~/px4_ws`) on the robot; edit there (or
> sync) for live changes.
