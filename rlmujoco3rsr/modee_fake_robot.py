"""Fake robot for the CURRENT runtime stack (robot_runtime/upper_controller_pc).

Speaks the same LCM protocol as the real Jetson driver + PX4 DDS bridge, but
simulates the 3-RSR closed-chain MuJoCo model. Conventions match TODAY's
core.py / lcm_controller.py / px4_dds_bridge.py (NOT the CASE-era ones):

  - body frame = leg FK frame = IMU frame = FRD (+X fwd, +Y right, +Z down)
  - world NED-like (+Z down); level+stationary acc = [0,0,-9.81] (specific force)
  - NO gyro_y sign flip (the old flip pair was removed from bridge+controller)
  - forward_kinematics.py with IDENTITY motor permutation, yaw offset 0

Joint mapping (recalibrated by _modee_calib.py against the runtime FK, err ~4mm):
    q_lcm[i]  = Q0_LCM - (q_sim[PERM[i]] - HOME_SIM)        (sign = -1)
    qd_lcm[i] = -qd_sim[PERM[i]]
    tau_sim[PERM[i]] = -tau_ff_lcm[i]
    frame map: v_frd = diag(1,-1,-1) @ Rz(+90 deg) @ v_sim

Run (pair with run_modee_on_sim.py):
    python3 modee_fake_robot.py --duration-s 20 --drop-start-s 1.5
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import argparse
import sys
import time

import mujoco
import numpy as np
import lcm

from runtime_paths import CONTROLLER_DIR, LCM_TYPES_DIR
sys.path.append(LCM_TYPES_DIR)
sys.path.append(CONTROLLER_DIR)

from python.hopper_data_lcmt import hopper_data_lcmt    # type: ignore
from python.hopper_cmd_lcmt import hopper_cmd_lcmt      # type: ignore
from python.hopper_imu_lcmt import hopper_imu_lcmt      # type: ignore
from python.gamepad_lcmt import gamepad_lcmt            # type: ignore
from python.motor_pwm_lcmt import motor_pwm_lcmt        # type: ignore
from modee.controllers.motor_utils import MotorTableModel  # type: ignore

XML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "three_leg_3rsr_closed.xml")

# ---- calibration (see _modee_calib.py output, 2026-07-04) ----
# NOTE: the 3-RSR leg has C3 (120 deg) symmetry, so the FK-deflection calibration
# admits THREE equivalent (PERM, yaw) branches. Only one is consistent with the
# FIXED IMU yaw below; the touchdown-direction probe selects it (MFR_PERM_BRANCH).
HOME_SIM = -0.060299
Q0_LCM = 0.0900
_PERM_BRANCHES = ((0, 2, 1), (1, 0, 2), (2, 1, 0))  # C3 cyclic relabelings
PERM = _PERM_BRANCHES[int(os.environ.get("MFR_PERM_BRANCH", "0"))]
JOINT_SIGN = -1.0
YAW_FRD = np.deg2rad(90.0)

HIP_QPOS = [7, 10, 13]
HIP_QVEL = [6, 9, 12]

_c, _s = np.cos(YAW_FRD), np.sin(YAW_FRD)
# v_frd = M @ v_sim  (sim base/world Z-up -> FRD/NED-like Z-down); M is proper (det=+1)
M_FRD = np.diag([1.0, -1.0, -1.0]) @ np.array([[_c, -_s, 0], [_s, _c, 0], [0, 0, 1.0]])

# current core.py prop convention (FRD), remapped 2026-07-06 (core.py:643):
#   arm 0 (-90deg, (0,-L))   -> PWM[3]
#   arm 1 (+150deg, (-x,+y)) -> PWM[2]
#   arm 2 (+30deg, (+x,+y))  -> PWM[1]
PROP_PWM_ARM_ANGLE_FRD = {3: -90.0, 2: +150.0, 1: +30.0}
TAU_LIM = float(os.environ.get("FAKE_TAU", "25.0"))


def quat_to_R(q):
    w, x, y, z = np.asarray(q, dtype=float) / max(1e-12, float(np.linalg.norm(q)))
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def R_to_quat(R):
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        return np.array([0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s])
    i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
    if i == 0:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        return np.array([(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s])
    if i == 1:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        return np.array([(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s])
    s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
    return np.array([(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s])


def rpy_zyx(R):
    """Aerospace ZYX rpy from R_wb (FRD->NED), matches core._R_to_rpy_xyz."""
    roll = float(np.arctan2(R[2, 1], R[2, 2]))
    pitch = float(np.arctan2(-R[2, 0], np.sqrt(max(1e-12, R[2, 1] ** 2 + R[2, 2] ** 2))))
    yaw = float(np.arctan2(R[1, 0], R[0, 0]))
    return roll, pitch, yaw


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration-s", type=float, default=20.0)
    # Gamepad edges must fire AFTER the controller process is up and subscribed,
    # otherwise it misses X (PD legs) / A (props) and never actuates.
    ap.add_argument("--y-at-s", type=float, default=2.5, help="Y edge (log/reset)")
    ap.add_argument("--x-at-s", type=float, default=3.0, help="X edge (PD legs on)")
    ap.add_argument("--a-at-s", type=float, default=3.5, help="A edge (props on)")
    ap.add_argument("--drop-start-s", type=float, default=4.5)
    ap.add_argument("--strict-1d", action="store_true",
                    help="lock base xy + attitude every step (pure vertical hopping)")
    ap.add_argument("--strict-2d", action="store_true",
                    help="lock base y + roll + yaw (x-z plane motion, pitch free)")
    ap.add_argument("--init-roll-deg", type=float, default=0.0,
                    help="initial base roll (sim X axis) applied at reset")
    ap.add_argument("--hold-hips", action="store_true",
                    help="keep hip POSITION actuators at home (ignore tau) - prop-only attitude test")
    ap.add_argument("--hold-att-free", action="store_true",
                    help="during pre-drop hold, pin xyz but leave attitude FREE")
    ap.add_argument("--print-roll", action="store_true", help="print roll/pitch trace at 5 Hz")
    ap.add_argument("--record-gif", type=str, default=None)
    ap.add_argument("--realtime", action="store_true", default=True)
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(XML)
    # controller sends torques; bypass hip position actuators (unless --hold-hips)
    if not args.hold_hips:
        m.actuator_gainprm[0:3, :] = 0.0
        m.actuator_biasprm[0:3, :] = 0.0
    d = mujoco.MjData(m)
    key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(m, d, key)
    home_ctrl = m.key_ctrl[key].copy()
    dt = m.opt.timestep
    if abs(args.init_roll_deg) > 1e-9:
        a2 = 0.5 * np.deg2rad(args.init_roll_deg)
        d.qpos[3:7] = [np.cos(a2), np.sin(a2), 0.0, 0.0]
        mujoco.mj_forward(m, d)

    # map PWM channels -> sim thrust actuators by arm angle in FRD
    mujoco.mj_forward(m, d)
    pwm_to_act = {}
    for pwm_idx, ang_des in PROP_PWM_ARM_ANGLE_FRD.items():
        best, bdif = None, 1e9
        for i in range(3):
            sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"thrust_site_{i+1}")
            p_simbody = d.site_xpos[sid] - d.qpos[0:3]
            p_frd = M_FRD @ p_simbody
            a = np.degrees(np.arctan2(p_frd[1], p_frd[0]))
            dif = abs((a - ang_des + 180) % 360 - 180)
            if dif < bdif:
                bdif, best = dif, i
        pwm_to_act[pwm_idx] = (best, bdif)
    print("PWM ch -> sim thrust actuator:",
          {k: f"thrust_{v[0]+1} (err {v[1]:.0f} deg)" for k, v in pwm_to_act.items()})

    motor_table = MotorTableModel.default_from_table()
    lc = lcm.LCM()
    tau_lcm = np.zeros(3)
    qd_filt = np.zeros(3)
    pwm_cmd = np.full(6, 1000.0)
    armed = [False]

    def on_cmd(_, buf):
        nonlocal tau_lcm
        tau_lcm = np.array(hopper_cmd_lcmt.decode(buf).tau_ff, dtype=float)

    def on_pwm(_, buf):
        nonlocal pwm_cmd
        msg = motor_pwm_lcmt.decode(buf)
        pwm_cmd = np.array(msg.pwm_values, dtype=float)
        armed[0] = int(msg.control_mode) > 0

    lc.subscribe("hopper_cmd_lcmt", on_cmd)
    lc.subscribe("motor_pwm_lcmt", on_pwm)

    gif_frames = []
    ren = cam = None
    if args.record_gif:
        ren = mujoco.Renderer(m, height=480, width=640)
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(cam)
        cam.distance = 2.6
        cam.elevation = -12
        cam.azimuth = 110

    sim_t = 0.0
    next_gp = next_pub = next_frame = 0.0
    base_zs, taus_log, foot_zs, contacts_log = [], [], [], []
    t_wall0 = time.time()
    prev_contact = False
    td_events = []  # (t, v_xy_sim, foot_offset_xy_sim) at each touchdown

    while sim_t < args.duration_s:
        for _ in range(16):
            if lc.handle_timeout(0) <= 0:
                break

        # synthetic gamepad: Y (log/reset) -> X (PD legs) -> A (props on)
        if sim_t >= next_gp:
            gp = gamepad_lcmt()
            gp.rightStickAnalog = [0.0, 0.0]
            gp.leftStickAnalog = [0.0, 0.0]
            gp.y = 1 if args.y_at_s <= sim_t < args.y_at_s + 0.2 else 0
            gp.x = 1 if args.x_at_s <= sim_t < args.x_at_s + 0.2 else 0
            gp.a = 1 if args.a_at_s <= sim_t < args.a_at_s + 0.2 else 0
            lc.publish("gamepad_lcmt", gp.encode())
            next_gp += 0.02

        # joint torques: tau_sim[PERM[i]] = JOINT_SIGN * tau_lcm[i]
        tau_sim = np.zeros(3)
        if armed[0] and not args.hold_hips:
            for i in range(3):
                tau_sim[PERM[i]] = float(np.clip(JOINT_SIGN * tau_lcm[i], -TAU_LIM, TAU_LIM))
        d.qfrc_applied[:] = 0.0
        for j, adr in enumerate(HIP_QVEL):
            d.qfrc_applied[adr] = tau_sim[j]
        if args.hold_hips:
            d.ctrl[0:3] = home_ctrl[0:3]
        d.ctrl[3:6] = home_ctrl[3:6]  # servos frozen

        # props: PWM -> thrust -> mapped sim thrust actuator
        thr6 = motor_table.thrust_from_pwm(pwm_cmd) if armed[0] else np.zeros(6)
        d.ctrl[6:9] = 0.0
        for pwm_idx, (act_i, _) in pwm_to_act.items():
            d.ctrl[6 + act_i] = float(np.clip(thr6[pwm_idx], 0.0, 10.0))

        mujoco.mj_step(m, d)
        if args.strict_1d:
            d.qpos[0:2] = 0
            d.qpos[3:7] = [1, 0, 0, 0]
            d.qvel[0:2] = 0
            d.qvel[3:6] = 0
        elif args.strict_2d:
            # keep x-z plane translation + pitch-about-sim-Y only.
            # NOTE frame map yaw=+90deg: FRD x (controller fwd) = sim -y? Careful:
            # v_frd = diag(1,-1,-1) Rz(90) v_sim -> sim X maps to FRD -Y (lateral),
            # sim Y maps to FRD -X... so controller-plane (FRD x-z) = sim (y,z).
            # Lock sim x + the rotation that couples out of that plane.
            q = np.asarray(d.qpos[3:7], dtype=float)
            R = quat_to_R(q)
            # allowed rotation: about sim X axis (which maps to FRD +Y = pitch axis)
            ang = np.arctan2(R[2, 1], R[1, 1])  # rotation about sim x
            d.qpos[0] = 0
            d.qpos[3:7] = [np.cos(ang / 2), np.sin(ang / 2), 0, 0]
            d.qvel[0] = 0
            d.qvel[4:6] = 0
        if args.drop_start_s > 0 and sim_t < args.drop_start_s:
            d.qpos[0:3] = [0, 0, 0.55]
            d.qvel[0:3] = 0
            if args.hold_att_free:
                pass  # attitude left free (prop attitude-loop sign test)
            else:
                d.qpos[3:7] = [1, 0, 0, 0]
                d.qvel[3:6] = 0
        sim_t += dt

        # publish at 1 kHz motor / 500 Hz IMU-equivalent (both at 1kHz is fine)
        if sim_t >= next_pub:
            q_sim = d.qpos[HIP_QPOS]
            qd_sim = np.array([d.qvel[a] for a in HIP_QVEL])
            qd_filt[:] = qd_filt + 0.2 * (qd_sim - qd_filt)  # ~40Hz LPF like AK60 driver

            jd = hopper_data_lcmt()
            jd.q = [float(Q0_LCM + JOINT_SIGN * (q_sim[PERM[i]] - HOME_SIM)) for i in range(3)]
            jd.qd = [float(JOINT_SIGN * qd_filt[PERM[i]]) for i in range(3)]
            jd.tauIq = [float(JOINT_SIGN * tau_sim[PERM[i]]) for i in range(3)]
            lc.publish("hopper_data_lcmt", jd.encode())

            # ---- IMU in FRD/NED (current px4_dds_bridge conventions) ----
            R_wb_sim = quat_to_R(d.qpos[3:7])          # sim body -> sim world (both Z-up)
            R_frd = M_FRD @ R_wb_sim @ M_FRD.T          # FRD body -> NED-like world
            gyro_frd = M_FRD @ d.qvel[3:6]              # body-local angular velocity, NO extra flips
            g_w_sim = np.array([0.0, 0.0, -9.81])
            sf_simbody = R_wb_sim.T @ (d.qacc[0:3] - g_w_sim)   # specific force, sim body
            if args.drop_start_s > 0 and sim_t < args.drop_start_s:
                sf_simbody = R_wb_sim.T @ (-g_w_sim)             # held: looks stationary
            acc_frd = M_FRD @ sf_simbody                # level+rest -> [0,0,-9.81]

            im = hopper_imu_lcmt()
            im.quat = [float(v) for v in R_to_quat(R_frd)]
            im.gyro = [float(v) for v in gyro_frd]
            im.acc = [float(v) for v in acc_frd]
            im.rpy = [float(v) for v in rpy_zyx(R_frd)]
            lc.publish("hopper_imu_lcmt", im.encode())
            next_pub += 0.001

        base_zs.append(d.qpos[2])
        taus_log.append(np.abs(tau_sim).max())
        foot_zs.append(d.qpos[21])
        in_contact = d.sensordata[-1] > 1e-6
        contacts_log.append(1.0 if in_contact else 0.0)

        # Raibert direction probe: at touchdown, does the foot land AHEAD of the
        # hip along the velocity (braking, correct) or BEHIND (accelerating, flipped)?
        if in_contact and not prev_contact and sim_t > args.drop_start_s:
            v_xy = np.array([d.qvel[0], d.qvel[1]])
            foot_off = np.array([d.qpos[19] - d.qpos[0], d.qpos[20] - d.qpos[1]])
            td_events.append((sim_t, v_xy.copy(), foot_off.copy()))
            vn = float(np.linalg.norm(v_xy))
            if vn > 0.03:
                dot = float(v_xy @ foot_off) / vn
                print(f"[TD] t={sim_t:5.2f} v_xy_sim=[{v_xy[0]:+.2f},{v_xy[1]:+.2f}] "
                      f"foot_off_sim=[{foot_off[0]:+.3f},{foot_off[1]:+.3f}] "
                      f"along_v={dot:+.3f} m ({'BRAKE ok' if dot > 0 else 'REVERSED'})",
                      flush=True)
        prev_contact = in_contact

        if args.print_roll and int(sim_t / dt) % int(0.2 / dt) == 0:
            R_dbg = quat_to_R(d.qpos[3:7])
            R_frd_dbg = M_FRD @ R_dbg @ M_FRD.T
            r_dbg, p_dbg, _ = rpy_zyx(R_frd_dbg)
            thr_now = [float(d.ctrl[6 + i]) for i in range(3)]
            print(f"t={sim_t:5.2f} roll={np.rad2deg(r_dbg):+6.1f} pitch={np.rad2deg(p_dbg):+6.1f} "
                  f"thrust={thr_now[0]:.1f},{thr_now[1]:.1f},{thr_now[2]:.1f} armed={int(armed[0])}",
                  flush=True)

        if ren is not None and sim_t >= next_frame:
            cam.lookat[:] = [d.qpos[0], d.qpos[1], max(0.5, d.qpos[2])]
            ren.update_scene(d, cam)
            gif_frames.append(ren.render())
            next_frame += 0.04

        if args.realtime:
            lag = (t_wall0 + sim_t) - time.time()
            if lag > 0:
                time.sleep(lag)

    bz = np.array(base_zs)
    ct = np.array(contacts_log)
    n = len(bz)
    third = n // 3
    print(f"\nRESULT: {sim_t:.1f}s  base_z min={bz.min():.3f} max={bz.max():.3f} final={bz[-1]:.3f}  "
          f"peak|tau|={max(taus_log):.2f} Nm")
    print(f"late-phase base_z mean={bz[-third:].mean():.3f}  oscillation={bz[-third:].max()-bz[-third:].min():.3f} m  "
          f"contact fraction={ct.mean()*100:.0f}%")
    if args.record_gif and gif_frames:
        import imageio.v2 as imageio
        imageio.mimsave(args.record_gif, gif_frames, fps=25)
        print("wrote", args.record_gif)


if __name__ == "__main__":
    main()
