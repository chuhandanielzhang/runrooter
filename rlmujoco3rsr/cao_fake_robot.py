"""Fake robot speaking the hopperHFAcase2026 LCM protocol, but simulating OUR
3RSR_package_2 model. Used to cross-validate the simulator: if Cao's proven
ModeE QP controller hops on this plant, the model/simulator is credible.

Mapping (calibrated by _cao_calib.py against the case repo's analytic FK;
yaw=+120 sign=-1 matches all 3 motor deflection vectors to 1 mm):
    q_lcm   = Q0_LCM - (q_sim - HOME_SIM)
    tau_sim = -tau_ff_lcm                          (clamped to +-9 Nm = AK60)
    v_imu   = Rz(+120) v_sim                       (quat_pub = quat_sim * q_z(-120))
    props: PWM -> thrust (case MotorTableModel) -> our thrust actuators per arm
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import sys
import time
import argparse

import numpy as np
import mujoco
import lcm

from runtime_paths import CASE_ROOT, CONTROLLER_DIR, LCM_TYPES_DIR
CASE = CASE_ROOT
sys.path.append(LCM_TYPES_DIR)
sys.path.append(CONTROLLER_DIR)

from python.hopper_data_lcmt import hopper_data_lcmt    # type: ignore
from python.hopper_cmd_lcmt import hopper_cmd_lcmt      # type: ignore
from python.hopper_imu_lcmt import hopper_imu_lcmt      # type: ignore
from python.gamepad_lcmt import gamepad_lcmt            # type: ignore
from python.motor_pwm_lcmt import motor_pwm_lcmt        # type: ignore
from modee.controllers.motor_utils import MotorTableModel  # type: ignore

XML = "three_leg_3rsr_closed.xml"
HOME_SIM = -0.060299
Q0_LCM = 0.0940                  # controller-side home angle (calibrated)
JOINT_SIGN = -1.0                # calibrated: sim +q retracts, controller +q extends
TAU_LIM = float(__import__("os").environ.get("FAKE_TAU", "9.0"))                    # AK60 peak
HIP_QPOS = [7, 10, 13]
HIP_QVEL = [6, 9, 12]
YAW = np.deg2rad(-120.0)         # R_sim<-imu = Rz(-120)  (v_imu = Rz(+120) v_sim)
_c, _s = np.cos(YAW), np.sin(YAW)
R_SIM_FROM_IMU = np.array([[_c, -_s, 0], [_s, _c, 0], [0, 0, 1.0]])
QZIMU = np.array([np.cos(YAW / 2), 0.0, 0.0, np.sin(YAW / 2)])  # wxyz, local z-rot by -120

# case arm conventions (IMU frame): RED at 120deg, GREEN at 0deg, BLUE at -120deg
ARM_ANGLES = {"RED": 120.0, "GREEN": 0.0, "BLUE": -120.0}
PWM_GROUPS = {"RED": (2,), "GREEN": (1,), "BLUE": (3,)}


def quat_mul(a, b):
    w1, x1, y1, z1 = a; w2, x2, y2, z2 = b
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2])


def quat_to_R(q):
    w, x, y, z = q / np.linalg.norm(q)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration-s", type=float, default=20.0)
    ap.add_argument("--strict-1d", action="store_true")
    ap.add_argument("--record-gif", type=str, default=None)
    ap.add_argument("--y-hold-s", type=float, default=0.5)
    ap.add_argument("--drop-start-s", type=float, default=1.5,
                    help="hold base at 0.55 m until t, then drop (bootstrap hopping); 0 = off")
    ap.add_argument("--realtime", action="store_true", default=True)
    args = ap.parse_args()

    m = mujoco.MjModel.from_xml_path(XML)
    # disable hip position actuators: Cao's controller sends torques (tau_ff),
    # which we apply directly via qfrc_applied
    m.actuator_gainprm[0:3, :] = 0.0
    m.actuator_biasprm[0:3, :] = 0.0
    d = mujoco.MjData(m)
    key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
    mujoco.mj_resetDataKeyframe(m, d, key)
    home_ctrl = m.key_ctrl[key].copy()
    dt = m.opt.timestep

    # map case arms -> our thrust actuators by matching prop-site XY angle in IMU frame
    arm_to_act = {}
    for arm, ang in ARM_ANGLES.items():
        best, bdif = None, 1e9
        for i in range(3):
            sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"thrust_site_{i+1}")
            p_sim = m.site_pos[sid] if m.site_bodyid[sid] == 0 else None
            # site is on prop body; compute world pos at home then to base frame
            mujoco.mj_forward(m, d)
            p = d.site_xpos[sid] - d.qpos[0:3]
            p_imu = R_SIM_FROM_IMU.T @ p
            a = np.degrees(np.arctan2(p_imu[1], p_imu[0]))
            dif = abs((a - ang + 180) % 360 - 180)
            if dif < bdif:
                bdif, best = dif, i
        arm_to_act[arm] = (best, bdif)
    print("arm -> sim thrust actuator:", {k: f"thrust_{v[0]+1} (err {v[1]:.0f} deg)" for k, v in arm_to_act.items()})

    motor_table = MotorTableModel.default_from_table()

    # SIM ISOLATION: loopback-only bus, never the real robot's 7667.
    lc = lcm.LCM(os.environ.get("LCM_DEFAULT_URL", "udpm://239.255.76.67:7669?ttl=0"))
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
        cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)
        cam.distance = 2.6; cam.elevation = -12; cam.azimuth = 110

    sim_t = 0.0
    next_gp = 0.0
    next_pub = 0.0
    next_frame = 0.0
    base_zs, taus_log = [], []
    foot_zs, hips_log, contacts_log = [], [], []
    t_wall0 = time.time()

    while sim_t < args.duration_s:
        for _ in range(16):
            if lc.handle_timeout(0) <= 0:
                break

        # synthetic gamepad (trigger ModeE start with Y at the beginning)
        if sim_t >= next_gp:
            gp = gamepad_lcmt()
            gp.rightStickAnalog = [0.0, 0.0]
            gp.leftStickAnalog = [0.0, 0.0]
            gp.y = 1 if sim_t < args.y_hold_s else 0
            gp.x = 1 if 0.5 <= sim_t < 0.7 else 0   # X edge -> PD mode (legs hop)
            gp.a = 1 if 1.0 <= sim_t < 1.2 else 0   # A edge -> PWMPD (props on)
            lc.publish("gamepad_lcmt", gp.encode())
            next_gp += 0.02

        # apply joint torques (bypass position actuators -> direct qfrc)
        tau = np.clip(JOINT_SIGN * tau_lcm, -TAU_LIM, TAU_LIM) if armed[0] else np.zeros(3)
        d.qfrc_applied[:] = 0.0
        for i, adr in enumerate(HIP_QVEL):
            d.qfrc_applied[adr] = tau[i]
        d.ctrl[3:6] = home_ctrl[3:6]        # servos frozen

        # props
        thr6 = motor_table.thrust_from_pwm(pwm_cmd) if armed[0] else np.zeros(6)
        for arm, idxs in PWM_GROUPS.items():
            act_i = arm_to_act[arm][0]
            d.ctrl[6 + act_i] = float(np.clip(sum(thr6[j] for j in idxs), 0.0, 10.0))

        mujoco.mj_step(m, d)
        if args.strict_1d:
            d.qpos[0:2] = 0; d.qpos[3:7] = [1, 0, 0, 0]
            d.qvel[0:2] = 0; d.qvel[3:6] = 0
        # bootstrap drop: hold the base in the air until the controller is armed,
        # then release -> real touchdown compression kicks off the hop limit cycle
        # (matches the case repo's own fake robot --hold-level-s behaviour; without
        # this the robot settles into a static equilibrium the energy controller
        # cannot escape: spring preload ~= weight and qd_shift stays ~0)
        if args.drop_start_s > 0 and sim_t < args.drop_start_s:
            d.qpos[0:3] = [0, 0, 0.55]; d.qpos[3:7] = [1, 0, 0, 0]
            d.qvel[0:6] = 0
        sim_t += dt

        # publish state at 1 kHz
        if sim_t >= next_pub:
            q_sim = d.qpos[HIP_QPOS]; qd_sim = np.array([d.qvel[a] for a in HIP_QVEL])
            # real AK60 drivers report low-pass filtered velocity; raw sim qd has
            # contact-impact spikes (up to 16 rad/s) that blow up the controller's
            # leg-kinematics vz estimate and kill its push energy. ~40 Hz LPF.
            qd_filt[:] = qd_filt + 0.2 * (qd_sim - qd_filt)
            jd = hopper_data_lcmt()
            jd.q = [float(v) for v in (Q0_LCM + JOINT_SIGN * (q_sim - HOME_SIM))]
            jd.qd = [float(v) for v in (JOINT_SIGN * qd_filt)]
            jd.tauIq = [float(v) for v in (JOINT_SIGN * tau)]
            lc.publish("hopper_data_lcmt", jd.encode())

            quat_sim = d.qpos[3:7]
            quat_pub = quat_mul(quat_sim, QZIMU)
            R_wi = quat_to_R(quat_pub)
            gyro_pub = R_SIM_FROM_IMU.T @ d.qvel[3:6]
            # real Pixhawk publishes gyro_y sign-flipped; Cao's lcm_controller
            # un-flips it (gyro[1] = -gyro[1]), so emulate the real convention
            gyro_pub[1] = -gyro_pub[1]
            g_w = np.array([0, 0, -9.81])
            a_w = d.qacc[0:3]
            acc_b = -(R_wi.T @ (a_w - g_w))
            im = hopper_imu_lcmt()
            im.quat = [float(v) for v in quat_pub]
            im.gyro = [float(v) for v in gyro_pub]
            im.acc = [float(v) for v in acc_b]
            R = R_wi
            im.rpy = [float(np.arctan2(R[2, 1], R[2, 2])),
                      float(np.arctan2(-R[2, 0], np.sqrt(R[2, 1]**2 + R[2, 2]**2))),
                      float(np.arctan2(R[1, 0], R[0, 0]))]
            lc.publish("hopper_imu_lcmt", im.encode())
            next_pub += 0.001

        base_zs.append(d.qpos[2]); taus_log.append(np.abs(tau).max())
        foot_zs.append(d.qpos[21]); hips_log.append(float(d.qpos[7]))
        contacts_log.append(1.0 if d.sensordata[-1] > 1e-6 else 0.0)

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
    print(f"\nRESULT: {sim_t:.1f}s  base_z min={bz.min():.3f} max={bz.max():.3f} "
          f"final={bz[-1]:.3f}  peak|tau|={max(taus_log):.2f} Nm")
    n = len(bz); third = n // 3
    print(f"late-phase base_z mean={bz[-third:].mean():.3f}  (alive&hopping if ~0.45+ and oscillating)")
    osc = bz[-third:].max() - bz[-third:].min()
    print(f"late-phase oscillation amplitude = {osc:.3f} m")
    # leg compression during stance: how much does the leg shorten while the foot
    # is on the ground? (SLIP-style buffering vs a stiff rigid hop)
    fz = np.array(foot_zs); leg_len = bz - fz; ct = np.array(contacts_log)
    hp = np.degrees(np.array(hips_log))
    if ct.sum() > 5:
        ll_c = leg_len[ct > 0.5]; hp_c = hp[ct > 0.5]
        print(f"STANCE compression: leg_len in-contact min={ll_c.min():.3f} "
              f"max={ll_c.max():.3f}  COMPRESSION STROKE={ll_c.max()-ll_c.min():.3f} m")
        print(f"STANCE hip swing: {hp_c.min():.1f} .. {hp_c.max():.1f} deg "
              f"(range {hp_c.max()-hp_c.min():.1f} deg)   contact fraction={ct.mean()*100:.0f}%")
    if args.record_gif and gif_frames:
        import imageio.v2 as imageio
        imageio.mimsave(args.record_gif, gif_frames, fps=25)
        print("wrote", args.record_gif)


if __name__ == "__main__":
    main()
