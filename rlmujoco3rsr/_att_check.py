"""Static attitude-loop diagnostic for Cao 3D mode on our fake robot.

Hold the base at a fixed small tilt, publish IMU/data exactly like
cao_fake_robot.py, and record the per-arm PWM the controller sends back.
If the differential thrust pushes the LOW side UP -> corrective (sign OK).
If it pushes the low side further down -> our quat/arm convention is wrong.
"""
import sys
import time
import numpy as np
import mujoco

sys.path.insert(0, "/home/abc/Hopper/hopperHFAcase2026/hopper_lcm_types/lcm_types")
sys.path.insert(0, "/home/abc/Hopper/hopperHFAcase2026/hopper_controller")
import lcm
from python.hopper_data_lcmt import hopper_data_lcmt    # type: ignore
from python.hopper_imu_lcmt import hopper_imu_lcmt      # type: ignore
from python.gamepad_lcmt import gamepad_lcmt            # type: ignore
from python.motor_pwm_lcmt import motor_pwm_lcmt        # type: ignore

from cao_fake_robot import (QZIMU, R_SIM_FROM_IMU, Q0_LCM, JOINT_SIGN,
                            HOME_SIM, quat_mul, quat_to_R)

XML = "three_leg_3rsr_closed.xml"
PWM_ARM = {"GREEN": 1, "RED": 2, "BLUE": 3}   # pwm channel per arm color

m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)
key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
mujoco.mj_resetDataKeyframe(m, d, key)
mujoco.mj_forward(m, d)
# arm tip positions in sim base frame (from prop sites)
arm_pos = {}
for i, name in [(1, "thrust_site_1"), (2, "thrust_site_2"), (3, "thrust_site_3")]:
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, name)
    arm_pos[f"sim{i}"] = d.site_xpos[sid] - d.qpos[0:3]

lc = lcm.LCM()
pwm_log = []
tau_log = []
def on_pwm(_, buf):
    msg = motor_pwm_lcmt.decode(buf)
    pwm_log.append(np.array(msg.pwm_values))
lc.subscribe("motor_pwm_lcmt", on_pwm)
from python.hopper_cmd_lcmt import hopper_cmd_lcmt      # type: ignore
def on_cmd(_, buf):
    msg = hopper_cmd_lcmt.decode(buf)
    tau_log.append(np.array(msg.tau_ff))
lc.subscribe("hopper_cmd_lcmt", on_cmd)

def run_case(tag, quat_sim, gyro_sim=(0.0, 0.0, 0.0)):
    """publish fixed pose for 4 s, return mean pwm of last 1 s."""
    pwm_log.clear(); tau_log.clear()
    R_ws = quat_to_R(quat_sim)
    t0 = time.time()
    next_gp = 0.0
    while True:
        t = time.time() - t0
        if t > 4.0:
            break
        q_lcm = Q0_LCM + JOINT_SIGN * (np.full(3, HOME_SIM) - HOME_SIM)
        jd = hopper_data_lcmt()
        jd.q = [float(v) for v in q_lcm]; jd.qd = [0.0]*3
        jd.tauIq = [0.0]*3
        lc.publish("hopper_data_lcmt", jd.encode())
        quat_pub = quat_mul(quat_sim, QZIMU)
        R_wi = quat_to_R(quat_pub)
        im = hopper_imu_lcmt()
        im.quat = [float(v) for v in quat_pub]
        g_pub = R_SIM_FROM_IMU.T @ np.asarray(gyro_sim, dtype=float)
        g_pub[1] = -g_pub[1]          # emulate real Pixhawk gyro_y flip
        im.gyro = [float(v) for v in g_pub]
        g_w = np.array([0, 0, -9.81])
        im.acc = [float(v) for v in -(R_wi.T @ (-g_w))]
        R = R_wi
        im.rpy = [float(np.arctan2(R[2,1], R[2,2])),
                  float(np.arctan2(-R[2,0], np.sqrt(R[2,1]**2+R[2,2]**2))),
                  float(np.arctan2(R[1,0], R[0,0]))]
        lc.publish("hopper_imu_lcmt", im.encode())
        if t >= next_gp:
            gp = gamepad_lcmt()
            gp.rightStickAnalog = [0.0, 0.0]; gp.leftStickAnalog = [0.0, 0.0]
            gp.y = 1 if t < 0.5 else 0
            gp.x = 1 if 0.5 <= t < 0.7 else 0
            gp.a = 1 if 1.0 <= t < 1.2 else 0
            lc.publish("gamepad_lcmt", gp.encode())
            next_gp += 0.02
        lc.handle_timeout(0)
        time.sleep(0.001)
    arr = np.array(pwm_log[-800:]) if pwm_log else np.zeros((1, 6))
    mean = arr.mean(axis=0)
    # pairing verified by cao_fake_robot geometry match:
    # sim thrust_1=RED->M2, thrust_2=BLUE->M3, thrust_3=GREEN->M1
    pairs = [("RED", "sim1", 2), ("BLUE", "sim2", 3), ("GREEN", "sim3", 1)]
    print(f"\n[{tag}]")
    for color, sim_name, ch in pairs:
        z = (R_ws @ arm_pos[sim_name])[2]
        print(f"   {color:6s} (pwm M{ch}): pwm={mean[ch]:6.0f}   arm world z = {z:+.4f}"
              f"   {'LOW' if z < -0.005 else ('HIGH' if z > 0.005 else '')}")
    if tau_log:
        tarr = np.array(tau_log[-800:])
        print(f"   tau_ff mean (lcm) = {tarr.mean(axis=0).round(2)}")
    return mean

a = np.deg2rad(6.0)
LEVEL = np.array([1.0, 0, 0, 0])
cases = [
    ("roll +6deg (about sim x)", np.array([np.cos(a/2), np.sin(a/2), 0, 0]), (0, 0, 0)),
    ("gyro +0.5 rad/s about sim x (level)", LEVEL, (0.5, 0, 0)),
    ("gyro -0.5 rad/s about sim x", LEVEL, (-0.5, 0, 0)),
    ("gyro +0.5 rad/s about sim y", LEVEL, (0, 0.5, 0)),
]
for tag, qs, gy in cases:
    run_case(tag, qs, gy)
