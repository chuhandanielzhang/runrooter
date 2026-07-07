"""In-process textbook prop-PD attitude test (no LCM, no ModeE).

Base pinned in air (xyz only), attitude free, hips position-held at home,
initial tilt. Control: tau_des = -kR*e - kW*omega (FRD), allocate to 3 arms by
least squares on the controller's arm layout, clip 0..10 N.

If this stabilizes -> plant + M_FRD mapping fine; divergence is in ModeE config.
"""
import sys

import mujoco
import numpy as np

sys.path.append(".")
from modee_fake_robot import M_FRD, PROP_PWM_ARM_ANGLE_FRD, quat_to_R, rpy_zyx  # noqa: E402

XML = "three_leg_3rsr_closed.xml"
L = 0.569451
ARM_FRD = {
    1: np.array([0.0, -L, 0.0]),
    2: np.array([-np.sqrt(3) / 2 * L, +0.5 * L, 0.0]),
    3: np.array([+np.sqrt(3) / 2 * L, +0.5 * L, 0.0]),
}
# moment map: tau_xy = A @ t   (thrust along -Z_frd)
A = np.stack([np.cross(ARM_FRD[i], np.array([0, 0, -1.0]))[0:2] for i in (1, 2, 3)], axis=1)
A_pinv = np.linalg.pinv(A)

m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)
key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
mujoco.mj_resetDataKeyframe(m, d, key)
a2 = 0.5 * np.deg2rad(10.0)
d.qpos[3:7] = [np.cos(a2), np.sin(a2), 0.0, 0.0]  # 10 deg about sim X
mujoco.mj_forward(m, d)

# pwm->act map (same as fake robot)
pwm_to_act = {}
for pwm_idx, ang_des in PROP_PWM_ARM_ANGLE_FRD.items():
    best, bdif = None, 1e9
    for i in range(3):
        sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"thrust_site_{i+1}")
        p_frd = M_FRD @ (d.site_xpos[sid] - d.qpos[0:3])
        a = np.degrees(np.arctan2(p_frd[1], p_frd[0]))
        dif = abs((a - ang_des + 180) % 360 - 180)
        if dif < bdif:
            bdif, best = dif, i
    pwm_to_act[pwm_idx] = best

kR, kW = 20.0, 4.0
T_base = 3.0  # collective bias per arm
dt = m.opt.timestep
for step in range(int(8.0 / dt)):
    R_frd = M_FRD @ quat_to_R(d.qpos[3:7]) @ M_FRD.T
    roll, pitch, yaw = rpy_zyx(R_frd)
    omega_frd = M_FRD @ d.qvel[3:6]
    tau_des = np.array([-kR * roll - kW * omega_frd[0],
                        -kR * pitch - kW * omega_frd[1]])
    t3 = np.clip(T_base + A_pinv @ tau_des, 0.0, 10.0)
    d.ctrl[:] = m.key_ctrl[key]
    for pwm_idx, act in pwm_to_act.items():
        d.ctrl[6 + act] = float(t3[pwm_idx - 1])
    mujoco.mj_step(m, d)
    d.qpos[0:3] = [0, 0, 0.55]
    d.qvel[0:3] = 0
    if step % int(0.5 / dt) == 0:
        print(f"t={step*dt:5.2f} roll={np.rad2deg(roll):+7.2f} pitch={np.rad2deg(pitch):+7.2f} "
              f"t3=[{t3[0]:.1f},{t3[1]:.1f},{t3[2]:.1f}]")
