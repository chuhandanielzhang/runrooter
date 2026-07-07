"""Static open-loop check of the propeller moment map.

For each PWM channel (1,2,3), apply thrust T on its mapped sim actuator with the
base pinned, and measure the angular acceleration -> body moment in FRD. Compare
with what the CURRENT controller believes (core.py prop_positions_b, thrust along
body -Z_frd i.e. up):

    expected tau_frd = r_frd x F_frd,  F_frd = (0,0,-T)

Controller arm layout (core.py, FRD): PWM1 arm at (0,-L), PWM2 at (-s3/2 L, +L/2),
PWM3 at (+s3/2 L, +L/2), L = 0.569451.
"""
import sys

import mujoco
import numpy as np

sys.path.append(".")
from modee_fake_robot import M_FRD, PROP_PWM_ARM_ANGLE_FRD, quat_to_R  # noqa: E402

XML = "three_leg_3rsr_closed.xml"
L = 0.569451
T = 5.0

m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)
key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")

# rebuild pwm->act map exactly like modee_fake_robot
mujoco.mj_resetDataKeyframe(m, d, key)
mujoco.mj_forward(m, d)
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

# controller expectation
ARM_FRD = {
    1: np.array([0.0, -L, 0.0]),
    2: np.array([-np.sqrt(3) / 2 * L, +0.5 * L, 0.0]),
    3: np.array([+np.sqrt(3) / 2 * L, +0.5 * L, 0.0]),
}
F_FRD = np.array([0.0, 0.0, -T])

print(f"{'PWM':>4} {'sim act':>8} | {'expected tau_frd (r x F)':>28} | {'measured tau_frd':>28} | sign match?")
for pwm_idx in (1, 2, 3):
    act = pwm_to_act[pwm_idx]
    mujoco.mj_resetDataKeyframe(m, d, key)
    d.ctrl[:] = m.key_ctrl[key]
    d.ctrl[6:9] = 0.0
    d.ctrl[6 + act] = T
    mujoco.mj_forward(m, d)
    # measured: total applied torque about COM in sim world -> map to FRD.
    # Use qacc of the free base (mass matrix weighted) — simpler: use qfrc_actuator
    # angular part? Instead integrate one small step and read angular velocity.
    d.qvel[:] = 0.0
    n_sub = 20
    for _ in range(n_sub):
        d.ctrl[6 + act] = T
        mujoco.mj_step(m, d)
    omega_sim = d.qvel[3:6].copy() / (n_sub * m.opt.timestep)  # ~ I^-1 tau
    # direction is what matters; map body-local omega_dot to FRD
    omega_frd = M_FRD @ omega_sim
    tau_exp = np.cross(ARM_FRD[pwm_idx], F_FRD)
    # compare directions of roll/pitch components
    def sgn(v):
        return "0" if abs(v) < 1e-6 else ("+" if v > 0 else "-")
    match_x = sgn(tau_exp[0]) == sgn(omega_frd[0]) or sgn(tau_exp[0]) == "0"
    match_y = sgn(tau_exp[1]) == sgn(omega_frd[1]) or sgn(tau_exp[1]) == "0"
    print(f"{pwm_idx:>4} {'thrust_'+str(act+1):>8} | {np.array2string(tau_exp, precision=2):>28} "
          f"| {np.array2string(omega_frd, precision=2):>28} | x:{match_x} y:{match_y}")
