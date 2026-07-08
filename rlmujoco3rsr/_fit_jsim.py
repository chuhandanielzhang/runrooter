"""Offline probe of the SIM mechanism's true foot-position Jacobian J_B(q).

For a grid of uniform hip angles (symmetric configs spanning the hopping
compression range), deflect each hip joint (central difference) and record the
settled foot displacement in the SIM BASE frame:

    J_B[:, j] = d foot_sim / d q_sim_j        (3x3 per grid point)

Saved to jsim_grid.npz (grid of mean hip angle + stacked Jacobians).
Used by modee_fake_robot --jshim to align force transmission with the
controller's analytic delta-leg model (J_A from runtime FK).
"""
import sys

import mujoco
import numpy as np

XML = "three_leg_3rsr_closed.xml"
DEF = 0.05

m = mujoco.MjModel.from_xml_path(XML)
key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
fid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "foot")


def settle(targets):
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, key)
    for k in range(5000):
        a = min(1.0, k / 2500)
        d.ctrl[0:3] = np.asarray(m.key_qpos[key][[7, 10, 13]]) * (1 - a) + a * np.asarray(targets)
        d.ctrl[3:] = m.key_ctrl[key][3:]
        mujoco.mj_step(m, d)
        d.qpos[0:3] = [0, 0, 0.7]
        d.qpos[3:7] = [1, 0, 0, 0]
        d.qvel[0:6] = 0
    mujoco.mj_forward(m, d)
    q_hip = np.array([d.qpos[7], d.qpos[10], d.qpos[13]])
    return (d.xpos[fid] - np.array([0, 0, 0.7])).copy(), q_hip


grid = np.arange(-0.95, 0.11, 0.10)   # uniform hip angle; leg len ~0.51 .. 0.36
qs, Js, lens = [], [], []
for qu in grid:
    p0, qhip0 = settle(np.full(3, qu))
    J = np.zeros((3, 3))
    for j in range(3):
        tp = np.full(3, qu); tp[j] += DEF
        tm = np.full(3, qu); tm[j] -= DEF
        pp, _ = settle(tp)
        pm, _ = settle(tm)
        J[:, j] = (pp - pm) / (2 * DEF)
    qs.append(float(np.mean(qhip0)))
    Js.append(J)
    lens.append(float(np.linalg.norm(p0)))
    print(f"q_uni={qu:+.2f} (settled mean={qs[-1]:+.3f})  leg_len={lens[-1]:.4f}  "
          f"|J|col=[{np.linalg.norm(J[:,0]):.3f},{np.linalg.norm(J[:,1]):.3f},{np.linalg.norm(J[:,2]):.3f}]")

np.savez("jsim_grid.npz", q_mean=np.array(qs), J=np.stack(Js), leg_len=np.array(lens))
print(f"saved jsim_grid.npz  ({len(qs)} points)")
