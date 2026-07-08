"""Decisive actuation-direction test at the OPERATING configuration.

Chain under test (exactly what flight swing does):
  foot_des (FRD, small offset) --controller runtime FK/J--> q_des_lcm
  --calibrated map--> q_sim targets --sim position actuators--> settled foot
  --M_FRD--> achieved foot displacement (FRD)

Reports the angle between commanded and achieved horizontal displacement at
leg length ~= l0 (0.455), where the controller actually operates. If this is
~180 deg, the Raibert reversal lives in the leg map at the operating point.
"""
import os
import sys

import mujoco
import numpy as np

sys.path.append(".")
from runtime_paths import CONTROLLER_DIR
sys.path.insert(0, CONTROLLER_DIR)
from forward_kinematics import ForwardKinematics  # runtime (read-only import)

from modee_fake_robot import M_FRD, HOME_SIM, Q0_LCM, PERM, JOINT_SIGN  # noqa: E402

XML = "three_leg_3rsr_closed.xml"
L_TARGET = float(os.environ.get("LDC_LEN", "0.44"))   # operating leg length
STEP_M = 0.03

fk = ForwardKinematics()

def fk_len(q_scalar):
    p, _ = fk.forward_kinematics(np.full(3, q_scalar))
    return float(np.linalg.norm(p)), np.asarray(p)

# --- find controller-side q_lcm* with leg length = L_TARGET ---
qs = np.arange(-0.3, 1.2, 0.005)
best_q, best_err = None, 1e9
for q in qs:
    try:
        ln, _ = fk_len(q)
    except Exception:
        continue
    if np.isfinite(ln) and abs(ln - L_TARGET) < best_err:
        best_err, best_q = abs(ln - L_TARGET), q
q_op = float(best_q)
ln_op, p_op = fk_len(q_op)
print(f"[fk] q_lcm*={q_op:.3f} -> leg len {ln_op:.4f} (target {L_TARGET})  foot={p_op.round(3)}")

# --- sim: find q_sim* with same leg length ---
m = mujoco.MjModel.from_xml_path(XML)
key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
fid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "foot")

def settle(targets):
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, key)
    for k in range(6000):
        a = min(1.0, k / 3000)
        d.ctrl[0:3] = HOME_SIM + a * (np.asarray(targets) - HOME_SIM)
        d.ctrl[3:] = m.key_ctrl[key][3:]
        mujoco.mj_step(m, d)
        d.qpos[0:3] = [0, 0, 0.7]
        d.qpos[3:7] = [1, 0, 0, 0]
        d.qvel[0:6] = 0
    mujoco.mj_forward(m, d)
    return d.xpos[fid] - np.array([0, 0, 0.7])

qsim = np.arange(-1.4, 0.4, 0.02)
best_qs, best_e = None, 1e9
for q in qsim:
    p = settle(np.full(3, q))
    e = abs(np.linalg.norm(p) - L_TARGET)
    if e < best_e:
        best_e, best_qs = e, q
q_sim_op = float(best_qs)
p_sim_op = settle(np.full(3, q_sim_op))
print(f"[sim] q_sim*={q_sim_op:.3f} -> leg len {np.linalg.norm(p_sim_op):.4f}")
print(f"[map check] q_lcm from map = {Q0_LCM + JOINT_SIGN*(q_sim_op - HOME_SIM):.3f}  vs fk q_lcm*={q_op:.3f}")

# --- direction test: command foot offsets via runtime kinematics ---
# numeric Jacobian of runtime FK at q_op
J = np.zeros((3, 3))
eps = 1e-4
p0, _ = fk.forward_kinematics(np.full(3, q_op))
p0 = np.asarray(p0)
for j in range(3):
    qq = np.full(3, q_op)
    qq[j] += eps
    pj, _ = fk.forward_kinematics(qq)
    J[:, j] = (np.asarray(pj) - p0) / eps

print(f"\n{'cmd (FRD)':>12} | {'achieved (FRD)':>20} | angle err")
for name, dvec in [("+x", [1, 0]), ("-x", [-1, 0]), ("+y", [0, 1]), ("-y", [0, -1])]:
    d_frd = np.array([dvec[0], dvec[1], 0.0]) * STEP_M
    dq_lcm = np.linalg.solve(J, d_frd)          # controller-side joint step
    q_lcm_des = np.full(3, q_op) + dq_lcm
    # map to sim: q_sim[PERM[i]] = HOME_SIM + (Q0_LCM - q_lcm[i]) / 1  (JOINT_SIGN=-1)
    q_sim_des = np.full(3, q_sim_op)
    for i in range(3):
        q_sim_des[PERM[i]] = HOME_SIM + JOINT_SIGN * (q_lcm_des[i] - Q0_LCM)
    p1 = settle(q_sim_des)
    dp_sim = p1 - p_sim_op
    dp_frd = M_FRD @ dp_sim
    a_cmd = np.degrees(np.arctan2(d_frd[1], d_frd[0]))
    a_ach = np.degrees(np.arctan2(dp_frd[1], dp_frd[0]))
    err = (a_ach - a_cmd + 180) % 360 - 180
    print(f"{name:>12} | [{dp_frd[0]:+.4f},{dp_frd[1]:+.4f},{dp_frd[2]:+.4f}] | {err:+6.1f} deg  |mag {np.hypot(dp_frd[0],dp_frd[1])/STEP_M:.2f}x")
