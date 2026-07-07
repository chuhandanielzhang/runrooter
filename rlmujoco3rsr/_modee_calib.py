"""Re-calibrate the sim->controller joint mapping against the CURRENT runtime stack
(robot_runtime/upper_controller_pc), whose conventions changed since the CASE era:

  - forward_kinematics.py: motor permutation now IDENTITY (was [1,2,0]), yaw offset 0
  - body frame = leg FK frame = IMU frame = FRD (+X fwd, +Y right, +Z down)
  - world NED-like (+Z down)

We search (perm, sign s, home q0, base->FRD yaw) such that
    q_lcm[i] = q0 + s * (q_sim[perm[i]] - HOME_SIM)
makes the runtime analytic FK reproduce our MuJoCo model's foot position and
per-joint deflection directions, where sim base vectors map to FRD via
    v_frd = diag(1,-1,-1) @ Rz(yaw) @ v_sim.
"""
import itertools
import sys

import mujoco
import numpy as np

from runtime_paths import CONTROLLER_DIR
sys.path.insert(0, CONTROLLER_DIR)
from forward_kinematics import ForwardKinematics  # runtime version

HOME_SIM = -0.060299
DEF = 0.25
HIP_QPOS = [7, 10, 13]

fk = ForwardKinematics()

m = mujoco.MjModel.from_xml_path("three_leg_3rsr_closed.xml")
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
        d.qpos[0:3] = [0, 0, 0.6]
        d.qpos[3:7] = [1, 0, 0, 0]
        d.qvel[0:6] = 0
    mujoco.mj_forward(m, d)
    return d.xpos[fid] - np.array([0, 0, 0.6])  # sim base frame (Z-up)


print("settling sim probes...")
p0_sim = settle(np.full(3, HOME_SIM))
sim_dirs = []
for j in range(3):
    t = np.full(3, HOME_SIM)
    t[j] += DEF
    sim_dirs.append(settle(t) - p0_sim)
    print(f"[sim] joint{j+1} +{DEF} -> d(sim base) = {sim_dirs[j].round(4)}")
print(f"[sim] home foot (sim base) = {p0_sim.round(4)}  len={np.linalg.norm(p0_sim):.4f}")

FLIP = np.diag([1.0, -1.0, -1.0])  # sim Z-up body -> FRD (Rx(180))


def to_frd(v, yaw):
    c, s_ = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1.0]])
    return FLIP @ (Rz @ v)


best = None
yaws = np.deg2rad(np.arange(-180, 180, 5.0))
for perm in itertools.permutations(range(3)):
    for s in (+1.0, -1.0):
        for q0 in np.arange(-1.2, 1.2, 0.01):
            try:
                pos0, chk = fk.forward_kinematics(np.full(3, q0))
            except Exception:
                continue
            if not np.all(np.isfinite(pos0)) or np.max(np.abs(chk)) > 1e-6:
                continue
            # FK z is +down; sim home foot z_frd = -p0_sim[2] > 0. quick gate:
            if abs(float(pos0[2]) - (-p0_sim[2])) > 0.03:
                continue
            # FK deflection vectors for each PHYSICAL lcm motor i
            d_fk = []
            ok = True
            for i in range(3):
                th = np.full(3, q0)
                th[i] += s * DEF
                try:
                    p, chk2 = fk.forward_kinematics(th)
                except Exception:
                    ok = False
                    break
                if not np.all(np.isfinite(p)) or np.max(np.abs(chk2)) > 1e-6:
                    ok = False
                    break
                d_fk.append(np.asarray(p) - np.asarray(pos0))
            if not ok:
                continue
            # match against sim dirs mapped to FRD for each yaw
            for yaw in yaws:
                tot = abs(float(pos0[2]) - (-p0_sim[2])) * 5
                for i in range(3):
                    d_sim_frd = to_frd(sim_dirs[perm[i]], yaw)
                    tot += float(np.linalg.norm(d_fk[i] - d_sim_frd))
                if best is None or tot < best[0]:
                    best = (tot, perm, s, q0, yaw)

if best is None:
    print("NO MATCH")
    sys.exit(1)

tot, perm, s, q0, yaw = best
print(f"\nBEST: perm(lcm_i <- sim_{{perm[i]}})={perm}  sign={s:+.0f}  q0={q0:.4f}  "
      f"yaw={np.rad2deg(yaw):+.1f} deg  total_err={tot:.4f}")
pos0, _ = fk.forward_kinematics(np.full(3, q0))
print(f"  FK home foot (FRD) = {np.asarray(pos0).round(4)}  vs sim->FRD {to_frd(p0_sim, yaw).round(4)}")
for i in range(3):
    th = np.full(3, q0)
    th[i] += s * DEF
    p, _ = fk.forward_kinematics(th)
    print(f"  lcm motor{i}: FK d={(np.asarray(p)-np.asarray(pos0)).round(4)}  "
          f"sim d(FRD)={to_frd(sim_dirs[perm[i]], yaw).round(4)}")
