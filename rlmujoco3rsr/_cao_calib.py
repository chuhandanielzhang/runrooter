"""Calibrate the sim->controller joint mapping for running Cao's ModeE on OUR model.

Find s (sign) and q0 (controller-side home angle) such that
    q_lcm = q0 + s * (q_sim - HOME_SIM)
makes the case repo's analytic FK reproduce our model's foot kinematics
(in the IMU frame = sim base frame yawed by -60 deg)."""
import sys
import numpy as np
import mujoco

sys.path.insert(0, "/home/abc/Hopper/hopperHFAcase2026/hopper_controller")
from forward_kinematics import ForwardKinematics

HOME_SIM = -0.060299
DEF = 0.25
import os
YAW = np.deg2rad(float(os.environ.get("CAL_YAW_DEG", "-60")))   # sim base -> IMU yaw
c, s_ = np.cos(YAW), np.sin(YAW)
R_IMU_FROM_SIM = np.array([[c, -s_, 0], [s_, c, 0], [0, 0, 1.0]])
# case FK outputs are in the delta/"vicon" frame: +Z DOWN (core.py robot2vicon)
VICON_FROM_IMU = np.diag([1.0, 1.0, -1.0])
R_IMU_FROM_SIM = VICON_FROM_IMU @ R_IMU_FROM_SIM

fk = ForwardKinematics()

# ---- our model: home leg vector + per-joint deflection directions (base frame) ----
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
        d.qpos[0:3] = [0, 0, 0.6]; d.qpos[3:7] = [1, 0, 0, 0]; d.qvel[0:6] = 0
    mujoco.mj_forward(m, d)
    return R_IMU_FROM_SIM @ (d.xpos[fid] - np.array([0, 0, 0.6]))  # -> IMU frame

p0_sim = settle(np.full(3, HOME_SIM))
print(f"[sim] home foot (IMU frame) = {p0_sim.round(4)}  leg_len={np.linalg.norm(p0_sim):.4f}")
sim_dirs = []
for j in range(3):
    t = np.full(3, HOME_SIM); t[j] += DEF
    p = settle(t)
    d = p - p0_sim
    sim_dirs.append(d)
    print(f"[sim] joint{j+1} +{DEF} -> d(IMU) = {d.round(4)}")

# ---- search q0, s on the analytic FK ----
best = None
for s in (+1.0, -1.0):
    for q0 in np.arange(-1.2, 1.2, 0.002):
        try:
            pos0, chk = fk.forward_kinematics(np.full(3, q0))
        except Exception:
            continue
        if not np.all(np.isfinite(pos0)) or np.max(np.abs(chk)) > 1e-6:
            continue
        err_z = abs(pos0[2] - p0_sim[2])
        if err_z > 0.02:
            continue
        # deflection direction match
        tot = err_z * 5
        ok = True
        for j in range(3):
            th = np.full(3, q0); th[j] += s * DEF
            try:
                p, chk2 = fk.forward_kinematics(th)
            except Exception:
                ok = False; break
            if not np.all(np.isfinite(p)) or np.max(np.abs(chk2)) > 1e-6:
                ok = False; break
            d_fk = p - pos0
            # compare direction + magnitude
            tot += np.linalg.norm(d_fk - sim_dirs[j])
        if not ok:
            continue
        if best is None or tot < best[0]:
            best = (tot, s, q0, pos0.copy())

if best is None:
    print("NO MATCH FOUND")
else:
    tot, s, q0, pos0 = best
    print(f"\nBEST: sign={s:+.0f}  q0={q0:.4f} rad   total_err={tot:.4f}")
    print(f"  FK home foot = {pos0.round(4)}  vs sim {p0_sim.round(4)}")
    for j in range(3):
        th = np.full(3, q0); th[j] += s * DEF
        p, _ = fk.forward_kinematics(th)
        print(f"  motor{j}: FK d={ (p-pos0).round(4) }  sim d={sim_dirs[j].round(4)}")
