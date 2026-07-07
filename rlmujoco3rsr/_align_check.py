"""Verify coordinate alignment: MuJoCo sim joints <-> physical motors (case repo).

Method: deflect one joint at a time on both sides, compare the XY direction the
foot moves in the IMU/body frame. The case repo's FK outputs are already
IMU-frame-aligned (KIN_YAW_OFFSET -30 deg + PERM applied internally).
"""
import sys
import numpy as np
import mujoco

sys.path.insert(0, "/home/abc/Hopper/hopperHFAcase2026/hopper_controller")
from forward_kinematics import ForwardKinematics

HOME = -0.060299
DEF = 0.25

# ---------- case repo analytic FK (physical motor order in, IMU frame out) ----------
fk = ForwardKinematics()
pos0, chk0 = fk.forward_kinematics(np.full(3, HOME))
print(f"[analytic] home foot pos (IMU frame) = {pos0.round(4)}  closure_check={chk0.round(6)}")

phys_dirs = {}
for j in range(3):
    th = np.full(3, HOME); th[j] += DEF
    p, _ = fk.forward_kinematics(th)
    dxy = p[:2] - pos0[:2]
    phys_dirs[j] = np.degrees(np.arctan2(dxy[1], dxy[0]))
    print(f"[analytic] phys motor {j} +{DEF}rad -> foot moves xy dir {phys_dirs[j]:7.1f} deg, dz={p[2]-pos0[2]:+.4f}")

# ---------- MuJoCo sim (pinned base, position servo to target) ----------
m = mujoco.MjModel.from_xml_path("three_leg_3rsr_closed.xml")
key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
fid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "foot")

def settle(targets):
    d = mujoco.MjData(m)
    mujoco.mj_resetDataKeyframe(m, d, key)
    for k in range(6000):
        a = min(1.0, k / 3000)
        d.ctrl[0:3] = HOME + a * (np.asarray(targets) - HOME)
        d.ctrl[3:] = m.key_ctrl[key][3:]
        mujoco.mj_step(m, d)
        d.qpos[0:3] = [0, 0, 0.6]; d.qpos[3:7] = [1, 0, 0, 0]; d.qvel[0:6] = 0
    mujoco.mj_forward(m, d)
    return d.xpos[fid] - np.array([0, 0, 0.6])   # foot pos in base frame

p0 = settle(np.full(3, HOME))
print(f"\n[mujoco] home foot pos (base frame) = {p0.round(4)}")
sim_dirs = {}
for j in range(3):
    t = np.full(3, HOME); t[j] += DEF
    p = settle(t)
    dxy = p[:2] - p0[:2]
    sim_dirs[j] = np.degrees(np.arctan2(dxy[1], dxy[0]))
    print(f"[mujoco] sim joint {j+1} +{DEF}rad -> foot moves xy dir {sim_dirs[j]:7.1f} deg, dz={p[2]-p0[2]:+.4f}")

# ---------- match ----------
print("\n---- mapping (closest direction) ----")
for sj, sd in sim_dirs.items():
    best = min(phys_dirs, key=lambda pj: abs((phys_dirs[pj] - sd + 180) % 360 - 180))
    err = abs((phys_dirs[best] - sd + 180) % 360 - 180)
    print(f"sim ctrl_joint_{sj+1}  <->  phys motor {best}   (direction err {err:.1f} deg)")
print(f"\nz check: analytic home z={pos0[2]:+.4f} vs mujoco home z={p0[2]:+.4f}")
