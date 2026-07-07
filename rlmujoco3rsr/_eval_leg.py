"""Evaluate the PURE-LEG low-level policy (leg_policy.params) on CPU MuJoCo.

Drives the leg with a force/length command and reports how well the actual foot
contact force tracks the commanded force (the whole point of the low level).
Also renders a gif. Props/servos are frozen at zero (pure leg).

Usage:
    python _eval_leg.py leg_policy.params [secs] [f_cmd 0..1] [l_cmd m]
    F_SCHED=step  python _eval_leg.py ...   # step f_cmd 0.3->0.8 at half time
"""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import sys
import numpy as np
import jax
import mujoco
import imageio.v2 as imageio

from train_leg import LegEnv, LEG_FRAME, HW_HIST, F_MAX, ACTION_SCALE, L_MIN, L_MAX
from train_hop import XML, HIP_HOME, MG
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics
from brax.io import model

PARAMS = sys.argv[1] if len(sys.argv) > 1 else "leg_policy.params"
SECS = float(sys.argv[2]) if len(sys.argv) > 2 else 8.0
F_CMD = float(sys.argv[3]) if len(sys.argv) > 3 else 0.6   # normalized [0,1]
L_CMD = float(sys.argv[4]) if len(sys.argv) > 4 else 0.40  # m
VX = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0      # horizontal vel cmd
VY = float(sys.argv[6]) if len(sys.argv) > 6 else 0.0
VXY = np.array([VX, VY], np.float32)
F_SCHED = os.environ.get("F_SCHED")                        # "step" -> 0.3 then 0.8
HIP_QPOS = [7, 10, 13]
HIP_QVEL = [6, 9, 12]
FOOT_Z = 21

env = LegEnv(hw_obs=True)
params = model.load_params(PARAMS)
net = ppo_networks.make_ppo_networks(
    env.observation_size, env.action_size,
    preprocess_observations_fn=running_statistics.normalize)
policy = jax.jit(ppo_networks.make_inference_fn(net)((params[0], params[1]),
                                                      deterministic=True))
key = jax.random.PRNGKey(0)

m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home"))
home_ctrl = m.key_ctrl[0].copy()


def f_cmd_at(t):
    if F_SCHED == "step":
        return 0.3 if t < SECS / 2 else 0.8
    return F_CMD


def frame_obs(prev_act, f_cmd, l_cmd):
    q, v = d.qpos, d.qvel
    w, x, y, z = q[3:7]
    grav = np.array([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)])
    R_T = np.array([
        [1-2*(y*y+z*z), 2*(x*y+z*w), 2*(x*z-y*w)],
        [2*(x*y-z*w), 1-2*(x*x+z*z), 2*(y*z+x*w)],
    ])
    v_hat = R_T @ v[0:3]
    contact = 1.0 if d.sensordata[-1] > 1e-6 else 0.0
    return np.concatenate([grav, v[3:6], q[HIP_QPOS] - HIP_HOME, v[HIP_QVEL],
                           v_hat, [contact], [f_cmd], VXY, [l_cmd], prev_act]).astype(np.float32)


prev_act = np.zeros(env.action_size, np.float32)
hist = np.tile(frame_obs(prev_act, f_cmd_at(0.0), L_CMD), HW_HIST)
n_ctrl = int(SECS * 100)
qposes, zs, f_act, f_des_log, taus, contacts, leglen = [], [], [], [], [], [], []
t = 0.0
for i in range(n_ctrl):
    fc = f_cmd_at(t)
    act, _ = policy(hist, key)
    act = np.clip(np.asarray(act), -1, 1)
    d.ctrl[0:3] = HIP_HOME + ACTION_SCALE * act[0:3]
    d.ctrl[3:6] = 0.0
    d.ctrl[6:9] = 0.0
    for _ in range(20):
        mujoco.mj_step(m, d)
        taus.append(np.abs(d.actuator_force[0:3]).max())
    t += 0.01
    prev_act = act
    hist = np.concatenate([hist[LEG_FRAME:], frame_obs(prev_act, fc, L_CMD)])
    qposes.append(d.qpos.copy())
    zs.append(d.qpos[2])
    ft = float(d.sensordata[-1])
    f_act.append(ft)
    f_des_log.append(fc * float(F_MAX))
    contacts.append(1.0 if ft > 1e-6 else 0.0)
    leglen.append(d.qpos[2] - d.qpos[FOOT_Z])
    up = 1 - 2*(d.qpos[4]**2 + d.qpos[5]**2)
    if d.qpos[2] < 0.20 or up < 0.3:
        print(f"FELL at t={t:.2f}s")
        break

zs = np.array(zs); f_act = np.array(f_act); f_des_log = np.array(f_des_log)
contacts = np.array(contacts); leglen = np.array(leglen)
air = contacts < 0.5
flights = int(np.sum(np.diff((~air).astype(int)) == 1))
incontact = contacts > 0.5
print(f"{PARAMS} on CPU MuJoCo, {t:.1f}s  (F_CMD={F_CMD} L_CMD={L_CMD} sched={F_SCHED}):")
print(f"  base_z min={zs.min():.3f} max={zs.max():.3f}  flights={flights}  airtime={air.mean()*100:.0f}%")
if incontact.sum() > 0:
    ferr = np.abs(f_act[incontact] - f_des_log[incontact]) / MG
    print(f"  STANCE force tracking: f_des={f_des_log[incontact].mean():.1f}N "
          f"f_act mean={f_act[incontact].mean():.1f}N  |err|={ferr.mean():.2f} xmg "
          f"({ferr.mean()*MG:.1f}N)")
    print(f"  peak GRF={f_act.max()/MG:.1f}x mg   peak|tau|={max(taus):.1f} Nm")
print(f"  leg_len mean={leglen.mean():.3f} (cmd L={L_CMD})  flight leg_len={leglen[air].mean() if air.sum() else float('nan'):.3f}")

mfull = mujoco.MjModel.from_xml_path(XML)
dr = mujoco.MjData(mfull)
ren = mujoco.Renderer(mfull, height=480, width=640)
cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)
cam.distance = 2.6; cam.elevation = -12; cam.azimuth = 110
frames = []
for k, q in enumerate(qposes):
    if k % 4:
        continue
    dr.qpos[:] = q
    mujoco.mj_forward(mfull, dr)
    cam.lookat[:] = [q[0], q[1], max(0.5, q[2])]
    ren.update_scene(dr, cam)
    frames.append(ren.render())
out = PARAMS.replace(".params", "") + "_leg.gif"
imageio.mimsave(out, frames, fps=25)
print("wrote", out)
