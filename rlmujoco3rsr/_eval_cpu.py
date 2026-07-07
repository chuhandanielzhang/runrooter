"""10s rollout of a --hw policy on classic CPU MuJoCo (sim2sim) + gif."""
import os
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("JAX_PLATFORMS", "cpu")
import sys
import numpy as np
import jax
import mujoco
import imageio.v2 as imageio

from train_hop import (HopEnv, XML, HIP_HOME, ACT_DIM, THRUST_MAX,
                       HW_FRAME, HW_HIST, HOP_PERIOD, hop_period, Z_STAND)
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics
from brax.io import model

PARAMS = sys.argv[1] if len(sys.argv) > 1 else "hop_policy_outdoor.params"
SECS = float(sys.argv[2]) if len(sys.argv) > 2 else 10.0
CMD = np.array([float(sys.argv[3]) if len(sys.argv) > 3 else 0.0,
                float(sys.argv[4]) if len(sys.argv) > 4 else 0.0], np.float32)
H_CMD = float(sys.argv[5]) if len(sys.argv) > 5 else 0.08
YAW_CMD = float(os.environ.get("YAW_CMD", 0.0))   # yaw-rate command (rad/s)
T_HOP = float(hop_period(H_CMD))
FRICTION = float(sys.argv[6]) if len(sys.argv) > 6 else None  # None = XML default
HIP_QPOS = [7, 10, 13]
HIP_QVEL = [6, 9, 12]
ACTION_SCALE = 0.80

env = HopEnv(hw_obs=True)
params = model.load_params(PARAMS)
net = ppo_networks.make_ppo_networks(
    env.observation_size, env.action_size,
    preprocess_observations_fn=running_statistics.normalize)
policy = jax.jit(ppo_networks.make_inference_fn(net)((params[0], params[1]),
                                                      deterministic=True))
key = jax.random.PRNGKey(0)

m = mujoco.MjModel.from_xml_path(XML)
if FRICTION is not None:
    m.geom_friction[:, 0] = FRICTION
    print(f"== slip test: sliding friction set to {FRICTION} ==")
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home"))
home_ctrl = m.key_ctrl[0].copy()

def frame_obs(prev_act, t):
    q, v = d.qpos, d.qvel
    w, x, y, z = q[3:7]
    grav = np.array([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)])
    R_T = np.array([
        [1-2*(y*y+z*z), 2*(x*y+z*w), 2*(x*z-y*w)],
        [2*(x*y-z*w), 1-2*(x*x+z*z), 2*(y*z+x*w)],
    ])
    v_hat = R_T @ v[0:3]
    ph = 2 * np.pi * t / T_HOP
    return np.concatenate([grav, v[3:6], q[HIP_QPOS] - HIP_HOME,
                           v[HIP_QVEL], v_hat, CMD, [H_CMD], [YAW_CMD], prev_act,
                           [np.cos(ph), np.sin(ph)]]).astype(np.float32)

CMD2 = os.environ.get("CMD2")            # "vx,vy,t_switch": mid-run command change
if CMD2:
    c2x, c2y, t_sw = [float(s) for s in CMD2.split(",")]
else:
    t_sw = None

prev_act = np.zeros(ACT_DIM, np.float32)
hist = np.tile(frame_obs(prev_act, 0.0), HW_HIST)
n_ctrl = int(SECS * 100)
qposes, zs, taus, thrs, touch = [], [], [], [], []
t = 0.0
for i in range(n_ctrl):
    if t_sw is not None and t >= t_sw:
        CMD[:] = [c2x, c2y]
        print(f"== t={t:.1f}s: command switched to {CMD} ==")
        t_sw = None
    act, _ = policy(hist, key)
    act = np.clip(np.asarray(act), -1, 1)
    d.ctrl[0:3] = HIP_HOME + ACTION_SCALE * act[0:3]
    d.ctrl[3:6] = home_ctrl[3:6]
    d.ctrl[6:9] = 0.5 * THRUST_MAX * (act[3:6] + 1.0)
    for _ in range(20):
        mujoco.mj_step(m, d)
        taus.append(np.abs(d.actuator_force[0:3]).max())
    t += 0.01
    prev_act = act
    hist = np.concatenate([hist[HW_FRAME:], frame_obs(prev_act, t)])
    qposes.append(d.qpos.copy())
    zs.append(d.qpos[2])
    thrs.append(d.ctrl[6:9].copy())
    touch.append(d.sensordata[-1])
    up = 1 - 2*(d.qpos[4]**2 + d.qpos[5]**2)
    if d.qpos[2] < 0.20 or up < 0.3:
        print(f"FELL at t={t:.2f}s")
        break

zs = np.array(zs); touch = np.array(touch); thrs = np.array(thrs)
air = touch < 1e-6
flights = int(np.sum(np.diff(air.astype(int)) == 1))
print(f"{PARAMS} on CPU MuJoCo, {t:.1f}s:")
print(f"  base_z min={zs.min():.3f} max={zs.max():.3f}  flights={flights}  airtime={air.mean()*100:.0f}%")
mg = 3.75 * 9.81
grf = touch[touch > 1e-6]
if len(grf):
    print(f"  GRF: mean={grf.mean()/mg:.1f}x mg  p95={np.percentile(grf,95)/mg:.1f}x  peak={grf.max()/mg:.1f}x mg (soft: peak <4x)")
qa = np.array(qposes)
foot_clear = qa[air, 21] if air.any() else np.array([0.0])
print(f"  flight foot clearance: mean={foot_clear.mean():.3f} m  max={foot_clear.max():.3f} m")
print(f"  peak|tau|={max(taus):.1f} Nm   thrust mean/arm={thrs.mean():.1f} N")
qarr = np.array(qposes)
print(f"  xy drift: max dist from origin = {np.linalg.norm(qarr[:, 0:2], axis=1).max():.2f} m  final = {np.linalg.norm(qarr[-1, 0:2]):.2f} m")
late = zs[len(zs)//2:]
print(f"  late-half: base_z mean={late.mean():.3f}  apex max={late.max():.3f}  hop amplitude={late.max()-late.min():.3f} m")
print(f"  height cmd: h_cmd={H_CMD:.3f} m -> apex target z={Z_STAND+H_CMD:.3f}  achieved apex={late.max():.3f}")

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
out = PARAMS.replace(".params", "") + "_10s.gif"
imageio.mimsave(out, frames, fps=25)
print("wrote", out)
