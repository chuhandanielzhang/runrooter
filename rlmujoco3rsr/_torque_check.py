"""Check hip motor torques used by a trained policy (CPU MuJoCo rollout).
Reference: AK60-6 rated ~3 Nm, peak ~9 Nm."""
import sys
import numpy as np
import jax
import mujoco

from train_hop import HopEnv, XML, HIP_HOME
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics
from brax.io import model

PARAMS = sys.argv[1] if len(sys.argv) > 1 else "hop_policy.params"
HIP_QPOS = [7, 10, 13]
HIP_QVEL = [6, 9, 12]

env = HopEnv()
params = model.load_params(PARAMS)
ppo_net = ppo_networks.make_ppo_networks(
    env.observation_size, env.action_size,
    preprocess_observations_fn=running_statistics.normalize)
policy = jax.jit(ppo_networks.make_inference_fn(ppo_net)((params[0], params[1]),
                                                          deterministic=True))

m = mujoco.MjModel.from_xml_path(XML)
d = mujoco.MjData(m)
mujoco.mj_resetDataKeyframe(m, d, mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home"))

def obs_cpu(d):
    q, v = d.qpos, d.qvel
    w, x, y, z = q[3:7]
    grav = np.array([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)])
    return np.concatenate([[q[2]], grav, v[0:3], v[3:6],
                           q[HIP_QPOS] - HIP_HOME, v[HIP_QVEL], [q[21]]]).astype(np.float32)

rng = jax.random.PRNGKey(0)
taus, touch = [], []
for i in range(400):
    rng, k = jax.random.split(rng)
    act, _ = policy(obs_cpu(d), k)
    d.ctrl[0:3] = HIP_HOME + 0.30 * np.clip(np.array(act), -1, 1)
    for _ in range(20):
        mujoco.mj_step(m, d)
        taus.append(d.actuator_force[0:3].copy())
    touch.append(d.sensordata[-1])

taus = np.abs(np.array(taus))           # |torque| per 0.5ms substep, shape (8000, 3)
peak = taus.max()
p99 = np.percentile(taus, 99)
p95 = np.percentile(taus, 95)
mean = taus.mean()
rms = np.sqrt((taus**2).mean())
frac_over_9 = (taus > 9.0).mean() * 100
frac_over_3 = (taus > 3.0).mean() * 100

print(f"params: {PARAMS}")
print(f"|tau| peak   = {peak:6.2f} Nm")
print(f"|tau| p99    = {p99:6.2f} Nm")
print(f"|tau| p95    = {p95:6.2f} Nm")
print(f"|tau| rms    = {rms:6.2f} Nm   mean = {mean:6.2f} Nm")
print(f"time over AK60 peak  (9 Nm): {frac_over_9:5.1f} %")
print(f"time over AK60 rated (3 Nm): {frac_over_3:5.1f} %")

# plot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
t = np.arange(taus.shape[0]) * 0.0005
fig, ax = plt.subplots(figsize=(9, 4))
for j in range(3):
    ax.plot(t, np.array(taus)[:, j], lw=0.6, label=f"hip {j+1}")
ax.axhline(9, c="r", ls="--", lw=1, label="AK60 peak 9 Nm")
ax.axhline(3, c="orange", ls="--", lw=1, label="AK60 rated 3 Nm")
ax.set_xlabel("time [s]"); ax.set_ylabel("|torque| [Nm]")
ax.set_title(f"hip motor torques during hopping ({PARAMS})")
ax.legend(ncol=5, fontsize=8); ax.grid(alpha=0.3); ax.margins(x=0)
out = PARAMS.replace(".params", "") + "_torque.png"
fig.tight_layout(); fig.savefig(out, dpi=110)
print("wrote", out)
