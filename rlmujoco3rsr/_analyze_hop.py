"""Is the trained policy really hopping? Check flight phases via foot contact force."""
import jax, jax.numpy as jp
import numpy as np

from train_hop import HopEnv
from brax.training.agents.ppo import networks as ppo_networks
from brax.training.acme import running_statistics
from brax.io import model

env = HopEnv()
params = model.load_params("hop_policy.params")
ppo_net = ppo_networks.make_ppo_networks(
    env.observation_size, env.action_size,
    preprocess_observations_fn=running_statistics.normalize)
policy = ppo_networks.make_inference_fn(ppo_net)((params[0], params[1]), deterministic=True)

jit_reset = jax.jit(env.reset)
jit_step = jax.jit(env.step)
jit_policy = jax.jit(policy)

rng = jax.random.PRNGKey(42)
state = jit_reset(rng)

# sensordata: touch sensor is the LAST entry (foot_touch_sensor)
base_z, foot_z, touch = [], [], []
for i in range(400):
    rng, k = jax.random.split(rng)
    act, _ = jit_policy(state.obs, k)
    state = jit_step(state, act)
    q = np.array(state.pipeline_state.qpos)
    base_z.append(q[2]); foot_z.append(q[21])
    touch.append(float(np.array(state.pipeline_state.sensordata)[-1]))

base_z, foot_z, touch = map(np.array, (base_z, foot_z, touch))
airborne = touch < 1e-6
# count flight phases (consecutive airborne runs >= 3 ctrl steps = 30ms)
runs, cnt = [], 0
for a in airborne:
    cnt = cnt + 1 if a else (runs.append(cnt) or 0 if cnt else 0)
if cnt: runs.append(cnt)
runs = [r for r in runs if r >= 3]

print(f"base_z:  min={base_z.min():.3f} max={base_z.max():.3f}")
print(f"foot_z:  min={foot_z.min():.3f} max={foot_z.max():.3f}")
print(f"touch force: mean={touch.mean():.2f}N max={touch.max():.2f}N  zero-frames={airborne.sum()}/{len(touch)}")
print(f"flight phases (>=30ms): {len(runs)}  durations(ctrl steps)={runs[:20]}")
print("VERDICT:", "REAL HOPPING (flight phases present)" if len(runs) >= 2 else
      "no real flight phase -> bouncing/balancing, not hopping yet")

# plot
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
t = np.arange(len(base_z)) * 0.01
fig, ax = plt.subplots(2, 1, figsize=(9, 5), sharex=True)
ax[0].plot(t, base_z, label="base z"); ax[0].plot(t, foot_z, label="foot z")
ax[0].axhline(0.6, ls="--", c="gray", lw=0.7); ax[0].set_ylabel("height [m]"); ax[0].legend(); ax[0].grid(alpha=0.3)
ax[1].plot(t, touch, c="tab:red"); ax[1].set_ylabel("foot contact force [N]"); ax[1].set_xlabel("time [s]"); ax[1].grid(alpha=0.3)
for a in ax: a.margins(x=0)
fig.suptitle("Trained policy rollout: heights & ground contact")
fig.tight_layout(); fig.savefig("hop_analysis.png", dpi=110)
print("wrote hop_analysis.png")
