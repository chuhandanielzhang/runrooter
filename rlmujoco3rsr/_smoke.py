import jax, jax.numpy as jp, numpy as np
from train_hop import HopEnv

env = HopEnv()
print("obs_size", env.observation_size, "act_size", env.action_size, "n_sub", env._n_sub)
reset = jax.jit(env.reset)
step = jax.jit(env.step)
st = reset(jax.random.PRNGKey(0))
print("reset ok: obs", st.obs.shape, "reward", float(st.reward), "done", float(st.done))
print("obs[:6]", np.array(st.obs[:6]).round(3))
a = jp.zeros(3)
for i in range(5):
    st = step(st, a)
    print(f"step {i}: base_z={float(st.metrics['base_z']):.3f} up={float(st.metrics['upright']):.3f} "
          f"reward={float(st.reward):.3f} done={float(st.done):.0f}")
# random actions
key=jax.random.PRNGKey(1)
st=reset(key)
for i in range(20):
    key,k=jax.random.split(key)
    a=0.5*jax.random.normal(k,(3,))
    st=step(st,a)
print("random rollout final: base_z=%.3f done=%.0f reward=%.3f"%(float(st.metrics['base_z']),float(st.done),float(st.reward)))
