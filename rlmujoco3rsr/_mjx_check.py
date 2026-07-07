import time, jax, jax.numpy as jp, numpy as np, mujoco
from mujoco import mjx

print("backend:", jax.default_backend(), jax.devices())
m = mujoco.MjModel.from_xml_path("three_leg_3rsr_closed.xml")
print(f"CPU model OK nq={m.nq} nv={m.nv} nu={m.nu} neq={m.neq}")
mx = mjx.put_model(m)

# single-env jit step
dx = mjx.make_data(mx).replace(qpos=jp.array(m.key_qpos[0]), ctrl=jp.array(m.key_ctrl[0]))
step = jax.jit(mjx.step)
t0=time.time()
for _ in range(100): dx = step(mx, dx)
dx.qpos.block_until_ready()
print(f"100 single MJX steps OK in {time.time()-t0:.2f}s (incl compile)  closure resid via qpos ok={not bool(np.isnan(np.array(dx.qpos)).any())}")

# batched rollout (what RL uses)
N = 2048
q0 = jp.array(m.key_qpos[0]); c0 = jp.array(m.key_ctrl[0])
batch = jax.vmap(lambda _: mjx.make_data(mx).replace(qpos=q0, ctrl=c0))(jp.arange(N))
bstep = jax.jit(jax.vmap(mjx.step, in_axes=(None,0)))
batch = bstep(mx, batch); batch.qpos.block_until_ready()  # compile
t0=time.time()
for _ in range(50): batch = bstep(mx, batch)
batch.qpos.block_until_ready()
dt=time.time()-t0
print(f"{N} envs x 50 steps in {dt:.2f}s -> {N*50/dt:,.0f} steps/s  (Ready for RL)")
