import time, numpy as np, mujoco, jax, jax.numpy as jp
from mujoco import mjx

m = mujoco.MjModel.from_xml_path("three_leg_3rsr_closed.xml")
d = mujoco.MjData(m)
key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
sc = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "foot_center")
ss = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"foot_site_{i}") for i in (1,2,3)]
bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")

# 1) closure residual under aggressive hip motion (CPU)
mujoco.mj_resetDataKeyframe(m, d, key)
rng = np.random.default_rng(0); maxr=0; maxbz=0; minbz=9
for k in range(4000):
    d.ctrl[0:3] = -0.060299 + 0.3*np.sin(0.01*k + rng.uniform(0,6,3))
    mujoco.mj_step(m, d)
    r = max(np.linalg.norm(d.site_xpos[s]-d.site_xpos[sc]) for s in ss)
    maxr=max(maxr,r); bz=d.xpos[bid][2]; maxbz=max(maxbz,bz); minbz=min(minbz,bz)
print(f"[closure] max residual under motion = {maxr*1000:.3f} mm   base_z range=[{minbz:.3f},{maxbz:.3f}]")

# 2) leg authority: from a stance touching ground, slam hips to extend and see base rise
mujoco.mj_resetDataKeyframe(m, d, key)
d.qpos[2] = 0.40  # lower base so foot near ground
mujoco.mj_forward(m, d)
for k in range(3000):
    # squat (k<800) then extend hard
    d.ctrl[0:3] = (-0.060299+0.5) if (800 < k < 1200) else -0.060299
    mujoco.mj_step(m, d)
print(f"[authority] after squat+extend: base_z={d.xpos[bid][2]:.3f} (start ~0.40), vz={d.cvel[bid][5]:+.2f}")

# 3) steps/s on GPU with iterations=50
mx = mjx.put_model(m)
N=2048; q0=jp.array(m.key_qpos[0]); c0=jp.array(m.key_ctrl[0])
batch = jax.vmap(lambda _: mjx.make_data(mx).replace(qpos=q0,ctrl=c0))(jp.arange(N))
bstep = jax.jit(jax.vmap(mjx.step, in_axes=(None,0)))
batch=bstep(mx,batch); batch.qpos.block_until_ready()
t0=time.time()
for _ in range(50): batch=bstep(mx,batch)
batch.qpos.block_until_ready(); dt=time.time()-t0
print(f"[speed] {N} envs x50 -> {N*50/dt:,.0f} steps/s (was 8,821 @ iters=200)")
