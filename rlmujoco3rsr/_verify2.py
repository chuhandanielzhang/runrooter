import numpy as np, mujoco

m = mujoco.MjModel.from_xml_path("three_leg_3rsr_closed.xml")
d = mujoco.MjData(m)
print(f"COMPILE OK nq={m.nq} nv={m.nv} nu={m.nu} neq={m.neq}")
print(f"total mass = {m.body_mass.sum():.4f} kg  weight = {m.body_mass.sum()*9.81:.2f} N")
print("actuators:", [mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_ACTUATOR,i) for i in range(m.nu)])

key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
mujoco.mj_resetDataKeyframe(m, d, key)
mujoco.mj_forward(m, d)

# closure residual
sc = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "foot_center")
ss = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"foot_site_{i}") for i in (1,2,3)]
resid = max(np.linalg.norm(d.site_xpos[s]-d.site_xpos[sc]) for s in ss)
print(f"home closure residual = {resid*1000:.3f} mm")

# thrust direction: site frame +X axis in world (gear=1 0 0 -> force along site X)
for i in (1,2,3):
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, f"thrust_site_{i}")
    R = d.site_xmat[sid].reshape(3,3)
    print(f"thrust_site_{i} +X world dir = {R[:,0].round(3)}")

# hover test: base FREE, hold hips, thrust = weight/3 split, servo 0
bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "base")
W = m.body_mass.sum()*9.81
for th in [0.0, W/3, 1.2*W/3]:
    mujoco.mj_resetDataKeyframe(m, d, key)
    d.ctrl[0:3] = -0.060299; d.ctrl[3:6] = 0.0; d.ctrl[6:9] = th
    for _ in range(2000):
        mujoco.mj_step(m, d)
    print(f"thrust/rotor={th:6.2f}N -> base_z after 1s = {d.xpos[bid][2]:.4f} (start 0.6) nan={np.isnan(d.qpos).any()}")
