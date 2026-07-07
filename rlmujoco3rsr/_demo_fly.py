import os
os.environ.setdefault("MUJOCO_GL","egl")
import numpy as np, mujoco, imageio.v2 as imageio

m = mujoco.MjModel.from_xml_path("three_leg_3rsr_closed.xml")
d = mujoco.MjData(m)
key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
mujoco.mj_resetDataKeyframe(m, d, key)
W = m.body_mass.sum()*9.81
hov = W/3.0
home = -0.060299

ren = mujoco.Renderer(m, height=480, width=640)
cam = mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)
cam.lookat[:] = [0,0,0.6]; cam.distance=2.6; cam.elevation=-12; cam.azimuth=110

frames=[]; dt=m.opt.timestep
T=4.0; n=int(T/dt)
bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY,"base")
for k in range(n):
    t=k*dt
    d.ctrl[0:3]=home
    # phase: 0-1s ramp to hover; 1-2.5s hover; 2.5-4s climb at 1.18x with slight tilt
    d.ctrl[3:6]=0  # servos straight (no open-loop tilt; vectoring is RL's job)
    if t<2.0:
        th=hov              # steady hover from t=0 (holds 0.6)
    else:
        th=hov*1.10         # climb straight up
    d.ctrl[6:9]=th
    mujoco.mj_step(m,d)
    if k% 40==0:
        cam.lookat[2]=max(0.6, d.xpos[bid][2])
        ren.update_scene(d,cam); frames.append(ren.render())

print("frames",len(frames),"final base_z=%.3f"%d.xpos[bid][2])
imageio.mimsave("demo_fly.gif", frames, fps=25)
imageio.mimsave("demo_fly.mp4", frames, fps=25, quality=8)
print("wrote demo_fly.gif / demo_fly.mp4")
