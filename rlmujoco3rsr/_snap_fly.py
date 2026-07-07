import os
os.environ.setdefault("MUJOCO_GL","egl")
import numpy as np, mujoco, imageio.v2 as imageio
m=mujoco.MjModel.from_xml_path("three_leg_3rsr_closed.xml"); d=mujoco.MjData(m)
key=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_KEY,"home"); mujoco.mj_resetDataKeyframe(m,d,key)
W=m.body_mass.sum()*9.81
bid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,"base")
for k in range(5000):
    d.ctrl[0:3]=-0.060299; d.ctrl[3:6]=0; d.ctrl[6:9]=W/3*(1.10 if k*m.opt.timestep>2 else 1.0)
    mujoco.mj_step(m,d)
ren=mujoco.Renderer(m,height=480,width=640); cam=mujoco.MjvCamera(); mujoco.mjv_defaultCamera(cam)
cam.lookat[:]=[0,0,d.xpos[bid][2]]; cam.distance=2.4; cam.elevation=-12; cam.azimuth=110
ren.update_scene(d,cam); imageio.imwrite("fly_pose.png", ren.render())
print("base_z=%.3f"%d.xpos[bid][2])
