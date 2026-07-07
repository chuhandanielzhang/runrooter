import mujoco, numpy as np
from PIL import Image

m = mujoco.MjModel.from_xml_path('three_leg_3rsr_closed.xml')
d = mujoco.MjData(m)
mujoco.mj_forward(m, d)
cam = mujoco.MjvCamera()
mujoco.mjv_defaultCamera(cam)
cam.lookat[:] = [0.0, 0.0, 0.45]
cam.distance = 1.8
cam.azimuth = 70
cam.elevation = -18
with mujoco.Renderer(m, 480, 640) as r:
    r.update_scene(d, cam)
    img = r.render()
Image.fromarray(img).save('_full_axes.png')
print('saved _full_axes.png')
