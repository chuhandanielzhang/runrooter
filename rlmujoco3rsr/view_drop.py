"""Interactive MuJoCo viewer: free-fall drop from home pose.

Controls:
  - left-drag: rotate camera, right-drag: pan, scroll: zoom
  - Space: pause/resume
  - Ctrl + right-drag on a body: apply force (drag the robot around)
  - Backspace: reset
"""
import mujoco
import mujoco.viewer

m = mujoco.MjModel.from_xml_path("three_leg_3rsr_closed.xml")
d = mujoco.MjData(m)

key = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_KEY, "home")
mujoco.mj_resetDataKeyframe(m, d, key)
# hold hips at home so the leg keeps shape while falling; servos straight, thrust off
d.ctrl[:] = m.key_ctrl[key]

mujoco.viewer.launch(m, d)
