"""Sim <-> real mapping for deploying hop_policy_hw.params on the OMEGA-style
3-RSR hopper (hopperHFAcase2026 interface: AK60 over CAN + Pixhawk IMU via LCM).

Verified numerically against hopperHFAcase2026/hopper_controller/forward_kinematics.py
(see _align_check.py):

  1. FRAME YAW: deflecting each motor, the foot-motion directions in the real
     IMU frame vs our sim base frame differ by a uniform +60 deg yaw about Z.
     => rotate IMU vectors (gravity, gyro) by R_z(YAW_SIM_FROM_IMU) before
        feeding the policy; joint indices then map 1:1 (no permutation):
        sim ctrl_joint_1/2/3  <->  physical motor 0/1/2
     (the case repo's own -30deg KIN_YAW_OFFSET + perm [1,2,0] are internal to
      its analytic FK and already accounted for in the comparison.)

  2. JOINT SIGN (CALIBRATE ON BENCH): geometry suggests real +q extends the leg
     while sim +q retracts it, but a pure direction test cannot distinguish a
     mirror; verify with ONE bench nudge: command +0.1 rad on motor 0; if the
     leg EXTENDS, set JOINT_SIGN = -1.

  3. KNOWN RESIDUAL GAPS (fix before free hopping):
     - rod length: sim 0.4028 m vs real d=0.398 m (~2.4 cm foot height offset)
     - home offset: sim home hip angle = -0.0603 rad; real home must be measured
       at the same physical stance and entered below.

Policy I/O (train_hop.py --hw, velocity-command version):
  obs  = 5-frame history of [grav(3), gyro(3), q-HOME(3), dq(3), v_hat_xy(2),
                             cmd_xy(2), prev_act(6), cos, sin]  (120 dims, 100 Hz)
  act  = 3 hip position targets (q_des = HOME + 0.30*act, kp=100 kd=3 via
         hopper_cmd_lcmt) + 3 per-arm thrusts ([0,6] N -> motor_pwm_lcmt PWM)
"""
import numpy as np

# CALIBRATED via _cao_calib.py + closed-loop validation with Cao's ModeE controller
# (cao_fake_robot.py): yaw=-120, sign=-1, q0=0.094 match all 3 motor deflection
# vectors of the case repo's analytic FK to ~1 mm, and the controller hops on our
# model with this mapping (23 liftoffs / 12 s).
YAW_SIM_FROM_IMU_DEG = -120.0        # v_sim = Rz(-120 deg) v_imu
JOINT_SIGN = -1                      # sim +q retracts leg, real/controller +q extends
Q_HOME_SIM = -0.060299               # sim home hip angle (rad)
Q_HOME_REAL = 0.0940                 # controller-side home angle at the same stance
ACTION_SCALE = 0.80          # rad; +-0.8 stroke enables up to ~25 cm commanded hops
HOP_PERIOD = 0.4                     # s, must match training
CTRL_DT = 0.01                       # 100 Hz policy

_c, _s = np.cos(np.deg2rad(YAW_SIM_FROM_IMU_DEG)), np.sin(np.deg2rad(YAW_SIM_FROM_IMU_DEG))
R_SIM_FROM_IMU = np.array([[_c, -_s, 0.0], [_s, _c, 0.0], [0.0, 0.0, 1.0]])


def imu_vec_to_sim(v_imu):
    """gravity / gyro vector: IMU body frame -> sim base frame."""
    return R_SIM_FROM_IMU @ np.asarray(v_imu, dtype=float)


def q_real_to_sim(q_real):
    return JOINT_SIGN * (np.asarray(q_real, dtype=float) - Q_HOME_REAL) + Q_HOME_SIM


def qd_real_to_sim(qd_real):
    return JOINT_SIGN * np.asarray(qd_real, dtype=float)


def action_to_q_des_real(action):
    """policy action (-1..1) -> real motor position targets (hopper_cmd_lcmt.q_des)."""
    q_sim_des = Q_HOME_SIM + ACTION_SCALE * np.clip(np.asarray(action, dtype=float), -1, 1)
    return JOINT_SIGN * (q_sim_des - Q_HOME_SIM) + Q_HOME_REAL


def gravity_from_quat(quat_wxyz):
    """projected gravity (unit, body frame) from IMU quaternion, then -> sim frame."""
    w, x, y, z = quat_wxyz
    g_body = np.array([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)])
    return imu_vec_to_sim(g_body)


# ---- foot FK in the sim base frame (for the Cao-style velocity estimator) ----
# uses the hardware-validated analytic FK from the case repo; its output is in
# the "vicon" frame (+Z down), which differs from the IMU frame by a Z flip
# (verified in _cao_calib.py).
_VICON_TO_IMU = np.diag([1.0, 1.0, -1.0])
_fk = None


def foot_pos_sim(q_real):
    """foot position in the sim base frame, directly from real motor angles."""
    global _fk
    if _fk is None:
        import os, sys
        case = os.environ.get("CASE_REPO", "/home/abc/Hopper/hopperHFAcase2026")
        sys.path.insert(0, os.path.join(case, "hopper_controller"))
        from forward_kinematics import ForwardKinematics
        _fk = ForwardKinematics()
    p_vicon, _ = _fk.forward_kinematics(np.asarray(q_real, dtype=float))
    return R_SIM_FROM_IMU @ (_VICON_TO_IMU @ p_vicon)


def foot_jac_sim(q_real, eps=1e-5):
    """numeric Jacobian d(foot_pos_sim)/d(q_real), 3x3."""
    q = np.asarray(q_real, dtype=float)
    p0 = foot_pos_sim(q)
    J = np.zeros((3, 3))
    for j in range(3):
        dq = np.zeros(3); dq[j] = eps
        J[:, j] = (foot_pos_sim(q + dq) - p0) / eps
    return J
