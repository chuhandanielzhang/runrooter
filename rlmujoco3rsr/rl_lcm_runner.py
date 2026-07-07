"""RL policy runner for the hopperHFAcase2026 stack (drop-in alternative to ModeECore).

Same LCM I/O as modee/lcm_controller.py:
    subscribe : hopper_data_lcmt (q, qd)        100+ Hz from CAN bridge
                hopper_imu_lcmt  (quat, gyro)   from flight controller
    publish   : hopper_cmd_lcmt  (q_des, kp, kd)        -> AK60 MIT mode
                motor_pwm_lcmt   (6ch PWM -> Betaflight)  -> fixed props, no servos

Policy: train_hop.py --dr --hw  (obs = 5-frame history of
        [grav(3) gyro(3) q-HOME(3) dq(3) v_hat_xy(2) cmd_xy(2) prev_act(6)
         cos sin] = 120 dims, act = 3 hip targets + 3 per-arm thrusts, 100 Hz)

v_hat is the Cao-style velocity estimate: during the stance half of the gait
clock, base velocity from leg kinematics (foot pinned to ground); during
flight, hold the last stance value. cmd defaults to (0,0) = hop in place.

Hardware mapping notes:
  - joints: sim ctrl_joint_1/2/3 <-> physical motor 0/1/2 (1:1, +60deg yaw on IMU
    vectors; see deploy_map.py; JOINT_SIGN still needs ONE bench nudge test)
  - props: 3 installed, per-arm PWM channels from core.py prop_pwm_idx_per_arm:
    arm0(RED)->M2, arm1(GREEN)->M1, arm2(BLUE)->M3.
    TODO bench-verify which SIM prop (thrust_1/2/3) is which COLOR arm.
  - thrust->PWM: Hopper4 mapping pwm = 1000 + sqrt(T / k_thrust)  (from core.py)

Run on the same machine that runs run_modee.py:
    JAX_PLATFORMS=cpu python rl_lcm_runner.py --params hop_policy_prop.params
"""
import os
import sys
import time
import argparse
import threading

import numpy as np

from runtime_paths import CASE_ROOT, LCM_TYPES_DIR, POLICIES_DIR
CASE_REPO = CASE_ROOT
sys.path.append(LCM_TYPES_DIR)

import lcm
from python.hopper_data_lcmt import hopper_data_lcmt   # type: ignore
from python.hopper_cmd_lcmt import hopper_cmd_lcmt     # type: ignore
from python.hopper_imu_lcmt import hopper_imu_lcmt     # type: ignore
from python.motor_pwm_lcmt import motor_pwm_lcmt       # type: ignore

import deploy_map as dm

# ---- policy / control constants (must match training) ----
CTRL_DT = 0.01                 # 100 Hz
HW_HIST = 5
ACT_DIM = 6
HW_FRAME = 19 + ACT_DIM        # 25: +v_hat(2) +cmd_xy(2) +h_cmd(1)

def hop_period(h_cmd):
    return 0.16 + 2.0 * np.sqrt(2.0 * h_cmd / 9.81)
KP_JOINT = 100.0               # = sim kp (MIT mode)
KD_JOINT = 3.0                 # = sim kv
THRUST_MAX = 6.0               # N per arm = 49% weight total (same clamp as sim)
# sim prop -> arm color verified by cao_fake_robot geometry match (err 0 deg):
#   thrust_1=RED->M2, thrust_2=BLUE->M3, thrust_3=GREEN->M1
ARM_PWM_IDX = (2, 3, 1)
K_THRUST = 1.47e-5             # N / (pwm-1000)^2, from core.py hopper4 mapping
PWM_MIN, PWM_MAX = 1000.0, 1700.0
STALE_S = 0.05                 # watchdog: stop if sensors older than this
RAMP_S = 1.0                   # soft-start: scale actions 0->1 over first second


def load_policy(params_path):
    """Brax PPO inference on CPU JAX (exact same network as training)."""
    import jax
    from train_hop import HopEnv  # also installs the compat shim
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.acme import running_statistics
    from brax.io import model
    env = HopEnv(hw_obs=True)
    assert env.observation_size == HW_FRAME * HW_HIST and env.action_size == ACT_DIM
    params = model.load_params(params_path)
    net = ppo_networks.make_ppo_networks(
        env.observation_size, env.action_size,
        preprocess_observations_fn=running_statistics.normalize)
    raw = jax.jit(ppo_networks.make_inference_fn(net)((params[0], params[1]),
                                                       deterministic=True))
    key = jax.random.PRNGKey(0)
    raw(np.zeros(HW_FRAME * HW_HIST, np.float32), key)  # compile once

    def policy(obs):
        act, _ = raw(obs.astype(np.float32), key)
        return np.asarray(act)
    return policy


def thrust_to_pwm(thrust_n):
    """per-arm thrust [N] -> 6ch PWM us (Hopper4 sqrt mapping from core.py)."""
    pwm = np.full(6, PWM_MIN)
    for i, t in enumerate(np.clip(thrust_n, 0.0, THRUST_MAX)):
        pwm[ARM_PWM_IDX[i]] = np.clip(PWM_MIN + np.sqrt(max(t, 0.0) / K_THRUST),
                                      PWM_MIN, PWM_MAX)
    return pwm


class VelocityEstimator:
    """Cao-style base velocity estimate (sim base frame, m/s).

    STANCE (gait-clock cos(phase) > 0): foot is pinned, so
        v_base = -(J(q) qd + omega x p_foot)   [all in sim base frame]
    FLIGHT: hold the last stance estimate (same as Cao's 'FLIGHT: HOLD XY')."""

    def __init__(self):
        self.v = np.zeros(3)

    def update(self, q_real, qd_real, gyro_sim, in_stance):
        if in_stance:
            p = dm.foot_pos_sim(q_real)
            v_foot = dm.foot_jac_sim(q_real) @ np.asarray(qd_real, dtype=float)
            self.v = -(v_foot + np.cross(gyro_sim, p))
        return self.v[0:2].copy()


class RLRunner:
    def __init__(self, policy, cmd=(0.0, 0.0), h_cmd=0.08):
        self.policy = policy
        self.cmd = np.asarray(cmd, dtype=float)
        self.h_cmd = float(h_cmd)
        self.t_hop = hop_period(self.h_cmd)
        self.lc = lcm.LCM()
        self.lock = threading.Lock()
        self.q = np.zeros(3); self.qd = np.zeros(3)
        self.quat = np.array([1.0, 0, 0, 0]); self.gyro = np.zeros(3)
        self.t_data = 0.0; self.t_imu = 0.0
        self.prev_act = np.zeros(ACT_DIM)
        self.hist = None
        self.t0 = None
        self.vel_est = VelocityEstimator()
        self.lc.subscribe("hopper_data_lcmt", self._on_data)
        self.lc.subscribe("hopper_imu_lcmt", self._on_imu)

    def _on_data(self, _, buf):
        msg = hopper_data_lcmt.decode(buf)
        with self.lock:
            self.q = np.array(msg.q); self.qd = np.array(msg.qd)
            self.t_data = time.time()

    def _on_imu(self, _, buf):
        msg = hopper_imu_lcmt.decode(buf)
        with self.lock:
            self.quat = np.array(msg.quat); self.gyro = np.array(msg.gyro)
            self.t_imu = time.time()

    def _frame(self, t):
        with self.lock:
            q, qd, quat, gyro = self.q.copy(), self.qd.copy(), self.quat.copy(), self.gyro.copy()
        grav_sim = dm.gravity_from_quat(quat)
        gyro_sim = dm.imu_vec_to_sim(gyro)
        q_sim = dm.q_real_to_sim(q) - dm.Q_HOME_SIM
        qd_sim = dm.qd_real_to_sim(qd)
        ph = 2 * np.pi * t / self.t_hop
        v_hat = self.vel_est.update(q, qd, gyro_sim, in_stance=np.cos(ph) > 0)
        return np.concatenate([grav_sim, gyro_sim, q_sim, qd_sim,
                               v_hat, self.cmd, [self.h_cmd],
                               self.prev_act, [np.cos(ph), np.sin(ph)]])

    def _publish(self, q_des, thrust_n):
        cmd = hopper_cmd_lcmt()
        cmd.q_des = [float(v) for v in q_des]
        cmd.qd_des = [0.0] * 3
        cmd.tau_ff = [0.0] * 3
        cmd.kp_joint = [KP_JOINT] * 3
        cmd.kd_joint = [KD_JOINT] * 3
        self.lc.publish("hopper_cmd_lcmt", cmd.encode())
        pwm = motor_pwm_lcmt()
        pwm.timestamp = int(time.time() * 1e6)
        pwm.pwm_values = [float(v) for v in thrust_to_pwm(thrust_n)]
        pwm.roll_error = pwm.pitch_error = pwm.roll_output = pwm.pitch_output = 0.0
        pwm.control_mode = 1
        self.lc.publish("motor_pwm_lcmt", pwm.encode())

    def _publish_safe(self):
        """damping-only joints, props off."""
        cmd = hopper_cmd_lcmt()
        cmd.q_des = [0.0] * 3; cmd.qd_des = [0.0] * 3; cmd.tau_ff = [0.0] * 3
        cmd.kp_joint = [0.0] * 3; cmd.kd_joint = [KD_JOINT] * 3
        self.lc.publish("hopper_cmd_lcmt", cmd.encode())
        pwm = motor_pwm_lcmt()
        pwm.timestamp = int(time.time() * 1e6)
        pwm.pwm_values = [PWM_MIN] * 6
        pwm.roll_error = pwm.pitch_error = pwm.roll_output = pwm.pitch_output = 0.0
        pwm.control_mode = -1
        self.lc.publish("motor_pwm_lcmt", pwm.encode())

    def run(self):
        th = threading.Thread(target=self._lcm_loop, daemon=True)
        th.start()
        print("waiting for sensor messages...")
        while time.time() - max(self.t_data, self.t_imu) > 1.0 or self.t_data == 0 or self.t_imu == 0:
            time.sleep(0.05)
        print("sensors live, starting 100 Hz policy loop (Ctrl-C to stop)")
        self.t0 = time.time()
        next_t = self.t0
        try:
            while True:
                now = time.time()
                t = now - self.t0
                if now - self.t_data > STALE_S or now - self.t_imu > STALE_S:
                    self._publish_safe()
                else:
                    if self.hist is None:
                        f = self._frame(t)
                        self.hist = np.tile(f, HW_HIST)
                    else:
                        self.hist = np.concatenate([self.hist[HW_FRAME:], self._frame(t)])
                    act = np.clip(self.policy(self.hist), -1.0, 1.0)
                    ramp = min(1.0, t / RAMP_S)
                    act = ramp * act
                    self.prev_act = act
                    q_des = dm.action_to_q_des_real(act[0:3])
                    thrust = 0.5 * THRUST_MAX * (act[3:6] + 1.0) * ramp
                    self._publish(q_des, thrust)
                next_t += CTRL_DT
                time.sleep(max(0.0, next_t - time.time()))
        except KeyboardInterrupt:
            pass
        finally:
            for _ in range(10):
                self._publish_safe()
                time.sleep(0.01)
            print("stopped: joints in damping, props off")

    def _lcm_loop(self):
        while True:
            self.lc.handle_timeout(100)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--params", default=os.path.join(POLICIES_DIR, "hop_policy_hwcal.params"))
    ap.add_argument("--vx", type=float, default=0.0, help="velocity command x (m/s), 0 = hop in place")
    ap.add_argument("--vy", type=float, default=0.0, help="velocity command y (m/s)")
    ap.add_argument("--hop-h", type=float, default=0.08, help="hop apex height command (m above stance, 0.03-0.15)")
    a = ap.parse_args()
    runner = RLRunner(load_policy(a.params), cmd=(a.vx, a.vy), h_cmd=a.hop_h)
    runner.run()
