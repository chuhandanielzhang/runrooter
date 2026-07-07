"""
Leg-driven hopping RL for the closed-loop 3-RSR aerial-legged robot (MJX + brax PPO).

Stage A (this script): the LEG is the main role. Policy commands only the 3 hip
motors; the 3 tilt servos and 3 rotor thrusts are frozen at home (added later).
Closed loop (equality connect) is simulated natively in MJX (the BRUCE / TopA 2025
approach), NOT a serial approximation.

Run (after `pip install "jax[cuda12]" brax mujoco-mjx`):
    python train_hop.py --timesteps 20_000_000
"""
import argparse, functools, time
import jax, jax.numpy as jp
import numpy as np

# Ampere GPUs (RTX 30xx) default to TF32 matmul which destabilizes RL training
# (mujoco_playground issue #86) -> force full float32 precision.
jax.config.update("jax_default_matmul_precision", "highest")

# --- compat shim: brax 0.14.2 uses jax.device_put_replicated, removed in jax 0.10+ ---
if not hasattr(jax, "device_put_replicated"):
    def _device_put_replicated(x, devices):
        n = len(devices)
        x = jax.tree_util.tree_map(
            lambda a: jp.broadcast_to(jp.asarray(a)[None], (n,) + jp.shape(a)), x)
        if n == 1:
            return jax.device_put(x, devices[0])
        return jax.device_put_sharded(
            [jax.tree_util.tree_map(lambda a, i=i: a[i], x) for i in range(n)], devices)
    jax.device_put_replicated = _device_put_replicated

import mujoco
from mujoco import mjx
from brax.envs.base import Env, State
from brax.training.agents.ppo import train as ppo

XML = "three_leg_3rsr_closed.xml"

# qpos/qvel layout (nq=26, nv=24):
#   base free  : qpos[0:7]   qvel[0:6]
#   leg joints : qpos[7:16]  = [hip1,cross1,lower1, hip2,..., hip3,...]   (hip at 7,10,13)
#   servos     : qpos[16:19] = [servo1,servo2,servo3]
#   foot free  : qpos[19:26] qvel[18:24]
HIP_QPOS = jp.array([7, 10, 13])
HIP_QVEL = jp.array([6, 9, 12])
HIP_HOME = -0.060299
BASE_NOMINAL_Z = 0.6

# observation noise scales (matched to obs layout below)
OBS_NOISE = jp.concatenate([
    jp.array([0.01]),          # base_z
    jp.full((3,), 0.02),       # projected gravity
    jp.full((3,), 0.05),       # base linvel (estimated on hardware -> noisy)
    jp.full((3,), 0.10),       # base angvel (gyro)
    jp.full((3,), 0.01),       # hip pos
    jp.full((3,), 0.15),       # hip vel
    jp.array([0.01]),          # foot_z
])
PUSH_PROB = 0.005              # per ctrl step -> ~2-3 pushes per 5 s episode
PUSH_VEL = 0.4                 # m/s kick magnitude

# ---- hardware-observable mode (deployable obs, like the OMEGA paper Eq.8) ----
# single frame: grav(3) gyro(3) q(3) dq(3) v_hat_xy(2) cmd_xy(2) h_cmd(1)
#               prev_action(6) phase(2) = 25
# v_hat_xy emulates the Cao-style velocity estimator (stance leg-kinematics + IMU
# hold in flight); in sim we use true base velocity + heavy noise as its stand-in.
# h_cmd = commanded hop apex height above standing (RSS2023 Cassie-style
# command conditioning: sample per episode, feed to obs, reward tracking).
ACT_DIM = 9                    # 3 hips + 3 prop thrusts + 3 tilt servos (thrust vectoring)
THRUST_MAX = 6.0               # N per prop; 3*6=18 N = 49% weight (assist only, leg dominates)
SERVO_SCALE = 0.6              # rad, max tilt command magnitude (thrust vectoring authority)
SERVO_RATE = 6.5               # rad/s, DS3218MG slew limit (0.16 s/60deg @ 6.8V)
TM_NOM = 0.04                  # s, first-order motor (thrust) time constant (SimpleFlight)
YAW_RATE_MAX = 1.0             # rad/s, commanded yaw-rate range (servo thrust-vectoring yaw)
HW_FRAME = 20 + ACT_DIM        # +1 vs before for the yaw-rate command
HW_HIST = 5                    # paper uses H=5 history
CMD_MAX = 1.2                  # m/s, velocity command range (fast forward hopping)
CMD_RESAMPLE_S = 2.5           # mean seconds between in-episode command switches
Z_STAND = 0.40                 # base height at stance (observed hopping baseline)
H_CMD_MIN, H_CMD_MAX = 0.03, 0.30   # commanded apex above Z_STAND (0.30 = high-jump mode)
MG = 3.75 * 9.81               # robot weight [N], for contact-force normalization
HW_NOISE = jp.concatenate([
    jp.full((3,), 0.02),       # projected gravity (from quat)
    jp.full((3,), 0.10),       # gyro
    jp.full((3,), 0.01),       # q
    jp.full((3,), 0.15),       # dq
    jp.full((2,), 0.10),       # v_hat (estimator is imperfect -> heavy noise)
    jp.zeros(4),               # cmd_xy + h_cmd + yaw_rate cmd (exact)
    jp.zeros(ACT_DIM),         # prev action (exact)
    jp.zeros(2),               # phase (exact)
])


def hop_period(h_cmd):
    """gait period scales with commanded height: stance (~0.16 s) + ballistic flight."""
    return 0.16 + 2.0 * jp.sqrt(2.0 * h_cmd / 9.81)


HOP_PERIOD = 0.4               # legacy constant (eval/runner default, h=0.06 -> ~0.38 s)


def domain_randomize(sys, rng):
    """Model-level DR (vmapped over envs), outdoor-grade ranges per the jumping
    literature (Cassie RSS2023, SF-TIM): absolute friction 0.25-1.25 (covers
    slippery/wet ground), mass +-15%, hip PD +-20/30%, damping +-30%, CoM shift."""
    @jax.vmap
    def rand(key):
        k1, k2, k3, k4, k5 = jax.random.split(key, 5)
        # absolute sliding friction (not a scale): 0.25 = wet smooth floor
        friction = sys.geom_friction.at[:, 0].set(
            jax.random.uniform(k1, (), minval=0.25, maxval=1.25))
        mass_scale = jax.random.uniform(k2, (), minval=0.85, maxval=1.15)
        body_mass = sys.body_mass * mass_scale
        body_inertia = sys.body_inertia * mass_scale
        k3, k3b = jax.random.split(k3)
        gain_scale = jax.random.uniform(k3, (), minval=0.8, maxval=1.2)
        gainprm = sys.actuator_gainprm.at[0:3, 0].multiply(gain_scale)
        biasprm = sys.actuator_biasprm.at[0:3, 1].multiply(gain_scale)
        # thrust coefficient k_f is the sim2real-sensitive param (SimpleFlight) ->
        # DR it +-15% on the 3 thrust actuators (indices 6,7,8)
        thr_scale = jax.random.uniform(k3b, (), minval=0.85, maxval=1.15)
        gainprm = gainprm.at[6:9, 0].multiply(thr_scale)
        dof_damping = sys.dof_damping * jax.random.uniform(k4, (), minval=0.7, maxval=1.3)
        # base CoM shift +-2 cm (payload/battery placement uncertainty)
        body_ipos = sys.body_ipos.at[1, 0:2].add(
            jax.random.uniform(k5, (2,), minval=-0.02, maxval=0.02))
        return friction, body_mass, body_inertia, gainprm, biasprm, dof_damping, body_ipos

    friction, body_mass, body_inertia, gainprm, biasprm, dof_damping, body_ipos = rand(rng)
    in_axes = jax.tree_util.tree_map(lambda x: None, sys)
    in_axes = in_axes.tree_replace({
        "geom_friction": 0, "body_mass": 0, "body_inertia": 0,
        "actuator_gainprm": 0, "actuator_biasprm": 0, "dof_damping": 0, "body_ipos": 0,
    })
    sys = sys.tree_replace({
        "geom_friction": friction, "body_mass": body_mass, "body_inertia": body_inertia,
        "actuator_gainprm": gainprm, "actuator_biasprm": biasprm,
        "dof_damping": dof_damping, "body_ipos": body_ipos,
    })
    return sys, in_axes


class HopEnv(Env):
    def __init__(self, ctrl_dt=0.01, action_scale=0.80,
                 fall_z=0.24, tilt_limit=0.6, randomize=False, hw_obs=False):
        self._hw = hw_obs
        self._ctrl_dt = ctrl_dt
        self._mj = mujoco.MjModel.from_xml_path(XML)
        self._mjx = mjx.put_model(self._mj)
        self._n_sub = max(1, int(round(ctrl_dt / self._mj.opt.timestep)))
        self._home_q = jp.array(self._mj.key_qpos[0])
        self._home_ctrl = jp.array(self._mj.key_ctrl[0])
        self._action_scale = action_scale
        self._fall_z = fall_z
        self._tilt_limit = tilt_limit
        self._randomize = randomize

    # brax's DomainRandomizationVmapWrapper swaps the model through env.sys
    @property
    def sys(self):
        return self._mjx

    @sys.setter
    def sys(self, new_sys):
        self._mjx = new_sys

    # ---- helpers ----
    def _obs(self, data) -> jp.ndarray:
        q, v = data.qpos, data.qvel
        base_quat = q[3:7]
        # projected gravity in base frame (upright -> ~[0,0,-1])
        w, x, y, z = base_quat
        grav = jp.array([
            2*(x*z - w*y),
            2*(y*z + w*x),
            1 - 2*(x*x + y*y),
        ])
        hip_pos = q[HIP_QPOS] - HIP_HOME
        hip_vel = v[HIP_QVEL]
        base_z = q[2:3]
        base_linvel = v[0:3]
        base_angvel = v[3:6]
        foot_z = q[21:22]            # foot free body z
        return jp.concatenate([base_z, grav, base_linvel, base_angvel,
                               hip_pos, hip_vel, foot_z])

    def _hw_frame(self, data, prev_action, t, cmd, h_cmd, yaw_cmd) -> jp.ndarray:
        """One frame of HARDWARE-AVAILABLE signals only (IMU + encoders + clock +
        velocity estimate + commands). Matches the real interface:
        hopper_imu_lcmt (quat/gyro) + hopper_data_lcmt (q/qd) + Cao-style v_hat."""
        q, v = data.qpos, data.qvel
        w, x, y, z = q[3], q[4], q[5], q[6]
        grav = jp.array([2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)])
        gyro = v[3:6]                      # freejoint angvel = body frame, same as IMU gyro
        # body-frame horizontal velocity (what a leg-kinematics estimator measures)
        R_T = jp.array([
            [1-2*(y*y+z*z), 2*(x*y+z*w), 2*(x*z-y*w)],
            [2*(x*y-z*w), 1-2*(x*x+z*z), 2*(y*z+x*w)],
        ])
        v_hat = R_T @ v[0:3]
        phase = 2 * jp.pi * t / hop_period(h_cmd)
        return jp.concatenate([grav, gyro, q[HIP_QPOS] - HIP_HOME, v[HIP_QVEL],
                               v_hat, cmd, h_cmd[None], yaw_cmd[None], prev_action,
                               jp.array([jp.cos(phase), jp.sin(phase)])])

    def reset(self, rng) -> State:
        rng, k1, k2, k3, k4, k5, k6, k7, k8, k9 = jax.random.split(rng, 10)
        q = self._home_q
        q = q.at[2].add(0.02 * jax.random.normal(k1))            # base height jitter
        q = q.at[HIP_QPOS].add(0.05 * jax.random.normal(k2, (3,)))
        data = mjx.make_data(self._mjx).replace(qpos=q, ctrl=self._home_ctrl)
        data = mjx.forward(self._mjx, data)
        # velocity command: 60% hop in place, 40% follow a random v_des
        zero_cmd = jax.random.uniform(k4) < 0.6
        cmd = jp.where(zero_cmd, jp.zeros(2),
                       jax.random.uniform(k5, (2,), minval=-CMD_MAX, maxval=CMD_MAX))
        # hop height command: per-episode apex target (command conditioning)
        h_cmd = jax.random.uniform(k6, (), minval=H_CMD_MIN, maxval=H_CMD_MAX)
        # yaw-rate command: 50% hold heading (0), 50% turn at a random rate
        yaw_cmd = jp.where(jax.random.uniform(k7) < 0.5, jp.float32(0.0),
                           jax.random.uniform(k8, (), minval=-YAW_RATE_MAX, maxval=YAW_RATE_MAX))
        # per-episode motor time constant (mild DR around TM_NOM); alpha = 1-e^{-dt/Tm}
        tm = jax.random.uniform(k3, (), minval=0.02, maxval=0.06) if self._randomize \
            else jp.float32(TM_NOM)
        alpha = 1.0 - jp.exp(-self._ctrl_dt / tm)
        metrics = {"base_z": q[2], "upright": jp.float32(1.0), "reward_air": jp.float32(0.0)}
        info = {
            "rng": rng,
            "prev_action": jp.zeros(ACT_DIM),
            "prev_action2": jp.zeros(ACT_DIM),
            "prev_contact": jp.float32(1.0),
            "prev_f": jp.float32(MG),
            "thr": jp.zeros(3),               # motor-lag thrust state [N]
            "servo": jp.zeros(3),             # slew-limited servo angle state [rad]
            "alpha": alpha,
            "t": jp.float32(0.0),
            "cmd": cmd,
            "h_cmd": h_cmd,
            "yaw_cmd": yaw_cmd,
            # per-episode one-step action delay (simulates control latency), 50% of envs
            "delay": (jax.random.uniform(k9) < 0.5).astype(jp.float32) if self._randomize
                     else jp.float32(0.0),
        }
        if self._hw:
            frame = self._hw_frame(data, jp.zeros(ACT_DIM), jp.float32(0.0), cmd, h_cmd, yaw_cmd)
            info["hist"] = jp.tile(frame, HW_HIST)
            obs = info["hist"]
        else:
            obs = self._obs(data)
        return State(data, obs, jp.float32(0), jp.float32(0), metrics, info)

    def step(self, state: State, action: jp.ndarray) -> State:
        action = jp.clip(action, -1.0, 1.0)
        info = dict(state.info)
        rng, k_push, k_kick, k_noise, k_rs, k_rs2, k_rs3 = jax.random.split(info["rng"], 7)
        info["rng"] = rng

        # in-episode command resampling: forces the policy to learn hard
        # accelerations and sudden stops (RSS2023 task randomization)
        resample = jax.random.uniform(k_rs) < (self._ctrl_dt / CMD_RESAMPLE_S)
        new_zero = jax.random.uniform(k_rs2) < 0.4
        new_cmd = jp.where(new_zero, jp.zeros(2),
                           jax.random.uniform(k_rs3, (2,), minval=-CMD_MAX, maxval=CMD_MAX))
        info["cmd"] = jp.where(resample, new_cmd, info["cmd"])

        applied = jp.where(info["delay"] > 0.5, info["prev_action"], action)
        a_prev, a_prev2 = info["prev_action"], info["prev_action2"]
        info["prev_action2"] = info["prev_action"]
        info["prev_action"] = action

        hip_cmd = HIP_HOME + self._action_scale * applied[0:3]
        # thrust target [-1,1] -> [0, THRUST_MAX] N, then first-order MOTOR LAG
        # (SimpleFlight): props can't change thrust instantly -> key sim2real term
        thrust_tgt = 0.5 * THRUST_MAX * (applied[3:6] + 1.0)
        thr = info["thr"] + info["alpha"] * (thrust_tgt - info["thr"])
        info["thr"] = thr
        # tilt servos = thrust vectoring (roll/pitch/yaw authority in flight), with a
        # SLEW-RATE LIMIT matching the real DS3218MG (can't snap instantly) -> the
        # policy can't rely on instant thrust-vectoring the hardware can't deliver
        servo_tgt = SERVO_SCALE * applied[6:9]
        dmax = SERVO_RATE * self._ctrl_dt
        servo_cmd = jp.clip(servo_tgt, info["servo"] - dmax, info["servo"] + dmax)
        info["servo"] = servo_cmd
        # actuator order: hips 0-2, servos 3-5, thrust 6-8
        ctrl = (self._home_ctrl.at[0:3].set(hip_cmd)
                .at[3:6].set(servo_cmd).at[6:9].set(thr))

        d0 = state.pipeline_state
        if self._randomize:
            # random base push (external disturbance)
            push = (jax.random.uniform(k_push) < PUSH_PROB).astype(jp.float32)
            kick = push * PUSH_VEL * jax.random.normal(k_kick, (3,))
            d0 = d0.replace(qvel=d0.qvel.at[0:3].add(kick))

        def f(d, _):
            return mjx.step(self._mjx, d.replace(ctrl=ctrl)), None
        data, _ = jax.lax.scan(f, d0, None, length=self._n_sub)

        q, v = data.qpos, data.qvel
        base_z = q[2]
        base_vz = v[2]
        # uprightness: projected gravity z (1 when upright)
        w, x, y, zq = q[3], q[4], q[5], q[6]
        up = 1 - 2*(x*x + y*y)

        f_touch = data.sensordata[-1]                            # foot touch sensor [N]
        in_contact = (f_touch > 1e-6).astype(jp.float32)
        foot_z = q[21]
        v_foot = v[18:21]                                        # foot free-body linear vel

        # --- hopping reward: sustained hops at the COMMANDED apex height ---
        h_cmd = info["h_cmd"]
        vz_des = jp.sqrt(2.0 * 9.81 * h_cmd)                     # takeoff speed for target apex
        r_alive = 1.0
        # push-off reward capped at the commanded takeoff speed (height controllability)
        r_air = 1.5 * jp.clip(base_vz, 0.0, vz_des)
        r_upright = 0.5 * up
        cmd = info["cmd"]
        # velocity-command tracking (cmd=0 -> hop strictly in place)
        v_err = v[0:2] - cmd
        r_vel = 1.5 * jp.exp(-4.0 * jp.sum(v_err ** 2))   # softer shaping for fast commands
        # Raibert foot placement: only when a velocity command is active. For
        # hop-in-place the foot must NOT be repositioned in the air (real robot
        # holds the leg at rest length), so this reward fades out near zero cmd.
        foot_offset = q[19:21] - q[0:2]                          # foot xy relative to base
        raibert_target = 0.15 * v[0:2] + 0.10 * v_err
        cmd_active = jp.clip(jp.sqrt(jp.sum(cmd ** 2)) / 0.2, 0.0, 1.0)
        r_place = 0.6 * cmd_active * (1.0 - in_contact) * jp.exp(-30.0 * jp.sum((foot_offset - raibert_target) ** 2))
        # position anchor only when commanded to stay (suppresses slow random-walk)
        zero_cmd = (jp.sum(cmd ** 2) < 1e-6).astype(jp.float32)
        c_pos = 0.10 * zero_cmd * jp.clip(jp.sum(q[0:2] ** 2), 0.0, 4.0)
        c_ctrl = 0.01 * jp.sum(action[0:3] ** 2)
        # penalize thrust usage so the LEG stays the primary actuator (props = assist only;
        # kept mild so props CAN help reach high commanded apexes)
        c_thrust = 0.05 * jp.sum(0.5 * (action[3:6] + 1.0))
        # yaw-rate tracking via servo thrust-vectoring (gyro_z follows command);
        # replaces the blanket spin penalty on the yaw axis with command tracking
        yaw_cmd = info["yaw_cmd"]
        r_yaw = 0.8 * jp.exp(-2.0 * (v[5] - yaw_cmd) ** 2)
        c_spin = 0.002 * jp.sum(v[3:5] ** 2)   # damp roll/pitch rate only (yaw is commanded)
        # --- SLIP-compliance: soft landing must come from STANCE COMPRESSION
        # (spring-damper buffering), not from leg tucking. A compressing leg
        # spreads the impulse -> low GRF peak and low GRF rise rate; we shape
        # exactly those two, so compression emerges as the only way to comply. ---
        # peak GRF: soft SLIP stance stays ~2-3x mg; penalize above 2.5x
        c_impact = 0.50 * jp.clip(f_touch / MG - 2.5, 0.0, 8.0)
        # GRF rise rate: a rigid leg shows a sharp spike at touchdown, a SLIP
        # spring loads up gradually over the compression stroke
        c_grf_rate = 0.05 * jp.clip((f_touch - info["prev_f"]) / MG, 0.0, 10.0)
        info["prev_f"] = f_touch
        c_chatter = 0.10 * jp.abs(in_contact - info["prev_contact"])
        info["prev_contact"] = in_contact
        c_slip = 0.10 * in_contact * jp.clip(jp.sum(v_foot[0:2] ** 2), 0.0, 4.0)
        # 2nd-order action smoothness (r_act in the flip paper) -> compliant motion
        c_smooth = 0.02 * jp.sum((action - 2.0 * a_prev + a_prev2) ** 2)
        # --- SLIP "rest length in air": the leg is a passive spring at its natural
        # length during flight, NOT an active limb. Real robot per-flight hip std
        # is ~4.5 deg; penalize flight-phase hip velocity AND active hip command
        # change so the leg locks at the push-off extension and only recompresses
        # on the next touchdown. ---
        c_air_hold = 0.04 * (1.0 - in_contact) * jp.sum(v[HIP_QVEL] ** 2)
        c_air_cmd = 0.10 * (1.0 - in_contact) * jp.sum((action[0:3] - a_prev[0:3]) ** 2)
        reward = (r_alive + r_air + r_upright + r_place + r_vel + r_yaw
                  - c_ctrl - c_thrust - c_spin - c_pos
                  - c_impact - c_grf_rate - c_chatter - c_slip - c_smooth
                  - c_air_hold - c_air_cmd)

        t_new = info["t"] + self._ctrl_dt
        if self._hw:
            # phase-schedule term: follow the height-dependent swing/stance clock
            phase = 2 * jp.pi * t_new / hop_period(h_cmd)
            want_stance = (jp.cos(phase) > 0).astype(jp.float32)
            r_phase = 0.3 * jp.where(want_stance > 0.5, in_contact, 1.0 - in_contact)
            # commanded flight arc: during the flight half the base should follow a
            # ballistic parabola peaking at Z_STAND + h_cmd (strong, dense height signal)
            s = jp.mod(phase - jp.pi / 2, 2 * jp.pi) / jp.pi      # 0..1 over flight half
            z_ref = Z_STAND + h_cmd * (1.0 - (2.0 * jp.clip(s, 0.0, 1.0) - 1.0) ** 2)
            r_track = 2.0 * (1.0 - want_stance) * jp.exp(-80.0 * (base_z - z_ref) ** 2)
            reward = reward + r_phase + r_track
        info["t"] = t_new

        fell = (base_z < self._fall_z) | (up < self._tilt_limit) | jp.isnan(base_z)
        done = fell.astype(jp.float32)

        if self._hw:
            frame = self._hw_frame(data, action, info["t"], cmd, h_cmd, info["yaw_cmd"])
            if self._randomize:
                frame = frame + HW_NOISE * jax.random.uniform(k_noise, frame.shape, minval=-1.0, maxval=1.0)
            info["hist"] = jp.concatenate([info["hist"][HW_FRAME:], frame])
            obs = info["hist"]
        else:
            obs = self._obs(data)
            if self._randomize:
                obs = obs + OBS_NOISE * jax.random.uniform(k_noise, obs.shape, minval=-1.0, maxval=1.0)
        state.metrics.update(base_z=base_z, upright=up, reward_air=r_air)
        return state.replace(pipeline_state=data, obs=obs, reward=reward, done=done, info=info)

    @property
    def observation_size(self):
        return HW_FRAME * HW_HIST if self._hw else 17

    @property
    def action_size(self):
        return ACT_DIM

    @property
    def backend(self):
        return "mjx"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=20_000_000)
    ap.add_argument("--num_envs", type=int, default=2048)
    ap.add_argument("--episode_len", type=int, default=500)
    ap.add_argument("--dr", action="store_true", help="enable domain randomization (sim2real)")
    ap.add_argument("--hw", action="store_true", help="hardware-observable obs only (IMU+q+dq, H=5 history)")
    ap.add_argument("--out", type=str, default="hop_policy.params")
    args = ap.parse_args()

    print("JAX backend:", jax.default_backend(), jax.devices())
    env = HopEnv(randomize=args.dr, hw_obs=args.hw)
    print(f"obs={env.observation_size} act={env.action_size} n_substeps={env._n_sub} DR={args.dr} HW={args.hw}")

    times = [time.time()]
    def progress(step, metrics):
        times.append(time.time())
        r = metrics.get("eval/episode_reward", float("nan"))
        print(f"[{step:>10,}] reward={r:8.3f}  ({times[-1]-times[-2]:.1f}s)")

    train_fn = functools.partial(
        ppo.train,
        num_timesteps=args.timesteps,
        num_evals=10,
        episode_length=args.episode_len,
        num_envs=args.num_envs,
        batch_size=256,
        num_minibatches=32,
        unroll_length=20,
        num_updates_per_batch=4,
        learning_rate=3e-4,
        entropy_cost=1e-2,
        discounting=0.97,
        normalize_observations=True,
        action_repeat=1,
        seed=0,
        randomization_fn=domain_randomize if args.dr else None,
    )
    make_inference_fn, params, _ = train_fn(environment=env, progress_fn=progress)

    from brax.io import model
    model.save_params(args.out, params)
    print("saved", args.out)


if __name__ == "__main__":
    main()
