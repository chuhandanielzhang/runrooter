"""
Vectorized stochastic SRB rollout for PAC-NMPC.

Prediction model = the same SRB abstraction used by the runtime stack
(core.py / CASE paper Eq. 1-3), extended with SAMPLED uncertainty so that a
batch of M rollouts sees M different "possible worlds":

  - ground height offset  (multi-modal touchdown timing — the key contact effect)
  - friction coefficient
  - mass / thrust scaling
  - process noise on v and omega

State  (per sample): x = [px, py, pz, vx, vy, vz, roll, pitch, wx, wy]   (10)
Input  (shared cmd) : u[k] = [fx, fy, fz, t_red, t_green, t_blue]        (6)

Contact is NOT scheduled: each sample triggers touchdown when its own foot
height crosses its own sampled ground height (spring-less anchor model), and
lifts off when the vertical velocity is upward and the leg is back at length.
This is what makes the resulting uncertainty over trajectories multi-modal.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .conventions import (
    MASS_KG, GRAVITY, I_BODY_DIAG, LEG_L0_M, PROP_ARM_POS_B, COM_B, MU_FRICTION,
)


@dataclass
class RolloutUncertainty:
    """Sampling distribution the CONTROLLER assumes (ideally fit from real logs)."""
    ground_z_std: float = 0.03        # m; multi-modal touchdown timing
    friction_mu_mean: float = MU_FRICTION
    friction_mu_std: float = 0.1
    mass_scale_std: float = 0.05
    thrust_scale_std: float = 0.08
    v_noise_std: float = 0.02         # per-step process noise (m/s)
    w_noise_std: float = 0.05         # per-step process noise (rad/s)

    def sample(self, M: int, rng: np.random.Generator) -> dict:
        return {
            "ground_z": rng.normal(0.0, self.ground_z_std, M),
            "mu": np.clip(rng.normal(self.friction_mu_mean, self.friction_mu_std, M), 0.05, 1.5),
            "mass": MASS_KG * np.clip(rng.normal(1.0, self.mass_scale_std, M), 0.7, 1.4),
            "thrust_scale": np.clip(rng.normal(1.0, self.thrust_scale_std, M), 0.5, 1.5),
        }

    def zero(self, M: int) -> dict:
        """Nominal (deterministic-model) samples — used by the baseline."""
        return {
            "ground_z": np.zeros(M),
            "mu": np.full(M, self.friction_mu_mean),
            "mass": np.full(M, MASS_KG),
            "thrust_scale": np.ones(M),
        }


@dataclass
class RolloutCostConfig:
    z_apex_des: float = 0.664         # CASE paper h_des
    w_z: float = 1.0
    w_vz: float = 0.1
    # stance push-off: track vz_ref = sqrt(2 g (z_apex - z)) while in contact,
    # same spirit as the runtime MIT-MPC's vz takeoff ramp (core.py mpc_xref).
    w_push: float = 12.0
    w_att: float = 80.0               # roll^2 + pitch^2
    w_w: float = 1.5
    w_vxy: float = 2.0
    w_u_f: float = 2e-5
    w_u_t: float = 2e-4
    w_du: float = 5e-4                # control-rate smoothness
    # constraint: attitude envelope (violation => "failure" event for PAC)
    att_limit_rad: float = np.deg2rad(25.0)
    fz_max: float = 200.0
    t_max_each: float = 10.0


class SRBBatchRollout:
    """Batched rollout of one control sequence under M sampled worlds."""

    def __init__(self, dt: float = 0.02, horizon: int = 20,
                 unc: RolloutUncertainty | None = None,
                 cost: RolloutCostConfig | None = None):
        self.dt = float(dt)
        self.N = int(horizon)
        self.unc = unc or RolloutUncertainty()
        self.cost = cost or RolloutCostConfig()
        self.Ixx, self.Iyy = float(I_BODY_DIAG[0]), float(I_BODY_DIAG[1])
        # rotor moment arms about COM (body frame; thrust along body +Z)
        self.r_props = PROP_ARM_POS_B - COM_B.reshape(1, 3)

    def rollout(
        self,
        x0: np.ndarray,               # (10,) current SRB state estimate
        U: np.ndarray,                # (S, N, 6) candidate control sequences
        v_xy_des: np.ndarray,         # (2,)
        in_contact0: bool,
        worlds: dict,                 # sampled uncertainty dict, arrays of size M
        rng: np.random.Generator,
        process_noise: bool = True,
        r_foot0_w: np.ndarray | None = None,  # actual foot pos about COM (world) if in contact
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluate S control sequences, each under M sampled worlds.

        Returns:
          J    (S, M): trajectory costs (normalized to [0, 1])
          Viol (S, M): 1.0 where the attitude constraint was violated, else 0.0
        """
        S, N, _ = U.shape
        M = len(worlds["ground_z"])
        dt = self.dt
        c = self.cost

        # broadcast state: (S, M, ...)
        x = np.broadcast_to(x0.reshape(1, 1, 10), (S, M, 10)).copy()
        contact = np.full((S, M), bool(in_contact0))
        # foot anchor r_foot (world, about COM): use the MEASURED anchor for the
        # current stance (model-mismatch killer), nominal straight-down for future
        # touchdowns inside the horizon.
        r_foot = np.zeros((S, M, 3))
        if in_contact0 and r_foot0_w is not None:
            r_foot[:] = np.asarray(r_foot0_w, dtype=float).reshape(1, 1, 3)
        else:
            r_foot[..., 2] = -LEG_L0_M

        mass = worlds["mass"].reshape(1, M)
        mu = worlds["mu"].reshape(1, M)
        gz = worlds["ground_z"].reshape(1, M)
        tsc = worlds["thrust_scale"].reshape(1, M)

        J = np.zeros((S, M))
        viol = np.zeros((S, M))

        for k in range(N):
            u = U[:, k, :]                                   # (S, 6)
            f = np.broadcast_to(u[:, None, 0:3], (S, M, 3)).copy()
            t = np.clip(u[:, None, 3:6], 0.0, c.t_max_each) * tsc[..., None]  # (S, M, 3)

            # ---- contact events (per sample) ----
            foot_z = x[..., 2] - LEG_L0_M                    # foot height (leg extended)
            touchdown = (~contact) & (foot_z <= gz) & (x[..., 5] < 0.0)
            contact = contact | touchdown
            # foot anchor kept at the nominal straight-down arm (-l0); sampled ground
            # height differences are cm-scale, so the moment-arm error is negligible,
            # but the TOUCHDOWN TIMING difference (which we do capture) is what matters.
            liftoff = contact & (x[..., 5] > 0.05) & (foot_z > gz + 0.005)
            contact = contact & (~liftoff)

            # ---- enforce physical contact force limits per sample ----
            f[~contact] = 0.0
            fz = np.clip(f[..., 2], 0.0, c.fz_max)
            fxy_norm = np.linalg.norm(f[..., 0:2], axis=-1) + 1e-9
            fxy_max = mu * fz
            scale = np.minimum(1.0, fxy_max / fxy_norm)
            f[..., 0] *= scale
            f[..., 1] *= scale
            f[..., 2] = fz

            # ---- dynamics (small-angle attitude, thrust along body z) ----
            roll, pitch = x[..., 6], x[..., 7]
            z_body = np.stack(
                [-np.sin(pitch), np.sin(roll) * np.cos(pitch), np.cos(roll) * np.cos(pitch)],
                axis=-1,
            )
            T_sum = t.sum(axis=-1)
            a = (f + z_body * T_sum[..., None]) / mass[..., None]
            a[..., 2] -= GRAVITY

            tau_c = np.cross(r_foot, f)                       # contact moment
            tau_p = np.einsum("pj,smp->smj", np.cross(self.r_props, np.array([0.0, 0.0, 1.0])), t)
            alpha_x = (tau_c[..., 0] + tau_p[..., 0]) / self.Ixx
            alpha_y = (tau_c[..., 1] + tau_p[..., 1]) / self.Iyy

            x[..., 0:3] += dt * x[..., 3:6]
            x[..., 3:6] += dt * a
            x[..., 6] += dt * x[..., 8]
            x[..., 7] += dt * x[..., 9]
            x[..., 8] += dt * alpha_x
            x[..., 9] += dt * alpha_y

            if process_noise:
                x[..., 3:6] += rng.normal(0.0, self.unc.v_noise_std, (S, M, 3))
                x[..., 8:10] += rng.normal(0.0, self.unc.w_noise_std, (S, M, 2))

            # ---- stage cost ----
            J += c.w_z * (x[..., 2] - c.z_apex_des) ** 2
            J += c.w_vz * x[..., 5] ** 2
            # push-off tracking (contact samples only): drive vz toward the
            # energy-consistent takeoff ramp instead of hovering
            vz_ref = np.sqrt(2.0 * GRAVITY * np.clip(c.z_apex_des - x[..., 2], 0.0, None))
            J += c.w_push * contact * (x[..., 5] - vz_ref) ** 2
            J += c.w_att * (x[..., 6] ** 2 + x[..., 7] ** 2)
            J += c.w_w * (x[..., 8] ** 2 + x[..., 9] ** 2)
            J += c.w_vxy * ((x[..., 3] - v_xy_des[0]) ** 2 + (x[..., 4] - v_xy_des[1]) ** 2)
            J += c.w_u_f * (f ** 2).sum(axis=-1) + c.w_u_t * (t ** 2).sum(axis=-1)
            if k > 0:
                du = U[:, k, :] - U[:, k - 1, :]
                J += c.w_du * np.broadcast_to((du ** 2).sum(axis=-1)[:, None], (S, M))

            # ---- constraint violation (attitude envelope) ----
            bad = (np.abs(x[..., 6]) > c.att_limit_rad) | (np.abs(x[..., 7]) > c.att_limit_rad)
            viol = np.maximum(viol, bad.astype(float))

        # normalize cost to [0,1] for PAC bound arithmetic
        J = J / self.N
        J = J / (1.0 + J)
        return J, viol
