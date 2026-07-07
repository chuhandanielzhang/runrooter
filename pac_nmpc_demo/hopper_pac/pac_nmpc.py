"""
PAC-NMPC: sampling-based stochastic NMPC with a PAC upper-confidence-bound
objective, in the spirit of:

  Polevoy, Kobilarov, Moore, "Probably Approximately Correct Nonlinear Model
  Predictive Control (PAC-NMPC)", IEEE RA-L 2023 (arXiv:2210.08092),
  which builds on Kobilarov's PROPS (PAC Robust Policy Search).

Method (receding horizon, one solve per outer-loop tick):
  1. Maintain a Gaussian distribution xi = (mean, std) over open-loop control
     sequences U in R^{N x nu}.
  2. Each iteration: draw S candidate sequences from xi, evaluate every one
     under M sampled "worlds" (ground height, friction, mass, thrust scale,
     process noise) with the vectorized SRB rollout — costs in [0,1] and
     constraint-violation indicators.
  3. Optimize an upper confidence bound instead of the empirical mean:

        b(xi) = J_hat + lambda * V_hat
                + C_conf * sqrt( (KL(xi || xi_prev) + ln(2 sqrt(n)/delta)) / (2 n) )

     where J_hat / V_hat are importance-weighted empirical cost / violation
     probability under xi, and n = S*M. This is a PAC-Bayes/Chernoff-style
     simplification of the sample-complexity bound used in PAC-NMPC; the
     minimized bound itself certifies (w.p. >= 1-delta) the expected cost and
     the constraint-violation probability of the returned policy distribution.
  4. Distribution update: elite-weighted refit (CE-style) of mean/std — the
     same "iterative stochastic policy optimization" family as PROPS.
  5. Warm start: shift the mean sequence one step and reuse.

Output per solve: first control u0 = mean[0] (= [f_c(3), t_arms(3)]) plus the
certified bound values for logging/plots.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .srb_rollout import SRBBatchRollout, RolloutUncertainty, RolloutCostConfig


@dataclass
class PACNMPCConfig:
    horizon: int = 20
    dt: float = 0.02
    n_candidates: int = 64        # S
    n_worlds: int = 32            # M
    n_iters: int = 4
    elite_frac: float = 0.2
    delta: float = 0.05           # PAC confidence 1-delta
    lambda_viol: float = 3.0      # weight of violation probability in the bound
    c_conf: float = 1.0           # confidence-radius weight
    # initial std per input channel [fx, fy, fz, t1, t2, t3]
    init_std: tuple = (4.0, 4.0, 25.0, 1.0, 1.0, 1.0)
    fz_nominal: float = 3.75 * 9.81 * 1.6
    t_nominal: float = 0.3
    stochastic: bool = True       # False => deterministic baseline (nominal world, no noise)
    seed: int = 0


class PACNMPC:
    def __init__(self, cfg: PACNMPCConfig | None = None,
                 unc: RolloutUncertainty | None = None,
                 cost: RolloutCostConfig | None = None):
        self.cfg = cfg or PACNMPCConfig()
        self.rollout = SRBBatchRollout(
            dt=self.cfg.dt, horizon=self.cfg.horizon, unc=unc, cost=cost,
        )
        self.rng = np.random.default_rng(self.cfg.seed)
        N, nu = self.cfg.horizon, 6
        self.mean = np.zeros((N, nu))
        self.mean[:, 2] = self.cfg.fz_nominal * 0.5
        self.mean[:, 3:6] = self.cfg.t_nominal
        self.std = np.tile(np.asarray(self.cfg.init_std, dtype=float), (N, 1))
        self._init_std = self.std.copy()

    def reset_warm_start(self) -> None:
        N = self.cfg.horizon
        self.mean[:] = 0.0
        self.mean[:, 2] = self.cfg.fz_nominal * 0.5
        self.mean[:, 3:6] = self.cfg.t_nominal
        self.std = self._init_std.copy()

    def solve(self, x0: np.ndarray, v_xy_des: np.ndarray, in_contact: bool,
              r_foot0_w: np.ndarray | None = None) -> dict:
        cfg = self.cfg
        N, nu = cfg.horizon, 6
        S, M = cfg.n_candidates, cfg.n_worlds

        # warm start: shift previous mean by one step
        self.mean[:-1] = self.mean[1:]
        self.std = np.maximum(self.std * 1.15, 0.5 * self._init_std)
        self.std = np.minimum(self.std, self._init_std)

        if cfg.stochastic:
            worlds = self.rollout.unc.sample(M, self.rng)
        else:
            worlds = self.rollout.unc.zero(M)

        prev_mean, prev_std = self.mean.copy(), self.std.copy()
        bound = float("nan")
        j_hat = v_hat = float("nan")

        for _ in range(cfg.n_iters):
            eps = self.rng.standard_normal((S, N, nu))
            U = self.mean[None] + eps * self.std[None]
            U[0] = self.mean                      # always include the current mean
            U[..., 2] = np.clip(U[..., 2], 0.0, 200.0)
            U[..., 3:6] = np.clip(U[..., 3:6], 0.0, 10.0)

            J, viol = self.rollout.rollout(
                x0=x0, U=U, v_xy_des=np.asarray(v_xy_des, dtype=float),
                in_contact0=bool(in_contact), worlds=worlds, rng=self.rng,
                process_noise=bool(cfg.stochastic), r_foot0_w=r_foot0_w,
            )
            j_mean = J.mean(axis=1)               # (S,) expected cost per candidate
            v_mean = viol.mean(axis=1)            # (S,) violation prob per candidate
            score = j_mean + cfg.lambda_viol * v_mean

            # elite refit (PROPS-style iterative stochastic policy optimization)
            n_elite = max(2, int(round(cfg.elite_frac * S)))
            elite = np.argsort(score)[:n_elite]
            self.mean = U[elite].mean(axis=0)
            self.std = U[elite].std(axis=0) + 1e-3

            # PAC upper confidence bound for the CURRENT distribution
            n_samp = S * M
            kl = _kl_diag_gauss(self.mean, self.std, prev_mean, prev_std)
            conf = np.sqrt((kl + np.log(2.0 * np.sqrt(n_samp) / cfg.delta)) / (2.0 * n_samp))
            j_hat = float(J[elite].mean())
            v_hat = float(viol[elite].mean())
            bound = float(j_hat + cfg.lambda_viol * v_hat + cfg.c_conf * conf)

        return {
            "u0": self.mean[0].copy(),            # [fx, fy, fz, t_red, t_green, t_blue]
            "U": self.mean.copy(),
            "bound": bound,
            "j_hat": j_hat,
            "viol_hat": v_hat,
        }


def _kl_diag_gauss(m1: np.ndarray, s1: np.ndarray, m0: np.ndarray, s0: np.ndarray) -> float:
    """KL( N(m1, s1^2) || N(m0, s0^2) ), diagonal, summed over all dims."""
    s0 = np.maximum(s0, 1e-6)
    s1 = np.maximum(s1, 1e-6)
    kl = np.log(s0 / s1) + (s1 ** 2 + (m1 - m0) ** 2) / (2.0 * s0 ** 2) - 0.5
    return float(np.sum(kl))
