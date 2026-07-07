"""
MIT lecture-style Convex MPC (Condensed QP) - GRF Only
======================================================

This follows the formulation described in /home/abc/Hopper/mit:
  - Optimize only ground reaction forces U (states are condensed out)
  - Discrete-time linear dynamics: x[k+1] = A x[k] + B u[k] + b
  - Objective:  ||A_qp x0 + B_qp U + xbar - x_ref||_L + ||U||_K
  - Constraints: friction pyramid + force bounds; flight steps force = 0

Adaptation for this project (single-foot hopper + tri-rotor baseline thrust):
  - n_foot = 1 -> u[k] ∈ R^3 (fx, fy, fz) in world frame
  - Baseline rotor thrust T_base is treated as a known external force along z_w
    and folded into the constant term b (so MPC remains GRF-only).

State x (13D, matching the lecture's 13k sizing):
  [p(3), v(3), rpy(3), omega(3), yaw_ref(1)]
Control u (3D per step):
  [fx, fy, fz]  (world)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any
import numpy as np

import osqp
import scipy.sparse as sp


def _skew(r: np.ndarray) -> np.ndarray:
    x, y, z = [float(v) for v in np.asarray(r, dtype=float).reshape(3)]
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)

def _Rz(theta: float) -> np.ndarray:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


@dataclass
class MITCondensedGRFMPCConfig:
    # horizon
    dt: float = 0.02
    N: int = 15

    # physical
    mu: float = 0.8
    fz_min: float = 10.0
    fz_max: float = 180.0

    # cost weights (diagonal W for each state in X ∈ R^{13N})
    w_px: float = 0.0
    w_py: float = 0.0
    w_pz: float = 60.0
    w_vx: float = 10.0
    w_vy: float = 10.0
    w_vz: float = 30.0
    w_roll: float = 80.0
    w_pitch: float = 80.0
    w_yaw: float = 0.0
    w_wx: float = 2.0
    w_wy: float = 2.0
    w_wz: float = 0.0
    w_yaw_ref: float = 0.0

    # force regularization (K = alpha I)
    alpha_u: float = 1e-4

    # solver
    max_iter: int = 2000
    eps_abs: float = 1e-4
    eps_rel: float = 1e-4

    # numeric bounds for fx/fy (keeps QP well-scaled)
    fxy_max: float = 120.0
    # additional physically-scaled bound: |fxy| <= fxy_max_ratio * (m*g)
    # This prevents "one-step hard brake" solutions that are feasible in SRB but tip the hopper at touchdown.
    fxy_max_ratio: float = 1.0


class MITCondensedGRFMPC:
    """
    Condensed-QP MPC that optimizes only GRF sequence U.
    """

    nx: int = 13
    nu: int = 3  # single foot

    def __init__(self, cfg: MITCondensedGRFMPCConfig | None = None):
        self.cfg = cfg or MITCondensedGRFMPCConfig()
        self._prob = osqp.OSQP()
        self._initialized = False
        self._last_contact_schedule: np.ndarray | None = None

        # cache
        self._H: sp.csc_matrix | None = None
        self._A: sp.csc_matrix | None = None
        self._l: np.ndarray | None = None
        self._u: np.ndarray | None = None

    def _weights_W(self) -> np.ndarray:
        c = self.cfg
        return np.array(
            [
                c.w_px,
                c.w_py,
                c.w_pz,
                c.w_vx,
                c.w_vy,
                c.w_vz,
                c.w_roll,
                c.w_pitch,
                c.w_yaw,
                c.w_wx,
                c.w_wy,
                c.w_wz,
                c.w_yaw_ref,
            ],
            dtype=float,
        )

    def _build_dynamics(
        self,
        *,
        m: float,
        g: float,
        I_body: np.ndarray,
        r_foot_w: np.ndarray,
        z_w: np.ndarray,
        T_base: float,
        yaw_ref: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build (A,B,b) for x[k+1] = A x[k] + B u[k] + b
        with u=[fx,fy,fz] in world frame.
        """
        dt = float(self.cfg.dt)
        nx, nu = self.nx, self.nu
        A = np.eye(nx, dtype=float)
        B = np.zeros((nx, nu), dtype=float)
        b = np.zeros(nx, dtype=float)

        m = float(max(1e-6, m))
        g = float(g)
        z_w = np.asarray(z_w, dtype=float).reshape(3)
        r = np.asarray(r_foot_w, dtype=float).reshape(3)

        # p <- v
        A[0:3, 3:6] = np.eye(3) * dt

        # v <- f
        B[3:6, 0:3] = np.eye(3) * (dt / m)

        # 2nd order term p <- f (more accurate condensed model)
        B[0:3, 0:3] += np.eye(3) * (0.5 * (dt**2) / m)

        # rpy <- omega (lecture yaw-linearization):
        #   d/dt [phi,theta,psi] ≈ Rz(-psi_r) * omega
        Rz_m = _Rz(-float(yaw_ref))
        A[6:9, 9:12] = Rz_m * dt

        # omega <- torque from GRF: tau = r x f
        # lecture inertia approximation:
        #   I ≈ Rz(psi_r) I_body Rz(psi_r)^T
        I_body = np.asarray(I_body, dtype=float).reshape(3, 3)
        Rz_p = _Rz(float(yaw_ref))
        I_w = (Rz_p @ I_body @ Rz_p.T).astype(float)
        I_w_inv = np.linalg.inv(I_w + 1e-9 * np.eye(3))
        B_omega = (dt * (I_w_inv @ _skew(r))).astype(float)  # (3,3)
        B[9:12, 0:3] = B_omega

        # yaw_ref integrates commanded yaw rate (here: handled via b[12])
        A[12, 12] = 1.0

        # constant acceleration from gravity + baseline thrust
        g_world = np.array([0.0, 0.0, -g], dtype=float)
        a0 = (float(T_base) / m) * z_w + g_world
        b[3:6] = dt * a0
        b[0:3] = 0.5 * (dt**2) * a0
        # yaw_ref update left to caller via b[12]

        return A, B, b

    def _condense(
        self,
        *,
        A: np.ndarray,
        B: np.ndarray,
        b: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build A_qp, B_qp, xbar such that:
          X = A_qp x0 + B_qp U + xbar
        where X stacks x[1..N], U stacks u[0..N-1].
        """
        N = int(self.cfg.N)
        nx, nu = self.nx, self.nu

        # powers of A
        Ap = [np.eye(nx, dtype=float)]
        for k in range(1, N + 1):
            Ap.append(Ap[-1] @ A)

        # A_qp
        A_qp = np.zeros((N * nx, nx), dtype=float)
        for k in range(N):
            A_qp[k * nx : (k + 1) * nx, :] = Ap[k + 1]

        # B_qp (block lower-triangular)
        B_qp = np.zeros((N * nx, N * nu), dtype=float)
        for k in range(N):  # row for x[k+1]
            for j in range(k + 1):  # input u[j] affects x[k+1]
                blk = Ap[k - j] @ B
                B_qp[k * nx : (k + 1) * nx, j * nu : (j + 1) * nu] = blk

        # xbar: response to constant b with x0=0, U=0
        xbar = np.zeros((N * nx,), dtype=float)
        xb = np.zeros(nx, dtype=float)
        for k in range(N):
            xb = A @ xb + b
            xbar[k * nx : (k + 1) * nx] = xb

        return A_qp, B_qp, xbar

    def solve(
        self,
        *,
        x0: np.ndarray,
        x_ref_seq: np.ndarray,
        contact_schedule: np.ndarray,
        m: float,
        g: float,
        I_body: np.ndarray,
        r_foot_w: np.ndarray,
        z_w: np.ndarray,
        T_base: float,
        yaw_rate_ref: float = 0.0,
    ) -> Dict[str, Any]:
        """
        Args:
          x0: (13,)
          x_ref_seq: (N, 13) reference for x[1..N]
          contact_schedule: (N,) 1=contact, 0=flight
        """
        cfg = self.cfg
        N = int(cfg.N)
        nx, nu = self.nx, self.nu

        x0 = np.asarray(x0, dtype=float).reshape(nx)
        x_ref_seq = np.asarray(x_ref_seq, dtype=float).reshape(N, nx)
        contact_schedule = np.asarray(contact_schedule, dtype=int).reshape(N)

        yaw_ref = float(x0[12])
        A, B, b = self._build_dynamics(m=m, g=g, I_body=I_body, r_foot_w=r_foot_w, z_w=z_w, T_base=T_base, yaw_ref=yaw_ref)
        # inject yaw_ref dynamics term
        b = np.asarray(b, dtype=float).copy()
        b[12] = float(cfg.dt) * float(yaw_rate_ref)

        A_qp, B_qp, xbar = self._condense(A=A, B=B, b=b)

        # L and K (diagonal)
        w = self._weights_W()
        W = np.diag(w)
        L = sp.kron(sp.eye(N, format="csc"), sp.csc_matrix(W))
        K = sp.eye(N * nu, format="csc") * float(cfg.alpha_u)

        # H, gvec
        Bsp = sp.csc_matrix(B_qp)
        H = 2.0 * (Bsp.T @ L @ Bsp + K)
        # residual without control: A_qp x0 + xbar - x_ref
        x_ref_stack = x_ref_seq.reshape(N * nx)
        res0 = (A_qp @ x0).reshape(N * nx) + xbar - x_ref_stack
        gvec = 2.0 * (Bsp.T @ (L @ res0))

        # Constraints on U:
        # bounds per u_k plus friction pyramid rows.
        # Use A_cons = [I; A_fric], l/u accordingly.
        mg = float(max(1e-6, m) * float(g))
        fxy_lim = float(min(float(cfg.fxy_max), float(cfg.fxy_max_ratio) * mg))
        # Variable bounds
        lU = np.zeros(N * nu, dtype=float)
        uU = np.zeros(N * nu, dtype=float)
        for k in range(N):
            if int(contact_schedule[k]) == 1:
                lU[k * nu + 0] = -float(fxy_lim)
                uU[k * nu + 0] = float(fxy_lim)
                lU[k * nu + 1] = -float(fxy_lim)
                uU[k * nu + 1] = float(fxy_lim)
                lU[k * nu + 2] = float(cfg.fz_min)
                uU[k * nu + 2] = float(cfg.fz_max)
            else:
                # flight: forces = 0
                lU[k * nu : (k + 1) * nu] = 0.0
                uU[k * nu : (k + 1) * nu] = 0.0

        # friction rows (only for contact steps)
        fr_rows = []
        fr_l = []
        fr_u = []
        mu = float(cfg.mu)
        for k in range(N):
            if int(contact_schedule[k]) != 1:
                continue
            base = k * nu
            # fx - mu fz <= 0
            r = np.zeros(N * nu, dtype=float)
            r[base + 0] = 1.0
            r[base + 2] = -mu
            fr_rows.append(r); fr_l.append(-np.inf); fr_u.append(0.0)
            # -fx - mu fz <= 0
            r = np.zeros(N * nu, dtype=float)
            r[base + 0] = -1.0
            r[base + 2] = -mu
            fr_rows.append(r); fr_l.append(-np.inf); fr_u.append(0.0)
            # fy - mu fz <= 0
            r = np.zeros(N * nu, dtype=float)
            r[base + 1] = 1.0
            r[base + 2] = -mu
            fr_rows.append(r); fr_l.append(-np.inf); fr_u.append(0.0)
            # -fy - mu fz <= 0
            r = np.zeros(N * nu, dtype=float)
            r[base + 1] = -1.0
            r[base + 2] = -mu
            fr_rows.append(r); fr_l.append(-np.inf); fr_u.append(0.0)

        A_id = sp.eye(N * nu, format="csc")
        if fr_rows:
            A_fric = sp.csc_matrix(np.vstack(fr_rows))
            A_cons = sp.vstack([A_id, A_fric], format="csc")
            l = np.concatenate([lU, np.asarray(fr_l, dtype=float)])
            u = np.concatenate([uU, np.asarray(fr_u, dtype=float)])
        else:
            A_cons = A_id
            l = lU
            u = uU

        # Setup/Update solver
        # NOTE: H depends on B_qp which depends on r_foot_w and yaw, both change every step.
        # For this small problem (3N variables), full re-setup is fast and avoids stale Hessian.
        # Warm-start from previous solution via OSQP internal state.
        schedule_changed = (
            (not self._initialized)
            or (self._last_contact_schedule is None)
            or (np.any(self._last_contact_schedule != contact_schedule))
        )
        # Always re-setup because P (Hessian) changes every step.
        # With 30 variables the setup is ~0.5ms.
        self._prob = osqp.OSQP()
        self._prob.setup(
            P=H,
            q=np.asarray(gvec, dtype=float),
            A=A_cons,
            l=np.asarray(l, dtype=float),
            u=np.asarray(u, dtype=float),
            verbose=False,
            max_iter=max(50, int(cfg.max_iter)),
            eps_abs=float(cfg.eps_abs),
            eps_rel=float(cfg.eps_rel),
            polish=False,   # Skip polish for real-time (saves ~30% runtime)
            warm_start=True,
        )
        self._initialized = True
        self._last_contact_schedule = contact_schedule.copy()

        res = self._prob.solve()
        status = str(getattr(res.info, "status", "unknown"))
        solved = status in ("solved", "solved inaccurate", "solved_inaccurate")
        U_opt = np.asarray(res.x, dtype=float) if solved and res.x is not None else np.zeros(N * nu, dtype=float)
        U_opt = U_opt.reshape(N, nu)

        return {
            "status": status,
            "U_opt": U_opt,
            "u0": U_opt[0].copy(),
            "obj_val": float(getattr(res.info, "obj_val", np.nan)),
        }


