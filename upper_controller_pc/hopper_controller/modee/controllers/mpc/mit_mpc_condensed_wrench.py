"""
MIT lecture-style Convex MPC (Condensed QP) - Wrench MPC (GRF + Tri-Rotor Thrusts)
===============================================================================

This extends the existing GRF-only condensed QP MPC by including tri-rotor thrusts
as decision variables (paper-grade "actuator-feasible" MPC):

  - Control per step: u[k] = [f_foot_w(3), t_rotors(3)]  (world-frame GRF, per-rotor thrusts)
  - Constraints:
      * Contact: friction pyramid + GRF bounds
      * Flight: GRF = 0
      * Rotors: 0 <= t_i <= t_max_each
      * (Optional cost) keep sum(t) near a reference value (assist-only preference), without trimming moments.
  - Yaw is NOT controlled: yaw weights are zero by default. The model keeps yaw state for linearization.

Dynamics model is the same SRB approximation as the lecture notes:
  x[k+1] = A x[k] + B u[k] + b
with x = [p(3), v(3), rpy(3), omega(3), yaw_ref(1)]  (13D)

Important modeling details (physics correctness):
  - r_foot_w and prop_r_w must be expressed about the BODY COM.
  - z_w is the body +Z axis in world (from current attitude); treated constant across the horizon.
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
class MITCondensedWrenchMPCConfig:
    # horizon
    dt: float = 0.02
    N: int = 12

    # contact / friction
    mu: float = 0.8
    fz_min: float = 10.0
    fz_max: float = 180.0
    fxy_max: float = 120.0
    fxy_max_ratio: float = 1.0  # |fxy| <= ratio*(m*g)

    # state tracking weights (diagonal, per-step)
    w_px: float = 0.0
    w_py: float = 0.0
    w_pz: float = 60.0
    w_vx: float = 12.0
    w_vy: float = 10.0
    w_vz: float = 30.0
    w_roll: float = 80.0
    w_pitch: float = 80.0
    w_yaw: float = 0.0
    w_wx: float = 2.0
    w_wy: float = 2.0
    w_wz: float = 0.0
    w_yaw_ref: float = 0.0

    # control regularization
    alpha_u_f: float = 1e-4
    alpha_u_t: float = 2e-4

    # optional soft preference on collective thrust (no trimming/scaling)
    w_tsum_ref: float = 0.0

    # --- Optional joint-torque awareness (approximate) ---
    # We approximate joint torques as:
    #   tau_cmd = A_tau_f * f_foot_w
    # where A_tau_f is provided by the caller (computed from current J and attitude).
    # This lets MPC avoid generating GRFs that are infeasible for the leg actuators.
    #
    # NOTE: This is an approximation because J and attitude will change over the horizon; we treat A_tau_f constant.
    w_tau: float = 0.0  # soft cost weight on ||A_tau_f f||^2 (sum over horizon, contact steps only)
    enforce_tau_limits: bool = False  # if True, add hard constraints |A_tau_f f| <= tau_cmd_max

    # Optional hard state constraints (condensed) on roll/pitch angles across the horizon.
    # Set to a float (deg) to enable, or None to disable.
    rp_limit_deg: float | None = None

    # If True, always re-setup OSQP each solve() call (correct but slower).
    # Required when using state constraints because their A matrix depends on B_qp which varies with r_foot_w.
    re_setup_each_solve: bool = False

    # solver
    max_iter: int = 2000
    eps_abs: float = 1e-4
    eps_rel: float = 1e-4


class MITCondensedWrenchMPC:
    """
    Condensed-QP MPC that optimizes GRF + per-rotor thrust sequence.
    """

    nx: int = 13
    nu: int = 6  # [f(3), t(3)]

    def __init__(self, cfg: MITCondensedWrenchMPCConfig | None = None):
        self.cfg = cfg or MITCondensedWrenchMPCConfig()
        self._prob = osqp.OSQP()
        self._initialized = False
        self._last_contact_schedule: np.ndarray | None = None

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
        prop_r_w: np.ndarray,
        z_w: np.ndarray,
        yaw_ref: float,
        yaw_rate_ref: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dt = float(self.cfg.dt)
        nx, nu = self.nx, self.nu
        A = np.eye(nx, dtype=float)
        B = np.zeros((nx, nu), dtype=float)
        b = np.zeros(nx, dtype=float)

        m = float(max(1e-6, m))
        g = float(g)
        z_w = np.asarray(z_w, dtype=float).reshape(3)
        r_foot_w = np.asarray(r_foot_w, dtype=float).reshape(3)
        prop_r_w = np.asarray(prop_r_w, dtype=float).reshape(3, 3)

        # p <- v
        A[0:3, 3:6] = np.eye(3) * dt

        # v <- [f, t]
        # f contribution
        B[3:6, 0:3] = np.eye(3) * (dt / m)
        B[0:3, 0:3] += np.eye(3) * (0.5 * (dt**2) / m)
        # thrust contribution: a = (1/m) * z_w * sum(t)
        Z = np.outer(z_w, np.ones(3, dtype=float))  # (3,3) each column = z_w
        B[3:6, 3:6] = (dt / m) * Z
        B[0:3, 3:6] += (0.5 * (dt**2) / m) * Z

        # rpy <- omega (lecture yaw-linearization):
        Rz_m = _Rz(-float(yaw_ref))
        A[6:9, 9:12] = Rz_m * dt

        # omega <- torque from GRF and rotor thrusts
        I_body = np.asarray(I_body, dtype=float).reshape(3, 3)
        Rz_p = _Rz(float(yaw_ref))
        I_w = (Rz_p @ I_body @ Rz_p.T).astype(float)
        I_w_inv = np.linalg.inv(I_w + 1e-9 * np.eye(3))

        # GRF torque: tau = r_foot x f
        B[9:12, 0:3] = (dt * (I_w_inv @ _skew(r_foot_w))).astype(float)

        # Rotor torque: tau = sum_i (r_i x (z_w * t_i)) = [cross(r_i, z_w)] t_i
        cols = np.zeros((3, 3), dtype=float)
        for i in range(3):
            cols[:, i] = np.cross(prop_r_w[i], z_w)
        B[9:12, 3:6] = (dt * (I_w_inv @ cols)).astype(float)

        # yaw_ref integrates commanded yaw rate (we keep yaw unregulated by cost, but keep for linearization)
        A[12, 12] = 1.0
        b[12] = dt * float(yaw_rate_ref)

        # constant acceleration from gravity
        g_world = np.array([0.0, 0.0, -g], dtype=float)
        b[3:6] = dt * g_world
        b[0:3] = 0.5 * (dt**2) * g_world

        return A, B, b

    def _condense(self, *, A: np.ndarray, B: np.ndarray, b: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        N = int(self.cfg.N)
        nx, nu = self.nx, self.nu

        Ap = [np.eye(nx, dtype=float)]
        for _ in range(1, N + 1):
            Ap.append(Ap[-1] @ A)

        A_qp = np.zeros((N * nx, nx), dtype=float)
        for k in range(N):
            A_qp[k * nx : (k + 1) * nx, :] = Ap[k + 1]

        B_qp = np.zeros((N * nx, N * nu), dtype=float)
        for k in range(N):
            for j in range(k + 1):
                blk = Ap[k - j] @ B
                B_qp[k * nx : (k + 1) * nx, j * nu : (j + 1) * nu] = blk

        xbar = np.zeros((N * nx,), dtype=float)
        xb = np.zeros(nx, dtype=float)
        for k in range(N):
            xb = A @ xb + b
            xbar[k * nx : (k + 1) * nx] = xb

        return A_qp, B_qp, xbar

    def _condense_ltv(
        self, *, A: np.ndarray, B_seq: list[np.ndarray], b: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Condense an LTV system with constant A and b, but time-varying B_k:
          x[k+1] = A x[k] + B_k u[k] + b

        Returns (A_qp, B_qp, xbar) such that:
          X = A_qp x0 + B_qp U + xbar
        where X stacks x[1..N] and U stacks u[0..N-1].
        """
        N = int(self.cfg.N)
        nx, nu = self.nx, self.nu
        if len(B_seq) != N:
            raise ValueError(f"B_seq length {len(B_seq)} != N {N}")

        Ap = [np.eye(nx, dtype=float)]
        for _ in range(1, N + 1):
            Ap.append(Ap[-1] @ A)

        A_qp = np.zeros((N * nx, nx), dtype=float)
        for k in range(N):
            A_qp[k * nx : (k + 1) * nx, :] = Ap[k + 1]

        B_qp = np.zeros((N * nx, N * nu), dtype=float)
        for k in range(N):
            for j in range(k + 1):
                Bj = np.asarray(B_seq[j], dtype=float).reshape(nx, nu)
                blk = Ap[k - j] @ Bj
                B_qp[k * nx : (k + 1) * nx, j * nu : (j + 1) * nu] = blk

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
        r_foot_w_seq: np.ndarray | None = None,
        prop_r_w: np.ndarray,
        z_w: np.ndarray,
        thrust_max_each: float,
        yaw_rate_ref: float = 0.0,
        thrust_sum_ref: float | None = None,
        thrust_sum_target: float | np.ndarray | None = None,
        thrust_sum_bounds: tuple[float, float] | tuple[np.ndarray, np.ndarray] | None = None,
        # Optional leg torque feasibility (approx, constant over horizon):
        A_tau_f: np.ndarray | None = None,
        tau_cmd_max: np.ndarray | None = None,
    ) -> Dict[str, Any]:
        cfg = self.cfg
        N = int(cfg.N)
        nx, nu = self.nx, self.nu

        x0 = np.asarray(x0, dtype=float).reshape(nx)
        x_ref_seq = np.asarray(x_ref_seq, dtype=float).reshape(N, nx)
        contact_schedule = np.asarray(contact_schedule, dtype=int).reshape(N)

        yaw_ref = float(x0[12])
        if r_foot_w_seq is None:
            A, B, b = self._build_dynamics(
                m=float(m),
                g=float(g),
                I_body=I_body,
                r_foot_w=r_foot_w,
                prop_r_w=prop_r_w,
                z_w=z_w,
                yaw_ref=yaw_ref,
                yaw_rate_ref=float(yaw_rate_ref),
            )
            A_qp, B_qp, xbar = self._condense(A=A, B=B, b=b)
        else:
            rseq = np.asarray(r_foot_w_seq, dtype=float).reshape(N, 3)
            B_seq: list[np.ndarray] = []
            A0 = None
            b0 = None
            for k in range(N):
                Ak, Bk, bk = self._build_dynamics(
                    m=float(m),
                    g=float(g),
                    I_body=I_body,
                    r_foot_w=rseq[k],
                    prop_r_w=prop_r_w,
                    z_w=z_w,
                    yaw_ref=yaw_ref,
                    yaw_rate_ref=float(yaw_rate_ref),
                )
                if A0 is None:
                    A0 = Ak
                    b0 = bk
                B_seq.append(Bk)
            A = np.asarray(A0, dtype=float)
            b = np.asarray(b0, dtype=float)
            A_qp, B_qp, xbar = self._condense_ltv(A=A, B_seq=B_seq, b=b)

        # State tracking weights
        w = self._weights_W()
        W = np.diag(w)
        L = sp.kron(sp.eye(N, format="csc"), sp.csc_matrix(W))

        # Control regularization (block diagonal)
        kdiag = np.array([cfg.alpha_u_f, cfg.alpha_u_f, cfg.alpha_u_f, cfg.alpha_u_t, cfg.alpha_u_t, cfg.alpha_u_t], dtype=float)
        K = sp.kron(sp.eye(N, format="csc"), sp.diags(kdiag, 0, format="csc"))

        Bsp = sp.csc_matrix(B_qp)
        H = 2.0 * (Bsp.T @ L @ Bsp + K)

        x_ref_stack = x_ref_seq.reshape(N * nx)
        res0 = (A_qp @ x0).reshape(N * nx) + xbar - x_ref_stack
        gvec = 2.0 * (Bsp.T @ (L @ res0))

        # Optional soft cost on thrust sum (per step)
        if (thrust_sum_ref is not None) and float(cfg.w_tsum_ref) > 0.0:
            Tref = float(thrust_sum_ref)
            wsum = float(cfg.w_tsum_ref)
            # Add per-step block: wsum * ones(3,3) on rotor thrust block, and q add: -wsum*Tref*[1,1,1]
            H_add = sp.lil_matrix((N * nu, N * nu), dtype=float)
            q_add = np.zeros(N * nu, dtype=float)
            for k in range(N):
                base = k * nu
                idx = np.arange(base + 3, base + 6)
                H_add[np.ix_(idx, idx)] += wsum * np.ones((3, 3), dtype=float)
                q_add[idx] += -wsum * Tref * np.ones(3, dtype=float)
            H = (H + sp.csc_matrix(H_add)).tocsc()
            gvec = np.asarray(gvec, dtype=float) + q_add

        # Optional soft cost on joint torques (normalized): sum_k ||diag(1/tau_max) * (A_tau_f f_k)||^2
        # This avoids over-penalizing vertical support due to the large prismatic torque scale.
        if (A_tau_f is not None) and float(cfg.w_tau) > 0.0:
            A_tau_f = np.asarray(A_tau_f, dtype=float).reshape(3, 3)
            wt = float(cfg.w_tau)
            tau_scale = np.ones(3, dtype=float)
            if tau_cmd_max is not None:
                tau_cmd_max = np.asarray(tau_cmd_max, dtype=float).reshape(3)
                tau_scale = 1.0 / np.maximum(1e-6, np.abs(tau_cmd_max))
            Wtau = np.diag(tau_scale**2).astype(float)
            # Add per-step to the GRF block only (u_k[0:3])
            H_add = sp.lil_matrix((N * nu, N * nu), dtype=float)
            # Equivalent to: sum_k wt * f_k^T (A^T Wtau A) f_k
            Qf = (A_tau_f.T @ Wtau @ A_tau_f).astype(float)
            for k in range(N):
                if int(contact_schedule[k]) != 1:
                    continue
                base = k * nu
                idx = np.arange(base + 0, base + 3)
                H_add[np.ix_(idx, idx)] += wt * Qf
            H = (H + sp.csc_matrix(H_add)).tocsc()

        # Constraints on U: bounds + friction pyramid
        mg = float(max(1e-6, float(m)) * float(g))
        fxy_lim = float(min(float(cfg.fxy_max), float(cfg.fxy_max_ratio) * mg))
        t_max = float(max(0.0, float(thrust_max_each)))

        lU = np.zeros(N * nu, dtype=float)
        uU = np.zeros(N * nu, dtype=float)
        for k in range(N):
            base = k * nu
            if int(contact_schedule[k]) == 1:
                # GRF bounds
                lU[base + 0] = -fxy_lim
                uU[base + 0] = +fxy_lim
                lU[base + 1] = -fxy_lim
                uU[base + 1] = +fxy_lim
                lU[base + 2] = float(cfg.fz_min)
                uU[base + 2] = float(cfg.fz_max)
            else:
                # Flight: GRF = 0
                lU[base + 0 : base + 3] = 0.0
                uU[base + 0 : base + 3] = 0.0

            # Rotor thrust bounds always active
            lU[base + 3 : base + 6] = 0.0
            uU[base + 3 : base + 6] = t_max

        # friction rows (only for contact steps)
        fr_rows = []
        fr_l = []
        fr_u = []
        mu = float(cfg.mu)
        for k in range(N):
            if int(contact_schedule[k]) != 1:
                continue
            base = k * nu
            # Friction pyramid (DIAMOND / inner approximation of cone):
            #   |fx| + |fy| <= mu * fz
            #
            # Implement as 4 linear constraints:
            #   +fx +fy - mu fz <= 0
            #   +fx -fy - mu fz <= 0
            #   -fx +fy - mu fz <= 0
            #   -fx -fy - mu fz <= 0
            #
            # This avoids the axis-aligned box (|fx|<=mu fz, |fy|<=mu fz) which can request diagonal
            # forces that exceed the true friction cone by a factor sqrt(2) and cause real slip.
            for sx, sy in ((+1.0, +1.0), (+1.0, -1.0), (-1.0, +1.0), (-1.0, -1.0)):
                r = np.zeros(N * nu, dtype=float)
                r[base + 0] = sx
                r[base + 1] = sy
                r[base + 2] = -mu
                fr_rows.append(r)
                fr_l.append(-np.inf)
                fr_u.append(0.0)

        A_id = sp.eye(N * nu, format="csc")
        # sum(thrust) constraint rows (always present; bounds set by caller)
        sum_rows = np.zeros((N, N * nu), dtype=float)
        for k in range(N):
            base = k * nu
            sum_rows[k, base + 3 : base + 6] = 1.0
        A_sum = sp.csc_matrix(sum_rows)

        # Optional hard constraints on joint torques: |A_tau_f f_k| <= tau_cmd_max
        A_tau = None
        l_tau = None
        u_tau = None
        if bool(cfg.enforce_tau_limits) and (A_tau_f is not None) and (tau_cmd_max is not None):
            A_tau_f = np.asarray(A_tau_f, dtype=float).reshape(3, 3)
            tau_cmd_max = np.asarray(tau_cmd_max, dtype=float).reshape(3)
            tau_cmd_max = np.abs(tau_cmd_max)
            rows = []
            lrows = []
            urows = []
            for k in range(N):
                if int(contact_schedule[k]) != 1:
                    continue
                base = k * nu
                for j in range(3):
                    r = np.zeros(N * nu, dtype=float)
                    # tau_j = A[j,:] f
                    r[base + 0 : base + 3] = A_tau_f[j, :]
                    rows.append(r)
                    lrows.append(-tau_cmd_max[j])
                    urows.append(+tau_cmd_max[j])
            if rows:
                A_tau = sp.csc_matrix(np.vstack(rows))
                l_tau = np.asarray(lrows, dtype=float)
                u_tau = np.asarray(urows, dtype=float)

        # Optional roll/pitch hard constraints: |roll|,|pitch| <= rp_limit
        A_rp = None
        l_rp = None
        u_rp = None
        if cfg.rp_limit_deg is not None:
            rp_lim = float(np.deg2rad(float(cfg.rp_limit_deg)))
            # X_stack = A_qp x0 + B_qp U + xbar  (stacks x[1..N])
            x_base_stack = (A_qp @ x0).reshape(N * nx) + xbar
            sel = np.zeros(2 * N, dtype=int)
            for k in range(N):
                sel[2 * k + 0] = k * nx + 6  # roll
                sel[2 * k + 1] = k * nx + 7  # pitch
            A_rp = sp.csc_matrix(np.asarray(B_qp[sel, :], dtype=float))
            off = np.asarray(x_base_stack[sel], dtype=float).reshape(2 * N)
            l_rp = (-rp_lim * np.ones(2 * N, dtype=float)) - off
            u_rp = (+rp_lim * np.ones(2 * N, dtype=float)) - off

        # bounds for sum(thrust)
        l_sum = -np.inf * np.ones(N, dtype=float)
        u_sum = np.inf * np.ones(N, dtype=float)
        if thrust_sum_target is not None:
            tgt = np.asarray(thrust_sum_target, dtype=float)
            if tgt.size == 1:
                tgt = np.ones(N, dtype=float) * float(tgt.reshape(-1)[0])
            tgt = tgt.reshape(N)
            l_sum = tgt.copy()
            u_sum = tgt.copy()
        elif thrust_sum_bounds is not None:
            lo, hi = thrust_sum_bounds
            lo = np.asarray(lo, dtype=float)
            hi = np.asarray(hi, dtype=float)
            if lo.size == 1:
                lo = np.ones(N, dtype=float) * float(lo.reshape(-1)[0])
            if hi.size == 1:
                hi = np.ones(N, dtype=float) * float(hi.reshape(-1)[0])
            lo = lo.reshape(N)
            hi = hi.reshape(N)
            l_sum = lo.copy()
            u_sum = hi.copy()

        if fr_rows:
            A_fric = sp.csc_matrix(np.vstack(fr_rows))
            mats = [A_id, A_fric, A_sum]
            l_parts = [lU, np.asarray(fr_l, dtype=float), l_sum]
            u_parts = [uU, np.asarray(fr_u, dtype=float), u_sum]
            if A_tau is not None:
                mats.append(A_tau)
                l_parts.append(np.asarray(l_tau, dtype=float))
                u_parts.append(np.asarray(u_tau, dtype=float))
            if A_rp is not None:
                mats.append(A_rp)
                l_parts.append(np.asarray(l_rp, dtype=float))
                u_parts.append(np.asarray(u_rp, dtype=float))
            A_cons = sp.vstack(mats, format="csc")
            l = np.concatenate(l_parts)
            u = np.concatenate(u_parts)
        else:
            mats = [A_id, A_sum]
            l_parts = [lU, l_sum]
            u_parts = [uU, u_sum]
            if A_tau is not None:
                mats.append(A_tau)
                l_parts.append(np.asarray(l_tau, dtype=float))
                u_parts.append(np.asarray(u_tau, dtype=float))
            if A_rp is not None:
                mats.append(A_rp)
                l_parts.append(np.asarray(l_rp, dtype=float))
                u_parts.append(np.asarray(u_rp, dtype=float))
            A_cons = sp.vstack(mats, format="csc")
            l = np.concatenate(l_parts)
            u = np.concatenate(u_parts)

        # Setup/Update solver (re-setup if contact pattern changed)
        force_setup = bool(cfg.re_setup_each_solve) or (cfg.rp_limit_deg is not None)
        if force_setup or (not self._initialized) or (self._last_contact_schedule is None) or (np.any(self._last_contact_schedule != contact_schedule)):
            self._prob = osqp.OSQP()
            self._prob.setup(
                P=H,
                q=np.asarray(gvec, dtype=float),
                A=A_cons,
                l=np.asarray(l, dtype=float),
                u=np.asarray(u, dtype=float),
                verbose=False,
                max_iter=int(cfg.max_iter),
                eps_abs=float(cfg.eps_abs),
                eps_rel=float(cfg.eps_rel),
                polish=False,
            )
            self._initialized = True
            self._last_contact_schedule = contact_schedule.copy()
        else:
            self._prob.update(q=np.asarray(gvec, dtype=float), l=np.asarray(l, dtype=float), u=np.asarray(u, dtype=float))

        res = self._prob.solve()
        status = str(getattr(res.info, "status", "unknown"))
        solved = status in ("solved", "solved inaccurate", "solved_inaccurate")
        U_opt = np.asarray(res.x, dtype=float) if solved and res.x is not None else np.zeros(N * nu, dtype=float)
        U_opt = U_opt.reshape(N, nu)

        u0 = U_opt[0].copy()
        return {
            "status": status,
            "U_opt": U_opt,
            "u0": u0,
            "f0": u0[0:3].copy(),
            "t0": u0[3:6].copy(),
            "obj_val": float(getattr(res.info, "obj_val", np.nan)),
        }


