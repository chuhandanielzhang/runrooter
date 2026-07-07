"""
QP-based SRB/SE(3) Whole-Body Control (WBC) using OSQP.

This is the "paper-style" formulation:
- Unified rigid-body dynamics constraints for the whole body (force + moment).
- Inequality constraints (friction pyramid + thrust bounds).
- Slack variables on dynamics equalities to guarantee feasibility, heavily penalized.

Decision variables:
  x = [f_foot_w(3), thrusts(3), slack(6)]
      f_foot_w = [fx,fy,fz] in world frame
      thrusts  = [t1,t2,t3] (N), each along body +Z axis rotated to world
      slack    = [sF(3), sTau(3)]

Constraints:
  Dynamics (equalities):
    f_foot + z_w * sum(t) + sF = F_des
    r_foot x f_foot + Σ r_i x (z_w * t_i) + sTau = Tau_des

  Contact/friction (stance only):
    fz >= 0
    |fx| <= mu * fz
    |fy| <= mu * fz
    fz <= fz_max

  Thrust bounds:
    0 <= ti <= t_i_max, and optionally Σ ti <= T_max_total (handled by per-motor cap here)

In flight:
  we set f_foot == 0 by bounds, leaving thrusts to satisfy the (softened) dynamics.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

import osqp
import scipy.sparse as sp


def _skew(r: np.ndarray) -> np.ndarray:
    x, y, z = r
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)


@dataclass
class WBCQPConfig:
    mu: float = 0.8
    # Minimum normal force in stance (N). Helps avoid "free-fall in stance" when tracking
    # a negative vertical acceleration during pre-compression / soft landing.
    # NOTE: this is only enforced when in_stance=True; in flight we still clamp f_foot=0.
    fz_min: float = 0.0
    fz_max: float = 120.0
    # Optional absolute clamp on horizontal contact force components in stance:
    #   |fx| <= fxy_abs_max, |fy| <= fxy_abs_max
    # This is useful to prevent excessive horizontal impulses when prioritizing attitude torque.
    fxy_abs_max: float | None = None
    thrust_total_ratio_max: float = 0.25  # max total thrust = ratio * m * g
    # Minimum per-rotor thrust (N). Keeping each rotor above a small minimum avoids losing moment authority
    # when one rotor hits the non-negativity bound (common in tri-rotor roll/pitch control).
    thrust_min_each: float = 0.0
    # NOTE: yaw is not controlled in MODEE; we intentionally do NOT model/compensate motor reaction torque
    # inside the SRB-QP. Yaw may drift.

    # --- Joint torque variables (stance) ---
    # We include joint torques tau (3) as decision variables and enforce the kinematic force/torque
    # consistency (stance only):
    #   tau_cmd = A_tau_f * f_foot_w
    # where, for MODEE, A_tau_f is typically:
    #   tau_cmd = J_body^T * (R_wb^T * (-f_foot_w))
    # so A_tau_f = -J_body^T * R_wb^T.
    #
    # This allows us to:
    #   - enforce joint torque limits directly in the QP (no post-clipping)
    #   - add a torque regularization cost (optional)
    enable_tau_vars: bool = True
    w_tau: float = 0.0  # joint torque regularization weight (small; optional)
    w_tau_ref: float = 0.0  # joint torque tracking/smoothing weight on ||S*(tau - tau_ref)||^2 (S uses tau_cmd_max if provided)

    # costs
    # NOTE: In OSQP objective 0.5*x^T P x + q^T x, setting:
    #   P_f = w_f * I, q_f = -w_f * f_ref
    # makes the cost equivalent to 0.5*w_f*||f - f_ref||^2 + const.
    w_f: float = 1e-4   # foot-force regularization weight
    w_t: float = 1e-4   # thrust regularization weight
    w_f_ref: float = 1e-2  # foot-force reference tracking weight (spring leg matching)
    w_t_ref: float = 0.0   # thrust reference tracking weight (unused for now)
    # Penalize common-mode (sum) thrust deviation to discourage "all three props push up together".
    # This is a SOFT objective, not a hard constraint:
    #   0.5*w_tsum*(sum(t)-T_ref)^2
    w_tsum_ref: float = 0.0
    # slack penalties (separate force vs torque)
    # Physics: if the requested torque is infeasible under actuator bounds, we'd rather accept torque error
    # than blow up thrust (PWM spike).
    #
    # NOTE: often we want to allow SOME horizontal force mismatch (to generate attitude torque),
    # but keep vertical force tracking very strict to avoid dropping the robot.
    w_slack_F: float = 1e4  # legacy scalar (used if w_slack_Fxy/Fz are None)
    w_slack_Fxy: float | None = None
    w_slack_Fz: float | None = None
    # Optional flight-phase slack weights (if None, reuse stance weights).
    w_slack_Fxy_flight: float | None = None
    w_slack_Fz_flight: float | None = None
    w_slack_tau_flight: float | None = None  # legacy scalar (if axis-specific not provided)
    # Axis-specific torque slack penalties (world frame):
    w_slack_tau_xy: float | None = None
    w_slack_tau_z: float | None = None
    w_slack_tau_flight_xy: float | None = None
    w_slack_tau_flight_z: float | None = None
    w_slack_tau: float = 1e3  # legacy scalar (used if w_slack_tau_xy/z are None)


class WBCQP:
    def __init__(self, cfg: WBCQPConfig):
        self.cfg = cfg
        self._prob = osqp.OSQP()
        self._initialized = False
        # last solution (for smoothness tracking of thrust)
        self._t_prev: np.ndarray | None = None
        self._tau_prev: np.ndarray | None = None

        # dimensions
        # x = [f_foot_w(3), thrusts(3), tau_cmd(3), slack(6)]
        self.nx = 15
        # constraints count:
        # 6 dynamics eq + 1 fz>=0 + 4 friction + 1 fz<=fzmax + 1 thrust-sum + 3 tau-map + nx identity bounds
        # We'll implement foot force bounds using variable bounds directly in OSQP constraints (identity rows).
        self._build_template()

    def _build_template(self):
        # P (objective)
        P = np.zeros((self.nx, self.nx), dtype=float)
        P[0:3, 0:3] = np.eye(3) * self.cfg.w_f
        # IMPORTANT (OSQP): the sparsity pattern of P must NOT change after setup.
        # We may later add a soft cost on total thrust (sum(t)-T_ref)^2, which introduces
        # off-diagonal coupling in the 3x3 thrust block. To keep P.nnz constant, reserve
        # the full 3x3 block here using a tiny epsilon on off-diagonals.
        eps = 1e-12
        P[3:6, 3:6] = (np.eye(3) * self.cfg.w_t) + (np.ones((3, 3), dtype=float) - np.eye(3)) * eps
        # Joint torques tau (regularization, optional)
        # IMPORTANT (OSQP): reserve diagonal nnz for the tau block so we can later add w_tau_ref without
        # changing sparsity.
        P[6:9, 6:9] = np.eye(3) * (float(max(0.0, float(self.cfg.w_tau))) + eps)
        # Slack: [sF(3), sTau(3)]
        wFxy = float(self.cfg.w_slack_F) if self.cfg.w_slack_Fxy is None else float(self.cfg.w_slack_Fxy)
        wFz = float(self.cfg.w_slack_F) if self.cfg.w_slack_Fz is None else float(self.cfg.w_slack_Fz)
        P[9:12, 9:12] = np.diag([wFxy, wFxy, wFz]).astype(float)
        wTau_xy = float(self.cfg.w_slack_tau) if self.cfg.w_slack_tau_xy is None else float(self.cfg.w_slack_tau_xy)
        wTau_z = float(self.cfg.w_slack_tau) if self.cfg.w_slack_tau_z is None else float(self.cfg.w_slack_tau_z)
        P[12:15, 12:15] = np.diag([wTau_xy, wTau_xy, wTau_z]).astype(float)
        # keep dense templates for per-step tracking updates
        self.P_base_dense = P.copy()
        self.q_base = np.zeros(self.nx, dtype=float)
        # OSQP expects P to contain ONLY the upper-triangular part (including diagonal).
        # Keeping a fixed upper-tri sparsity pattern is critical for fast update(Px=...).
        self.P = sp.csc_matrix(np.triu(self.P_base_dense))
        self.q = self.q_base.copy()

        # Build A with fixed sparsity:
        rows = []
        cols = []
        data = []

        def add_block(r0, c0, M):
            M = np.asarray(M, dtype=float)
            rr, cc = M.shape
            for i in range(rr):
                for j in range(cc):
                    v = float(M[i, j])
                    if v != 0.0:
                        rows.append(r0 + i)
                        cols.append(c0 + j)
                        data.append(v)

        row = 0
        self._idx_dyn = slice(row, row + 6)
        # dynamics rows (filled later numerically, but keep structure):
        # We'll reserve dense structure for:
        # [I3, z_w(3x3), 0, I3, 0] and [skew(r_foot), M_prop(3x3), 0, 0, I3]
        add_block(row, 0, np.eye(3))          # f
        add_block(row, 3, np.ones((3, 3)))    # placeholder for z_w * thrusts (each column = z_w)
        add_block(row, 9, np.eye(3))          # slack force
        row += 3
        add_block(row, 0, np.ones((3, 3)))    # placeholder for skew(r_foot)
        add_block(row, 3, np.ones((3, 3)))    # placeholder for prop moment columns
        add_block(row, 12, np.eye(3))         # slack moment
        row += 3

        # inequality rows:
        self._idx_fz_ge0 = row
        # fz >= 0  => [0,0,1] f >= 0
        add_block(row, 0, np.array([[0.0, 0.0, 1.0]]))
        row += 1

        self._idx_fric = slice(row, row + 4)
        # Friction pyramid (axis-aligned box / square pyramid approximation):
        #   |fx| <= mu * fz
        #   |fy| <= mu * fz
        #
        # This matches MIT Convex MPC constraints (Di Carlo et al., 2018) and
        # keeps MPC planning and WBC tracking in the same feasible set.
        add_block(row, 0, np.array([[+1.0, 0.0, -self.cfg.mu]])); row += 1
        add_block(row, 0, np.array([[-1.0, 0.0, -self.cfg.mu]])); row += 1
        add_block(row, 0, np.array([[0.0, +1.0, -self.cfg.mu]])); row += 1
        add_block(row, 0, np.array([[0.0, -1.0, -self.cfg.mu]])); row += 1

        self._idx_fz_le = row
        add_block(row, 0, np.array([[0.0, 0.0, 1.0]]))
        row += 1

        # sum(thrusts) <= T_max_total
        self._idx_tsum = row
        add_block(row, 3, np.array([[1.0, 1.0, 1.0]]))
        row += 1

        # Stance-only torque mapping equality:
        #   tau_cmd - A_tau_f * f = 0
        # We reserve a dense 3x3 block for A_tau_f and a 3x3 identity on tau.
        self._idx_tau_map = slice(row, row + 3)
        add_block(row, 0, np.ones((3, 3)))  # placeholder for (-A_tau_f)
        add_block(row, 6, np.eye(3))        # tau
        row += 3

        # variable bounds via identity rows:
        self._idx_bounds = slice(row, row + self.nx)
        add_block(row, 0, np.eye(self.nx))
        row += self.nx

        self.m = row
        self.A = sp.csc_matrix((data, (rows, cols)), shape=(self.m, self.nx))
        # Build a fast lookup from (row,col) -> index into A.data for numeric updates.
        # (Safe because sparsity is fixed after setup.)
        coo = self.A.tocoo()
        self._A_index = {(int(r), int(c)): int(k) for k, (r, c) in enumerate(zip(coo.row, coo.col))}

        # default bounds
        self.l = -np.inf * np.ones(self.m, dtype=float)
        self.u = np.inf * np.ones(self.m, dtype=float)

        # dynamics equalities
        self.l[self._idx_dyn] = 0.0
        self.u[self._idx_dyn] = 0.0

        # inequality bounds defaults
        # fz>=fz_min (stance will override per-step; default here keeps template consistent)
        self.l[self._idx_fz_ge0] = float(max(0.0, self.cfg.fz_min))
        self.u[self._idx_fz_ge0] = np.inf
        # friction <=0
        self.l[self._idx_fric] = -np.inf
        self.u[self._idx_fric] = 0.0
        # fz <= fz_max
        self.l[self._idx_fz_le] = -np.inf
        self.u[self._idx_fz_le] = self.cfg.fz_max

        # thrust sum bound (filled per-step because depends on m*g)
        self.l[self._idx_tsum] = -np.inf
        self.u[self._idx_tsum] = np.inf

        # tau mapping constraint (disabled by default; enabled in stance if A_tau_f provided)
        self.l[self._idx_tau_map] = -np.inf
        self.u[self._idx_tau_map] = np.inf

        # variable bounds initialized later per step (stance/flight + thrust limits)

        self._prob.setup(P=self.P, q=self.q, A=self.A, l=self.l, u=self.u, verbose=False, polish=True)
        self._initialized = True

    def update_and_solve(
        self,
        m: float,
        g: float,
        z_w: np.ndarray,
        r_foot_w: np.ndarray,
        prop_r_w: np.ndarray,
        F_des: np.ndarray,
        Tau_des: np.ndarray,
        in_stance: bool,
        *,
        thrust_sum_target: float | None = None,
        thrust_sum_bounds: tuple[float, float] | None = None,
        thrust_sum_ref: float | None = None,
        thrust_max_each: float | None = None,
        f_ref: np.ndarray | None = None,
        thrust_ref: np.ndarray | None = None,
        # Optional leg torque mapping (stance):
        A_tau_f: np.ndarray | None = None,
        tau_cmd_max: np.ndarray | None = None,
        # Optional joint torque reference (flight swing task / smoothing):
        tau_ref: np.ndarray | None = None,
    ) -> dict:
        """
        Solve the SRB/Potato WBC-QP.

        Returns dict with:
          - f_foot_w (3,)
          - thrusts (3,)
          - slack (6,)
          - status (str)
          - obj_val (float)
        """
        z_w = np.asarray(z_w, dtype=float).reshape(3)
        r_foot_w = np.asarray(r_foot_w, dtype=float).reshape(3)
        prop_r_w = np.asarray(prop_r_w, dtype=float).reshape(3, 3)
        F_des = np.asarray(F_des, dtype=float).reshape(3)
        Tau_des = np.asarray(Tau_des, dtype=float).reshape(3)

        # If caller didn't provide a thrust reference but thrust tracking is enabled,
        # use previous solution as reference to reduce thrust/PWM chatter.
        if thrust_ref is None and float(self.cfg.w_t_ref) > 0.0 and self._t_prev is not None:
            thrust_ref = np.asarray(self._t_prev, dtype=float).reshape(3).copy()

        # NOTE: we intentionally do NOT auto-fill tau_ref from previous solution here.
        # The caller decides when tau tracking/smoothing should be active (e.g., in flight only),
        # to avoid interfering with stance dynamics / liftoff detection.

        # Build per-step P/q (tracking)
        P_dense = self.P_base_dense.copy()
        q = self.q_base.copy()

        # Phase-dependent slack weighting:
        # - stance: keep vertical force tracking very strict (to not drop)
        # - flight: allow force mismatch so thrust sum can shift (common-mode) to realize roll/pitch moments
        if not bool(in_stance):
            if self.cfg.w_slack_Fxy_flight is not None:
                P_dense[9, 9] = float(self.cfg.w_slack_Fxy_flight)
                P_dense[10, 10] = float(self.cfg.w_slack_Fxy_flight)
            if self.cfg.w_slack_Fz_flight is not None:
                P_dense[11, 11] = float(self.cfg.w_slack_Fz_flight)
            # flight torque slack overrides (axis-specific preferred)
            if (self.cfg.w_slack_tau_flight_xy is not None) or (self.cfg.w_slack_tau_flight_z is not None) or (self.cfg.w_slack_tau_flight is not None):
                wxy = (
                    float(self.cfg.w_slack_tau_flight)
                    if self.cfg.w_slack_tau_flight_xy is None
                    else float(self.cfg.w_slack_tau_flight_xy)
                )
                wz = (
                    float(self.cfg.w_slack_tau_flight)
                    if self.cfg.w_slack_tau_flight_z is None
                    else float(self.cfg.w_slack_tau_flight_z)
                )
                if self.cfg.w_slack_tau_flight is not None:
                    # if only legacy scalar provided, apply to both
                    wxy = float(self.cfg.w_slack_tau_flight) if self.cfg.w_slack_tau_flight_xy is None else wxy
                    wz = float(self.cfg.w_slack_tau_flight) if self.cfg.w_slack_tau_flight_z is None else wz
                P_dense[12:15, 12:15] = np.diag([wxy, wxy, wz]).astype(float)
        if f_ref is not None and float(self.cfg.w_f_ref) > 0.0:
            f_ref = np.asarray(f_ref, dtype=float).reshape(3)
            P_dense[0:3, 0:3] += np.eye(3) * float(self.cfg.w_f_ref)
            # Objective: 0.5*x^T P x + q^T x
            # With P += w*I and q += -w*f_ref, this yields 0.5*w*||f - f_ref||^2 + const.
            q[0:3] += -float(self.cfg.w_f_ref) * f_ref
        if thrust_ref is not None and float(self.cfg.w_t_ref) > 0.0:
            thrust_ref = np.asarray(thrust_ref, dtype=float).reshape(3)
            P_dense[3:6, 3:6] += np.eye(3) * float(self.cfg.w_t_ref)
            q[3:6] += -float(self.cfg.w_t_ref) * thrust_ref

        # Optional soft tracking/smoothing for joint torques (flight swing task).
        # IMPORTANT: keep enough authority on the prismatic axis in flight so the leg extends reliably.
        # We intentionally do NOT normalize by tau_cmd_max^2 here; that normalization made the prismatic
        # tracking weight effectively ~0 and caused mid-air false touchdown triggers.
        if tau_ref is not None and float(self.cfg.w_tau_ref) > 0.0:
            tau_ref = np.asarray(tau_ref, dtype=float).reshape(3)
            wtr = float(self.cfg.w_tau_ref)
            scale2 = np.ones(3, dtype=float)
            P_dense[6:9, 6:9] += np.diag(wtr * scale2).astype(float)
            q[6:9] += -(wtr * scale2) * tau_ref

        # Soft tracking for total thrust: penalize (sum(t) - T_ref)^2
        if thrust_sum_ref is not None and float(self.cfg.w_tsum_ref) > 0.0:
            Tref = float(thrust_sum_ref)
            wsum = float(self.cfg.w_tsum_ref)
            # 0.5*wsum*(1^T t - Tref)^2 = 0.5*t^T (wsum*11^T) t + (-wsum*Tref*1)^T t + const
            P_dense[3:6, 3:6] += wsum * np.ones((3, 3), dtype=float)
            q[3:6] += -wsum * Tref * np.ones(3, dtype=float)

        # IMPORTANT: keep upper-triangular only so OSQP P.nnz stays constant
        self.P = sp.csc_matrix(np.triu(P_dense))
        self.q = q

        # Update dynamics RHS: A x - b = 0  => set l=u=b, by moving b into bounds using constant term:
        # Since OSQP supports only Ax within bounds, we encode equality as Ax = b by setting l=u=b,
        # so we need A to represent the left side directly.
        b = np.concatenate([F_des, Tau_des], axis=0)
        self.l[self._idx_dyn] = b
        self.u[self._idx_dyn] = b

        # Update A numeric values for dynamics blocks (in-place on CSC data)
        A_data = self.A.data
        # rows 0:3, cols 3:6 -> each column = z_w
        for i in range(3):
            for j in range(3):
                idx = self._A_index.get((i, 3 + j))
                if idx is not None:
                    A_data[idx] = z_w[i]
        # rows 3:6, cols 0:3 -> skew(r_foot_w)
        S = _skew(r_foot_w)
        for i in range(3):
            for j in range(3):
                idx = self._A_index.get((3 + i, 0 + j))
                if idx is not None:
                    A_data[idx] = S[i, j]
        # rows 3:6, cols 3:6 -> prop moment columns:
        #   tau_i = r_i x (z_w * t_i)
        # NOTE: We intentionally do NOT include motor reaction torque here (yaw is not controlled).
        for j in range(3):
            col = np.cross(prop_r_w[j], z_w)
            for i in range(3):
                idx = self._A_index.get((3 + i, 3 + j))
                if idx is not None:
                    A_data[idx] = col[i]

        # --- Stance-only torque mapping equality: tau - A_tau_f f = 0 ---
        if bool(self.cfg.enable_tau_vars) and bool(in_stance) and (A_tau_f is not None):
            A_tau_f = np.asarray(A_tau_f, dtype=float).reshape(3, 3)
            r0 = int(self._idx_tau_map.start)
            # A block is stored as (-A_tau_f) on f columns
            for i in range(3):
                for j in range(3):
                    idx = self._A_index.get((r0 + i, 0 + j))
                    if idx is not None:
                        A_data[idx] = -float(A_tau_f[i, j])
            self.l[self._idx_tau_map] = 0.0
            self.u[self._idx_tau_map] = 0.0
        else:
            # Disable in flight or if mapping not provided.
            self.l[self._idx_tau_map] = -np.inf
            self.u[self._idx_tau_map] = +np.inf

        # variable bounds (identity rows)
        lb = -np.inf * np.ones(self.nx, dtype=float)
        ub = np.inf * np.ones(self.nx, dtype=float)

        # thrust bounds
        T_max_total = self.cfg.thrust_total_ratio_max * float(m) * float(g)
        # per-motor cap:
        # - if thrust_max_each is given: use hardware/sim cap
        # - else: conservative cap = total cap (historical behavior)
        T_max_each = float(thrust_max_each) if thrust_max_each is not None else float(T_max_total)
        t_min = float(max(0.0, float(self.cfg.thrust_min_each)))
        t_min = float(min(t_min, float(T_max_each)))
        lb[3:6] = t_min
        ub[3:6] = T_max_each

        # joint torque bounds (only meaningful if we use tau vars)
        if bool(self.cfg.enable_tau_vars) and (tau_cmd_max is not None):
            tau_cmd_max = np.asarray(tau_cmd_max, dtype=float).reshape(3)
            tau_cmd_max = np.abs(tau_cmd_max)
            lb[6:9] = -tau_cmd_max
            ub[6:9] = +tau_cmd_max
        else:
            lb[6:9] = -np.inf
            ub[6:9] = +np.inf
        # thrust sum constraint:
        # - if thrust_sum_bounds is provided: enforce l <= sum(thrust) <= u (assist within a small band)
        # - elif thrust_sum_target is not None: enforce equality sum(thrust) = target (baseline spinning)
        # - else: enforce inequality sum(thrust) <= T_max_total
        if thrust_sum_bounds is not None:
            tl, tu = thrust_sum_bounds
            tl = float(max(0.0, tl))
            tu = float(max(tl, tu))
            self.l[self._idx_tsum] = tl
            self.u[self._idx_tsum] = tu
        elif thrust_sum_target is not None:
            tgt = float(thrust_sum_target)
            self.l[self._idx_tsum] = tgt
            self.u[self._idx_tsum] = tgt
        else:
            self.l[self._idx_tsum] = -np.inf
            self.u[self._idx_tsum] = float(T_max_total)

        if in_stance:
            # allow foot forces; keep friction constraints active
            lb[0:3] = -np.inf
            ub[0:3] = np.inf
            # Optional clamp on horizontal force
            if self.cfg.fxy_abs_max is not None:
                fxy = float(abs(float(self.cfg.fxy_abs_max)))
                if np.isfinite(fxy):
                    lb[0] = -fxy
                    ub[0] = +fxy
                    lb[1] = -fxy
                    ub[1] = +fxy
            # Enforce a stance-only minimum normal force.
            self.l[self._idx_fz_ge0] = float(max(0.0, self.cfg.fz_min))
            self.u[self._idx_fz_le] = self.cfg.fz_max
            self.u[self._idx_fric] = 0.0
        else:
            # flight: clamp foot force to 0
            lb[0:3] = 0.0
            ub[0:3] = 0.0
            # No minimum in flight.
            self.l[self._idx_fz_ge0] = 0.0
            # relax friction constraints (still ok)
            self.u[self._idx_fz_le] = 0.0

        # slack unbounded
        lb[9:15] = -np.inf
        ub[9:15] = np.inf

        # write into constraint bounds for identity rows
        self.l[self._idx_bounds] = lb
        self.u[self._idx_bounds] = ub

        # push updates (including P if it changed)
        # Note: OSQP update supports Px parameter, but we update P.data in-place
        # If P changed, we need to pass Px to update
        self._prob.update(Px=self.P.data, q=self.q, Ax=self.A.data, l=self.l, u=self.u)
        res = self._prob.solve()
        x = res.x if res.x is not None else np.zeros(self.nx, dtype=float)

        f = x[0:3].copy()
        t = x[3:6].copy()
        tau_cmd = x[6:9].copy()
        s = x[9:15].copy()
        status = str(getattr(res.info, "status", "unknown"))
        obj_val = float(getattr(res.info, "obj_val", 0.0))

        # Ensure thrusts are physically realizable before downstream PWM/motor-table execution.
        # OSQP may return slightly infeasible values; also actuators cannot produce negative thrust.
        T_max_total = self.cfg.thrust_total_ratio_max * float(m) * float(g)
        T_max_each = float(thrust_max_each) if thrust_max_each is not None else float(T_max_total)
        t_min = float(max(0.0, float(self.cfg.thrust_min_each)))
        t_min = float(min(t_min, float(T_max_each)))
        t_proj = np.asarray(t, dtype=float).copy()
        t_proj = np.clip(t_proj, t_min, float(T_max_each))
        if thrust_sum_bounds is not None:
            tgt_l, tgt_u = thrust_sum_bounds
            tgt_l = float(np.clip(float(tgt_l), 3.0 * float(t_min), 3.0 * float(T_max_each)))
            tgt_u = float(np.clip(float(tgt_u), tgt_l, 3.0 * float(T_max_each)))

            def _proj_to_sum(target_sum: float):
                nonlocal t_proj
                for _ in range(10):
                    ssum = float(np.sum(t_proj))
                    err = float(target_sum - ssum)
                    if abs(err) <= 1e-6:
                        break
                    if err > 0.0:
                        room = np.clip(float(T_max_each) - t_proj, 0.0, None)
                        rsum = float(np.sum(room))
                        if rsum <= 1e-12:
                            break
                        t_proj = t_proj + room * (err / rsum)
                    else:
                        avail = np.clip(t_proj - float(t_min), 0.0, None)
                        asum = float(np.sum(avail))
                        if asum <= 1e-12:
                            break
                        t_proj = t_proj + avail * (err / asum)
                    t_proj = np.clip(t_proj, float(t_min), float(T_max_each))

            ssum = float(np.sum(t_proj))
            if ssum < tgt_l - 1e-6:
                _proj_to_sum(tgt_l)
            elif ssum > tgt_u + 1e-6:
                _proj_to_sum(tgt_u)
        elif thrust_sum_target is not None:
            tgt = float(np.clip(float(thrust_sum_target), 3.0 * float(t_min), 3.0 * float(T_max_each)))
            for _ in range(10):
                ssum = float(np.sum(t_proj))
                err = float(tgt - ssum)
                if abs(err) <= 1e-6:
                    break
                if err > 0.0:
                    room = np.clip(float(T_max_each) - t_proj, 0.0, None)
                    rsum = float(np.sum(room))
                    if rsum <= 1e-12:
                        break
                    t_proj = t_proj + room * (err / rsum)
                else:
                    avail = np.clip(t_proj - float(t_min), 0.0, None)
                    asum = float(np.sum(avail))
                    if asum <= 1e-12:
                        break
                    t_proj = t_proj + avail * (err / asum)  # err is negative
                t_proj = np.clip(t_proj, float(t_min), float(T_max_each))
        t = np.asarray(t_proj, dtype=float).copy()
        self._t_prev = np.asarray(t, dtype=float).copy()
        # Clip tiny infeasibilities and store previous tau for smoothing.
        if tau_cmd_max is not None:
            tmax = np.asarray(tau_cmd_max, dtype=float).reshape(3)
            tau_cmd = np.clip(tau_cmd, -np.abs(tmax), +np.abs(tmax))
        self._tau_prev = np.asarray(tau_cmd, dtype=float).copy()
        return {
            "f_foot_w": f,
            "thrusts": t,
            "tau_cmd": np.asarray(tau_cmd, dtype=float).copy(),
            "slack": s,
            "status": status,
            "obj_val": obj_val,
        }


