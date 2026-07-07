"""
Full-body (manipulator-equation) WBC-QP using MuJoCo dynamics.

This follows the formulation used in MIT Mini-Cheetah software notes:
  H(q) qdd + b(q,qd) = S^T tau + J_c^T f_c + Σ J_p,i^T f_p,i
  (stance contact constraint) J_c qdd + Jdot_c qd = 0

We solve a single QP each control step:
  decision x = [qdd(nv), tau(nu), f_c(3), thrusts(3), s_dyn(nv), s_contact(3)]

Constraints:
  - dynamics equality (with slack s_dyn)
  - stance contact acceleration equality (with slack s_contact); disabled in flight
  - friction pyramid on f_c (stance); in flight, f_c is clamped to 0
  - torque and thrust bounds

Cost (weighted least squares):
  - base linear accel tracking (height/vel/jump reference)
  - base angular accel tracking (attitude leveling / desired rpy)
  - swing-foot accel tracking (flight)
  - regularization on qdd, tau, f_c, thrusts, slacks
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np

import mujoco
import osqp
import scipy.sparse as sp


@dataclass
class FullBodyWBCConfig:
    # contact
    mu: float = 0.8
    fz_min: float = 0.0
    fz_max: float = 400.0

    # bounds
    tau_limit: float = 300.0
    shift_tau_limit: float = 300.0
    thrust_max_each: float = 5.0
    thrust_sum_max: float | None = None  # if None, no sum bound

    # task weights
    w_base_lin: float = 50.0
    # NOTE: keep yaw *uncontrolled* by default (MiniCheetah-style options vary; for this hopper we keep yaw free).
    # roll/pitch share w_base_ang, yaw uses w_base_ang_yaw (default 0).
    w_base_ang: float = 10.0
    w_base_ang_yaw: float = 0.0
    w_swing_foot: float = 5.0

    # regularization weights
    w_qdd: float = 1e-3
    w_tau: float = 1e-3
    w_fc: float = 1e-4
    w_thrust: float = 1e-4
    w_slack_dyn: float = 1e6
    w_slack_contact: float = 1e6

    # optional tracking of planned contact force / thrust (e.g., from MPC)
    w_fc_ref: float = 0.0
    # scale for the Z component of the force tracking weight:
    # - set to 0.0 to track only horizontal (fx,fy) from MPC while letting vertical (fz) be driven by hop/height tasks.
    w_fc_ref_z_scale: float = 1.0
    w_thrust_ref: float = 0.0

    # swing gains (in world)
    swing_kp: float = 200.0
    swing_kd: float = 10.0

    # optional hard-ish height cap (single-step lookahead using base vertical acceleration)
    # Enforces: z_next = z + vz*dt + 0.5*az*dt^2 <= z_max
    z_max: float | None = None
    z_max_dt: float = 0.01
    z_max_enforce_vz_cap: bool = True
    gravity: float = 9.81


class FullBodyWBCQP:
    def __init__(self, cfg: FullBodyWBCConfig):
        self.cfg = cfg
        self._prob = osqp.OSQP()
        self._initialized = False
        self._last_nvar = None
        self._last_ncon = None

    def _build_and_setup(self, P: sp.csc_matrix, q: np.ndarray, A: sp.csc_matrix, l: np.ndarray, u: np.ndarray):
        # Re-setup each time if dimensions changed; otherwise OSQP update is possible but not necessary here.
        self._prob = osqp.OSQP()
        self._prob.setup(P=P, q=q, A=A, l=l, u=u, verbose=False, polish=True)
        self._initialized = True
        self._last_nvar = P.shape[0]
        self._last_ncon = A.shape[0]

    @staticmethod
    def _quat_to_R_wb(quat_wxyz: np.ndarray) -> np.ndarray:
        w, x, y, z = [float(v) for v in quat_wxyz]
        return np.array([
            [w*w + x*x - y*y - z*z, 2*(x*y - w*z),       2*(x*z + w*y)],
            [2*(x*y + w*z),         w*w - x*x + y*y - z*z, 2*(y*z - w*x)],
            [2*(x*z - w*y),         2*(y*z + w*x),       w*w - x*x - y*y + z*z],
        ], dtype=float)

    def solve(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        *,
        base_body_id: int,
        foot_body_id: int,
        joint_dof_ids: list[int],
        prop_positions_body: np.ndarray,
        desired_base_lin_acc_w: np.ndarray,
        desired_base_ang_acc_w: np.ndarray,
        desired_swing_foot_pos_w: np.ndarray | None,
        in_stance: bool,
        dyn_override: dict | None = None,
        thrust_sum_target: float | None = None,
        f_contact_ref_w: np.ndarray | None = None,
        thrusts_ref: np.ndarray | None = None,
    ) -> dict:
        """
        Returns dict with:
          - tau (nu,)
          - f_contact_w (3,)
          - thrusts (3,)
          - slack_dyn_norm, slack_contact_norm
          - status
        """
        nv = int(model.nv)
        nu = int(len(joint_dof_ids))  # we command these dofs directly

        qvel = data.qvel.copy()

        # ---- Dynamics + Jacobians ----
        base_pos_w = data.xpos[base_body_id].copy()
        # Track COM-point tasks (MiniCheetah-style): use MuJoCo body COM position if available.
        try:
            base_com_w = data.xipos[base_body_id].copy()  # COM position of this body in world
        except Exception:
            base_com_w = base_pos_w.copy()
        foot_pos_w = data.xpos[foot_body_id].copy()

        if dyn_override is None:
            # MuJoCo backend
            M = np.zeros((nv, nv), dtype=float)
            mujoco.mj_fullM(model, M, data.qM)
            bias = data.qfrc_bias.copy()  # Coriolis + gravity

            # base point jacobian (linear only) + Jdotqdot (linear)
            Jb_p = np.zeros((3, nv), dtype=float)
            Jb_r = np.zeros((3, nv), dtype=float)
            mujoco.mj_jac(model, data, Jb_p, Jb_r, base_com_w, base_body_id)
            Jb_p_dot = np.zeros((3, nv), dtype=float)
            Jb_r_dot = np.zeros((3, nv), dtype=float)
            mujoco.mj_jacDot(model, data, Jb_p_dot, Jb_r_dot, base_com_w, base_body_id)
            Jdotqdot_base = (Jb_p_dot @ qvel).copy()
            Jdotqdot_base_ang = (Jb_r_dot @ qvel).copy()

            # foot point jacobian (linear only) + Jdotqdot (linear)
            Jc = np.zeros((3, nv), dtype=float)
            mujoco.mj_jac(model, data, Jc, None, foot_pos_w, foot_body_id)
            Jc_dot = np.zeros((3, nv), dtype=float)
            mujoco.mj_jacDot(model, data, Jc_dot, None, foot_pos_w, foot_body_id)
            Jdotqdot_foot = (Jc_dot @ qvel).copy()
        else:
            # Pinocchio (or other) backend override
            M = np.asarray(dyn_override["M"], dtype=float).reshape(nv, nv)
            bias = np.asarray(dyn_override["b"], dtype=float).reshape(nv)
            Jb_p = np.asarray(dyn_override["J_base_lin_w"], dtype=float).reshape(3, nv)
            Jdotqdot_base = np.asarray(dyn_override["Jdotqdot_base_lin_w"], dtype=float).reshape(3)
            Jb_r = np.asarray(dyn_override["J_base_ang_w"], dtype=float).reshape(3, nv)
            Jdotqdot_base_ang = np.asarray(dyn_override["Jdotqdot_base_ang_w"], dtype=float).reshape(3)
            Jc = np.asarray(dyn_override["J_foot_lin_w"], dtype=float).reshape(3, nv)
            Jdotqdot_foot = np.asarray(dyn_override["Jdotqdot_foot_lin_w"], dtype=float).reshape(3)

        # base rotation for thrust direction
        quat_wxyz = data.xquat[base_body_id].copy()  # [w,x,y,z]
        R_wb = self._quat_to_R_wb(quat_wxyz)
        z_w = R_wb @ np.array([0.0, 0.0, 1.0], dtype=float)

        # prop Jacobian contributions (each thrust is along z_w at a point attached to base)
        prop_positions_body = np.asarray(prop_positions_body, dtype=float).reshape(3, 3)
        prop_cols = np.zeros((nv, 3), dtype=float)
        for i in range(3):
            p_w = base_pos_w + (R_wb @ prop_positions_body[i])
            Jp = np.zeros((3, nv), dtype=float)
            mujoco.mj_jac(model, data, Jp, None, p_w, base_body_id)
            # generalized force from thrust ti is Jp^T * (z_w * ti) => column = Jp^T z_w
            prop_cols[:, i] = Jp.T @ z_w

        # ---- Decision variables ----
        # x = [qdd(nv), tau(nu), f_c(3), thrusts(3), s_dyn(nv), s_c(3)]
        idx_qdd = slice(0, nv)
        idx_tau = slice(idx_qdd.stop, idx_qdd.stop + nu)
        idx_fc = slice(idx_tau.stop, idx_tau.stop + 3)
        idx_th = slice(idx_fc.stop, idx_fc.stop + 3)
        idx_sdyn = slice(idx_th.stop, idx_th.stop + nv)
        idx_sc = slice(idx_sdyn.stop, idx_sdyn.stop + 3)
        nvar = idx_sc.stop

        # ---- Constraints ----
        # 1) Dynamics: M qdd - S^T tau - Jc^T fc - prop_cols * thrusts + s_dyn = -bias
        rows = []
        cols = []
        vals = []
        b = []

        def add_A_block(r0: int, c0: int, mat: np.ndarray):
            mat = np.asarray(mat, dtype=float)
            rr, cc = mat.shape
            ii, jj = np.nonzero(mat)
            for i, j in zip(ii, jj):
                rows.append(r0 + int(i))
                cols.append(c0 + int(j))
                vals.append(float(mat[i, j]))

        r = 0
        # dynamics rows: nv
        add_A_block(r, idx_qdd.start, M)

        # -S^T on tau block
        S_T = np.zeros((nv, nu), dtype=float)
        for j, dof in enumerate(joint_dof_ids):
            S_T[int(dof), j] = 1.0
        add_A_block(r, idx_tau.start, -S_T)

        # -Jc^T on contact force block
        add_A_block(r, idx_fc.start, -(Jc.T))

        # -prop_cols on thrust block
        add_A_block(r, idx_th.start, -prop_cols)

        # +I on s_dyn block
        add_A_block(r, idx_sdyn.start, np.eye(nv))

        b_dyn = (-bias).copy()
        b.extend(list(b_dyn))
        r_dyn = slice(r, r + nv)
        r += nv

        # 2) Contact acceleration constraint (stance): Jc qdd + s_c = -Jc_dot qvel
        add_A_block(r, idx_qdd.start, Jc)
        add_A_block(r, idx_sc.start, np.eye(3))
        b_c = (-Jdotqdot_foot).copy()
        b.extend(list(b_c))
        r_contact = slice(r, r + 3)
        r += 3

        # 2.5) Optional base height upper bound (one-step lookahead)
        r_zmax = None
        r_vzcap = None
        if self.cfg.z_max is not None:
            # az = e3^T (Jb_p qdd) + Jdotqdot_base_z
            e3J = Jb_p[2:3, :]  # 1 x nv
            add_A_block(r, idx_qdd.start, e3J)
            b.extend([0.0])
            r_zmax = r
            r += 1
            if bool(self.cfg.z_max_enforce_vz_cap):
                # also cap upward vertical velocity so ballistic apex stays <= z_max (stance only)
                add_A_block(r, idx_qdd.start, e3J)
                b.extend([0.0])
                r_vzcap = r
                r += 1

        # 3) Friction + normal bounds on fc (always present as constraints, but in flight fc is clamped to 0)
        #    fz >= fz_min  =>  [0,0,1] fc >= fz_min
        add_A_block(r, idx_fc.start, np.array([[0.0, 0.0, 1.0]]))
        b.extend([0.0])  # placeholder
        r_fz_ge = r
        r += 1
        #    fz <= fz_max
        add_A_block(r, idx_fc.start, np.array([[0.0, 0.0, 1.0]]))
        b.extend([0.0])
        r_fz_le = r
        r += 1
        #    friction pyramid
        mu = float(self.cfg.mu)
        fric_rows = [
            [1.0, 0.0, -mu],   # fx - mu fz <= 0
            [-1.0, 0.0, -mu],  # -fx - mu fz <= 0
            [0.0, 1.0, -mu],   # fy - mu fz <= 0
            [0.0, -1.0, -mu],  # -fy - mu fz <= 0
        ]
        add_A_block(r, idx_fc.start, np.array(fric_rows, dtype=float))
        b.extend([0.0, 0.0, 0.0, 0.0])
        r_fric = slice(r, r + 4)
        r += 4

        # 4) Optional thrust sum constraint:
        #    - if thrust_sum_target is not None: sum(thrust) == target (realistic: motors always spinning at base thrust)
        #    - else if cfg.thrust_sum_max is not None: sum(thrust) <= max
        r_tsum = None
        if (thrust_sum_target is not None) or (self.cfg.thrust_sum_max is not None):
            add_A_block(r, idx_th.start, np.array([[1.0, 1.0, 1.0]], dtype=float))
            b.extend([0.0])
            r_tsum = r
            r += 1

        # 5) Variable bounds via identity rows
        add_A_block(r, 0, np.eye(nvar))
        b.extend([0.0] * nvar)
        r_bounds = slice(r, r + nvar)
        r += nvar

        A = sp.csc_matrix((vals, (rows, cols)), shape=(r, nvar))

        # Build l/u
        l = -np.inf * np.ones(r, dtype=float)
        u = np.inf * np.ones(r, dtype=float)

        # dynamics eq
        l[r_dyn] = np.asarray(b_dyn, dtype=float)
        u[r_dyn] = np.asarray(b_dyn, dtype=float)

        # contact acceleration eq: only active in stance
        if in_stance:
            l[r_contact] = np.asarray(b_c, dtype=float)
            u[r_contact] = np.asarray(b_c, dtype=float)
        else:
            # disable by leaving as +/- inf
            pass

        # fz bounds + friction
        l[r_fz_ge] = float(self.cfg.fz_min)
        u[r_fz_ge] = np.inf
        l[r_fz_le] = -np.inf
        u[r_fz_le] = float(self.cfg.fz_max)
        l[r_fric] = -np.inf
        u[r_fric] = 0.0

        # thrust sum
        if r_tsum is not None:
            if thrust_sum_target is not None:
                tgt = float(thrust_sum_target)
                l[r_tsum] = tgt
                u[r_tsum] = tgt
            else:
                l[r_tsum] = -np.inf
                u[r_tsum] = float(self.cfg.thrust_sum_max)

        # variable bounds
        lb = -np.inf * np.ones(nvar, dtype=float)
        ub = np.inf * np.ones(nvar, dtype=float)

        # qdd bounds: unbounded
        # tau bounds (3 motors)
        lb[idx_tau] = -float(self.cfg.tau_limit)
        ub[idx_tau] = float(self.cfg.tau_limit)
        if nu >= 3:
            # shift joint often has different limit
            lb[idx_tau.start + 2] = -float(self.cfg.shift_tau_limit)
            ub[idx_tau.start + 2] = float(self.cfg.shift_tau_limit)

        # contact force bounds:
        if in_stance:
            lb[idx_fc] = -np.inf
            ub[idx_fc] = np.inf
        else:
            lb[idx_fc] = 0.0
            ub[idx_fc] = 0.0

        # thrust bounds
        lb[idx_th] = 0.0
        ub[idx_th] = float(self.cfg.thrust_max_each)

        # slack bounds
        lb[idx_sdyn] = -np.inf
        ub[idx_sdyn] = np.inf
        lb[idx_sc] = -np.inf
        ub[idx_sc] = np.inf

        l[r_bounds] = lb
        u[r_bounds] = ub

        # base height cap bound
        if r_zmax is not None:
            # z_next <= z_max  => 0.5*az*dt^2 <= z_max - z - vz*dt
            dtc = float(self.cfg.z_max_dt)
            z_now = float(base_com_w[2])
            # COM linear velocity estimate from Jacobian (more consistent than free-joint v when COM offset exists)
            v_com = (Jb_p @ qvel).reshape(3)
            vz_now = float(v_com[2])
            rhs = (2.0 / (dtc * dtc)) * (float(self.cfg.z_max) - z_now - vz_now * dtc) - float(Jdotqdot_base[2])
            # only enforce when rhs is finite; allow negative rhs (forces downward accel)
            l[r_zmax] = -np.inf
            u[r_zmax] = float(rhs)

        # vertical velocity cap bound (stance only)
        if (r_vzcap is not None) and in_stance:
            dtc = float(self.cfg.z_max_dt)
            z_now = float(base_com_w[2])
            v_com = (Jb_p @ qvel).reshape(3)
            vz_now = float(v_com[2])
            g = float(self.cfg.gravity)
            vz_cap = float(np.sqrt(max(0.0, 2.0 * g * (float(self.cfg.z_max) - z_now))))
            rhs_v = (vz_cap - vz_now) / dtc - float(Jdotqdot_base[2])
            l[r_vzcap] = -np.inf
            u[r_vzcap] = float(rhs_v)
        elif r_vzcap is not None:
            # disable in flight
            l[r_vzcap] = -np.inf
            u[r_vzcap] = np.inf

        # ---- Objective ----
        P = np.zeros((nvar, nvar), dtype=float)
        q_vec = np.zeros(nvar, dtype=float)

        # Regularization
        P[idx_qdd, idx_qdd] += np.eye(nv) * float(self.cfg.w_qdd)
        P[idx_tau, idx_tau] += np.eye(nu) * float(self.cfg.w_tau)
        P[idx_fc, idx_fc] += np.eye(3) * float(self.cfg.w_fc)
        P[idx_th, idx_th] += np.eye(3) * float(self.cfg.w_thrust)
        P[idx_sdyn, idx_sdyn] += np.eye(nv) * float(self.cfg.w_slack_dyn)
        P[idx_sc, idx_sc] += np.eye(3) * float(self.cfg.w_slack_contact)

        # Task: base linear acceleration
        a_lin_des = np.asarray(desired_base_lin_acc_w, dtype=float).reshape(3)
        b_lin = a_lin_des - Jdotqdot_base
        W_lin = np.eye(3) * float(self.cfg.w_base_lin)
        P[idx_qdd, idx_qdd] += (Jb_p.T @ W_lin @ Jb_p) * 2.0
        q_vec[idx_qdd] += (-2.0 * (Jb_p.T @ W_lin @ b_lin))

        # Task: base angular acceleration
        a_ang_des = np.asarray(desired_base_ang_acc_w, dtype=float).reshape(3)
        b_ang = a_ang_des - Jdotqdot_base_ang
        W_ang = np.diag(
            [
                float(self.cfg.w_base_ang),
                float(self.cfg.w_base_ang),
                float(self.cfg.w_base_ang_yaw),
            ]
        ).astype(float)
        P[idx_qdd, idx_qdd] += (Jb_r.T @ W_ang @ Jb_r) * 2.0
        q_vec[idx_qdd] += (-2.0 * (Jb_r.T @ W_ang @ b_ang))

        # Optional tracking: stance contact force (world)
        if (f_contact_ref_w is not None) and float(self.cfg.w_fc_ref) > 0.0:
            f_ref = np.asarray(f_contact_ref_w, dtype=float).reshape(3)
            wf = float(self.cfg.w_fc_ref)
            wf_z = float(wf * float(getattr(self.cfg, "w_fc_ref_z_scale", 1.0)))
            Wf = np.diag([wf, wf, wf_z]).astype(float)
            P[idx_fc, idx_fc] += (2.0 * Wf)
            q_vec[idx_fc] += (-2.0 * (Wf @ f_ref))

        # Optional tracking: per-rotor thrusts
        if (thrusts_ref is not None) and float(self.cfg.w_thrust_ref) > 0.0:
            t_ref = np.asarray(thrusts_ref, dtype=float).reshape(3)
            wt = float(self.cfg.w_thrust_ref)
            P[idx_th, idx_th] += (np.eye(3) * (2.0 * wt))
            q_vec[idx_th] += (-2.0 * wt) * t_ref

        # Task: swing foot (only meaningful in flight)
        if (not in_stance) and (desired_swing_foot_pos_w is not None):
            p_des = np.asarray(desired_swing_foot_pos_w, dtype=float).reshape(3)
            # current foot vel in world
            v_foot = Jc @ qvel
            # PD accel command
            a_foot_cmd = float(self.cfg.swing_kp) * (p_des - foot_pos_w) + float(self.cfg.swing_kd) * (0.0 - v_foot)
            b_foot = a_foot_cmd - Jdotqdot_foot
            W_foot = np.eye(3) * float(self.cfg.w_swing_foot)
            P[idx_qdd, idx_qdd] += (Jc.T @ W_foot @ Jc) * 2.0
            q_vec[idx_qdd] += (-2.0 * (Jc.T @ W_foot @ b_foot))

        P = sp.csc_matrix((P + P.T) * 0.5)  # ensure symmetric

        # ---- Solve ----
        # Note: OSQP only supports fast updates when sparsity pattern is fixed.
        # Here P/A change sparsity due to task Jacobians, so we re-setup each step for correctness.
        self._build_and_setup(P, q_vec, A, l, u)

        res = self._prob.solve()
        x = res.x if res.x is not None else np.zeros(nvar, dtype=float)

        tau = x[idx_tau].copy()
        f_c = x[idx_fc].copy()
        thrusts = x[idx_th].copy()
        s_dyn = x[idx_sdyn].copy()
        s_c = x[idx_sc].copy()

        return {
            'tau': tau,
            'f_contact_w': f_c,
            'thrusts': thrusts,
            'slack_dyn_norm': float(np.linalg.norm(s_dyn)),
            'slack_contact_norm': float(np.linalg.norm(s_c)),
            'status': str(res.info.status) if hasattr(res, 'info') else 'unknown',
        }


