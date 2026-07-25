#!/usr/bin/env python3
"""Full-order constrained inverse dynamics for the 3RSR hopper.

Implements FULL_ORDER_3RSR.md Sec. 5/6: the open-tree EOM with 9 loop-closure
constraints (3x ball connect at the shared foot) is solved per tick for the
unknown hip torques tau, loop multipliers lambda, and (in stance) contact
force f_c:

    M vdot + bias - passive - known_inputs = S^T tau + G^T lambda + J_c^T f_c

Run as a script to execute the torque-replay verification:
  1. simulate a drop + squat motion with the model's own position actuators,
  2. at each frame re-derive tau from (q, v, vdot) alone,
  3. compare against the torque the actuators actually applied, and against
     the static Jacobian map tau_static = -J_leg^T f_c used by the current
     controller.

Usage: python3 full_order_id.py [--duration 6.0] [--plot demos/full_order_id_verify.png]
"""
import argparse
import os

import numpy as np
import mujoco

XML = os.path.join(os.path.dirname(os.path.abspath(__file__)), "three_leg_3rsr_closed.xml")

HIP_JOINTS = ["ctrl_joint_1", "ctrl_joint_2", "ctrl_joint_3"]
CHAIN_SITES = ["foot_site_1", "foot_site_2", "foot_site_3"]
FOOT_SITE = "foot_center"


class FullOrderID:
    """Constrained inverse dynamics on the full 24-dof model.

    All model quantities (M, bias, passive, constraint Jacobians) are read
    from a MuJoCo model/data pair, so there is no hand-coded kinematics that
    can drift out of sync with the plant.
    """

    def __init__(self, model: mujoco.MjModel):
        self.model = model
        nv = model.nv
        self.nv = nv
        self.hip_dofs = np.array(
            [model.jnt_dofadr[model.joint(n).id] for n in HIP_JOINTS], dtype=int)
        self.chain_site_ids = np.array([model.site(n).id for n in CHAIN_SITES], dtype=int)
        self.foot_site_id = model.site(FOOT_SITE).id
        self.foot_body_id = model.body("foot").id
        self.foot_geom_id = model.geom("foot_collision").id
        # passive chain dofs (cross/lower pins) + foot free-joint dofs, used
        # for the frozen-base leg Jacobian in the static map
        pass_names = [f"cross_pin_{i}" for i in (1, 2, 3)] + [f"lower_pin_{i}" for i in (1, 2, 3)]
        self.passive_dofs = np.array(
            [model.jnt_dofadr[model.joint(n).id] for n in pass_names], dtype=int)
        fadr = model.jnt_dofadr[model.joint("foot_free").id]
        self.foot_lin_dofs = np.arange(fadr, fadr + 3)
        self._scratch = mujoco.MjData(model)

    # ---- building blocks -------------------------------------------------
    def loop_jacobian(self, data: mujoco.MjData) -> np.ndarray:
        """G in R^{9 x nv}: rows are d(s_i - p_foot)/dv for the 3 chains."""
        nv = self.nv
        G = np.zeros((9, nv))
        jf = np.zeros((3, nv))
        mujoco.mj_jacSite(self.model, data, jf, None, self.foot_site_id)
        js = np.zeros((3, nv))
        for k, sid in enumerate(self.chain_site_ids):
            js[:] = 0.0
            mujoco.mj_jacSite(self.model, data, js, None, sid)
            G[3 * k:3 * k + 3] = js - jf
        return G

    def contact_jacobian_and_force(self, data: mujoco.MjData):
        """(J_c, f_c_world) for the foot-floor contact, or (None, None)."""
        for ci in range(data.ncon):
            c = data.contact[ci]
            if self.foot_geom_id in (c.geom1, c.geom2):
                Jc = np.zeros((3, self.nv))
                mujoco.mj_jac(self.model, data, Jc, None, c.pos, self.foot_body_id)
                fl = np.zeros(6)
                mujoco.mj_contactForce(self.model, data, ci, fl)
                R = c.frame.reshape(3, 3)  # rows = contact frame axes in world
                f_w = R.T @ fl[:3]         # force on geom2 body; foot is geom2 side
                if c.geom1 == self.foot_geom_id:
                    f_w = -f_w
                return Jc, f_w
        return None, None

    def leg_jacobian_world(self, data: mujoco.MjData) -> np.ndarray:
        """J in R^{3x3}: foot linear velocity (world) per hip rate, base frozen.

        Solves the loop constraint G v = 0 restricted to (theta, passive, p_f)
        columns:  [G_pass G_pf] x = -G_theta.
        """
        G = self.loop_jacobian(data)
        Gd = np.hstack([G[:, self.passive_dofs], G[:, self.foot_lin_dofs]])  # 9x9
        Gt = G[:, self.hip_dofs]                                             # 9x3
        x = np.linalg.solve(Gd, -Gt)                                         # 9x3
        return x[6:9, :]  # rows of p_f block

    # ---- main solve -------------------------------------------------------
    def solve(self, data: mujoco.MjData, vdot: np.ndarray,
              known_qfrc: np.ndarray):
        """Least-squares solve of A x = r for x = (tau, lambda[, f_c]).

        vdot:       generalized acceleration to realize (24,)
        known_qfrc: generalized forces of all *known* inputs (servos, props,
                    external pushes) - must NOT include hip torques,
                    passive damping, or constraint forces.
        Returns (tau, lam, f_c_or_None, residual_norm).
        """
        m, d = self.model, data
        nv = self.nv
        M = np.zeros((nv, nv))
        mujoco.mj_fullM(m, M, d.qM)
        r = M @ vdot + d.qfrc_bias - d.qfrc_passive - known_qfrc

        G = self.loop_jacobian(d)
        ST = np.zeros((nv, 3))
        ST[self.hip_dofs, np.arange(3)] = 1.0
        Jc, f_meas = self.contact_jacobian_and_force(d)
        blocks = [ST, G.T]
        if Jc is not None:
            blocks.append(Jc.T)
        A = np.hstack(blocks)
        x, *_ = np.linalg.lstsq(A, r, rcond=None)
        res = float(np.linalg.norm(A @ x - r))
        tau = x[:3]
        lam = x[3:12]
        fc = x[12:15] if Jc is not None else None
        return tau, lam, fc, res, f_meas


# ---------------------------------------------------------------------------
# verification: torque replay on a drop + squat motion
# ---------------------------------------------------------------------------
def run_verification(duration: float, plot_path: str, freq_hz: float = 1.5,
                     amp: float = 0.35):
    model = mujoco.MjModel.from_xml_path(XML)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    fid = FullOrderID(model)

    hip_act = [model.actuator(f"ctrl_motor_{i}").id for i in (1, 2, 3)]
    q_home = data.ctrl[hip_act[0]]

    n_steps = int(duration / model.opt.timestep)
    rec = {k: [] for k in ("t", "tau_true", "tau_id", "tau_static", "res",
                           "contact", "fz", "fc_id")}

    for step in range(n_steps):
        t = step * model.opt.timestep
        # squat: settle 1 s after the drop, then sinusoidal hip oscillation
        a = amp if t > 1.0 else 0.0
        q_cmd = q_home + a * np.sin(2 * np.pi * freq_hz * (t - 1.0))
        for a in hip_act:
            data.ctrl[a] = q_cmd
        mujoco.mj_step(model, data)

        if step % 4 != 0:  # sample at 500 Hz
            continue
        # after mj_step, qacc/qfrc_* belong to the pre-integration state;
        # recompute everything consistently at the *current* (q, v, ctrl)
        mujoco.mj_forward(model, data)

        # ground truth: what the hip actuators actually applied
        tau_true = data.qfrc_actuator[fid.hip_dofs].copy()
        # known inputs = all actuator forces except the hip rows
        known = data.qfrc_actuator.copy()
        known[fid.hip_dofs] = 0.0

        tau_id, lam, fc_id, res, f_meas = fid.solve(data, data.qacc, known)

        # static Jacobian map (what the controller uses today):
        # quasi-static hip torque resisting the measured GRF
        if f_meas is not None:
            Jleg = fid.leg_jacobian_world(data)
            tau_static = -Jleg.T @ f_meas
            fz = f_meas[2]
        else:
            tau_static = np.zeros(3)
            fz = 0.0

        rec["t"].append(t)
        rec["tau_true"].append(tau_true)
        rec["tau_id"].append(tau_id)
        rec["tau_static"].append(tau_static)
        rec["res"].append(res)
        rec["contact"].append(f_meas is not None)
        rec["fz"].append(fz)
        rec["fc_id"].append(fc_id if fc_id is not None else np.zeros(3))

    t = np.array(rec["t"])
    tau_true = np.array(rec["tau_true"])
    tau_id = np.array(rec["tau_id"])
    tau_static = np.array(rec["tau_static"])
    contact = np.array(rec["contact"])

    err_id = tau_id - tau_true
    err_st = tau_static - tau_true
    # skip the initial drop transient for the summary numbers
    sel = t > 0.6
    sel_st = sel & contact
    rms_id = np.sqrt((err_id[sel] ** 2).mean())
    rms_st = np.sqrt((err_st[sel_st] ** 2).mean())
    rms_id_st = np.sqrt((err_id[sel_st] ** 2).mean())
    print(f"samples: {sel.sum()} total, {sel_st.sum()} in stance")
    print(f"tau range (true, stance): [{tau_true[sel_st].min():+.2f}, {tau_true[sel_st].max():+.2f}] Nm")
    print(f"RMS error full-order ID  (all)    : {rms_id:.4f} Nm")
    print(f"RMS error full-order ID  (stance) : {rms_id_st:.4f} Nm")
    print(f"RMS error static Jac map (stance) : {rms_st:.4f} Nm")
    print(f"mean lstsq residual              : {np.mean(rec['res']):.4f}")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(4, 1, figsize=(11, 11), sharex=True)
        for k in range(3):
            ax = axes[k]
            ax.plot(t, tau_true[:, k], "k-", lw=1.6, label="true (actuator)")
            ax.plot(t, tau_id[:, k], "r--", lw=1.2, label="full-order ID")
            ax.plot(t, tau_static[:, k], "b:", lw=1.2, label="static Jac map")
            ax.set_ylabel(f"hip {k+1} [Nm]")
            ax.grid(alpha=0.3)
            if k == 0:
                ax.legend(loc="upper right", fontsize=9)
        axes[3].plot(t, np.array(rec["fz"]), "g-", lw=1.2, label="GRF z (measured)")
        axes[3].plot(t, np.array(rec["fc_id"])[:, 2], "m--", lw=1.0, label="GRF z (ID est)")
        axes[3].set_ylabel("F_z [N]")
        axes[3].set_xlabel("t [s]")
        axes[3].grid(alpha=0.3)
        axes[3].legend(loc="upper right", fontsize=9)
        fig.suptitle("3RSR full-order inverse dynamics: torque replay verification")
        fig.tight_layout()
        os.makedirs(os.path.dirname(plot_path), exist_ok=True)
        fig.savefig(plot_path, dpi=110)
        print(f"plot -> {plot_path}")
    except Exception as e:  # matplotlib optional
        print(f"(plot skipped: {e})")

    return rms_id, rms_id_st, rms_st


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=6.0)
    ap.add_argument("--freq", type=float, default=1.5)
    ap.add_argument("--amp", type=float, default=0.35)
    ap.add_argument("--plot", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "demos", "full_order_id_verify.png"))
    args = ap.parse_args()
    run_verification(args.duration, args.plot, args.freq, args.amp)
