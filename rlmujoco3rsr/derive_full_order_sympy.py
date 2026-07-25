#!/usr/bin/env python3
"""Analytical (closed-form) full-order dynamics of the 3RSR hopper.

Symbolically derives, with SymPy, the open-tree floating-base dynamics of
FULL_ORDER_3RSR.md Sec. 3:

    M(q) q'' + h(q, q') = tau_gen,      h = C q' + g       (24 coordinates)
    Phi(q) = 0,  G(q) = dPhi/dq  in R^{9 x 24}             (loop closure)

Generalized coordinates (Euler ZYX for the two free bodies, so plain
Lagrangian mechanics applies -- no quaternion/quasi-velocity subtleties):

    q = [ p_b(3), (roll,pitch,yaw)_b,          base
          (theta_i, alpha_i, beta_i) i=1..3,   RSR chains (hip, cross, lower)
          sigma_1..3,                          prop tilt servos
          p_f(3), (roll,pitch,yaw)_f ]         foot free body

Kinematic structure is closed-form symbolic; the numeric inertial/geometric
parameters (masses, COMs, inertia tensors, fixed transforms, joint axes,
armature) are read from the compiled MuJoCo model so they cannot drift from
the plant.

Derivation method (per body b, all closed form):
    v_b   = Jv_b(q) q',        Jv_b = d p_com,b / dq
    w_b   = Jw_b(q) q',        from unskew(Rdot R^T)
    M     = sum_b  m_b Jv^T Jv + Jw^T I_w Jw          (+ rotor armature)
    h     = sum_b  Jv^T m_b (a_vp - g) + Jw^T (I_w al_vp + w x I_w w)
            where a_vp, al_vp are the velocity-product accelerations
            (d/dt of v, w holding q'' = 0).

The script then VERIFIES the analytical model against MuJoCo at random
states (q, q', q'') through the velocity map  v_mj = E(q) q'_euler :

    M_e  ==  E^T M_mj E
    tau_e = M_e q'' + h  ==  E^T ( M_mj (E q'' + Edot q') + qfrc_bias )
    G_e  ==  G_mj E

and finally code-generates a standalone numpy module `gen_full_order_dyn.py`
(no sympy dependency at runtime) for online feedforward use.

Usage: python3 derive_full_order_sympy.py [--ntest 20] [--no-gen]
"""
import argparse
import os
import time

import numpy as np
import sympy as sp
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
XML = os.path.join(HERE, "three_leg_3rsr_closed.xml")
GEN = os.path.join(HERE, "gen_full_order_dyn.py")

GRAV = 9.81


# ---------------------------------------------------------------------------
# symbolic helpers
# ---------------------------------------------------------------------------
def rot_x(a):
    c, s = sp.cos(a), sp.sin(a)
    return sp.Matrix([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = sp.cos(a), sp.sin(a)
    return sp.Matrix([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = sp.cos(a), sp.sin(a)
    return sp.Matrix([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rot_axis(axis, a):
    """Rodrigues rotation about a (numeric) unit axis by symbolic angle."""
    ax = np.asarray(axis, dtype=float)
    if np.allclose(ax, [1, 0, 0]):
        return rot_x(a)
    if np.allclose(ax, [0, 1, 0]):
        return rot_y(a)
    if np.allclose(ax, [0, 0, 1]):
        return rot_z(a)
    k = sp.Matrix(ax)
    K = skew(k)
    return sp.eye(3) + sp.sin(a) * K + (1 - sp.cos(a)) * K * K


def skew(v):
    return sp.Matrix([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])


def unskew(S):
    return sp.Matrix([S[2, 1], S[0, 2], S[1, 0]])


def euler_zyx(roll, pitch, yaw):
    """R = Rz(yaw) Ry(pitch) Rx(roll)."""
    return rot_z(yaw) * rot_y(pitch) * rot_x(roll)


def quat2mat_np(quat_wxyz):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(quat_wxyz, dtype=float))
    return m.reshape(3, 3)


# ---------------------------------------------------------------------------
# model construction
# ---------------------------------------------------------------------------
class Body:
    def __init__(self, name, R_w, p_w, mass, ipos, iR, idiag):
        self.name = name
        self.R_w = R_w          # world orientation, symbolic 3x3
        self.p_w = p_w          # world origin, symbolic 3x1
        self.mass = float(mass)
        self.ipos = np.asarray(ipos, dtype=float)    # COM in body frame
        self.iR = np.asarray(iR, dtype=float)        # inertial frame in body
        self.idiag = np.asarray(idiag, dtype=float)  # principal inertia

    @property
    def p_com(self):
        return self.p_w + self.R_w * sp.Matrix(self.ipos)

    @property
    def I_world(self):
        """Inertia about COM in world frame (symbolic via R_w)."""
        I_body = sp.Matrix(self.iR @ np.diag(self.idiag) @ self.iR.T)
        return self.R_w * I_body * self.R_w.T


def build_symbolic_model():
    model = mujoco.MjModel.from_xml_path(XML)

    def fixed(name):
        b = model.body(name)
        return quat2mat_np(b.quat), np.array(b.pos, dtype=float)

    def inert(name):
        b = model.body(name)
        return float(b.mass[0]), b.ipos.copy(), quat2mat_np(b.iquat), b.inertia.copy()

    def jaxis(name):
        return model.joint(name).axis.copy()

    # ---- coordinates ----
    x, y, z = sp.symbols("x y z")
    ph, th, ps = sp.symbols("phi theta psi")            # base roll/pitch/yaw
    legq = [sp.symbols(f"th{i} al{i} be{i}") for i in (1, 2, 3)]
    sig = list(sp.symbols("sg1 sg2 sg3"))
    xf, yf, zf = sp.symbols("xf yf zf")
    phf, thf, psf = sp.symbols("phif thetaf psif")

    q = [x, y, z, ph, th, ps]
    for trio in legq:
        q += list(trio)
    q += sig + [xf, yf, zf, phf, thf, psf]
    q = sp.Matrix(q)                                     # 24x1
    qd = sp.Matrix(sp.symbols(" ".join(f"dq{k}" for k in range(24))))

    # ---- base ----
    p_b = sp.Matrix([x, y, z])
    R_b = euler_zyx(ph, th, ps)
    bodies = [Body("base", R_b, p_b, *inert("base"))]

    chain_sites_sym = []      # world position of foot_site_i (symbolic)
    for i in (1, 2, 3):
        thi, ali, bei = legq[i - 1]
        Rl, pl = fixed(f"leg{i}")
        R_leg = R_b * sp.Matrix(Rl)
        p_leg = p_b + R_b * sp.Matrix(pl)
        Rh, phh = fixed(f"hip{i}")
        R_hip = R_leg * sp.Matrix(Rh)
        p_hip = p_leg + R_leg * sp.Matrix(phh)
        Ru, pu = fixed(f"upper{i}")
        R_up = R_hip * sp.Matrix(Ru) * rot_axis(jaxis(f"ctrl_joint_{i}"), thi)
        p_up = p_hip + R_hip * sp.Matrix(pu)
        bodies.append(Body(f"upper{i}", R_up, p_up, *inert(f"upper{i}")))
        Rc, pc = fixed(f"cross{i}")
        R_cr = R_up * sp.Matrix(Rc) * rot_axis(jaxis(f"cross_pin_{i}"), ali)
        p_cr = p_up + R_up * sp.Matrix(pc)
        bodies.append(Body(f"cross{i}", R_cr, p_cr, *inert(f"cross{i}")))
        Rlo, plo = fixed(f"lower{i}")
        R_lo = R_cr * sp.Matrix(Rlo) * rot_axis(jaxis(f"lower_pin_{i}"), bei)
        p_lo = p_cr + R_cr * sp.Matrix(plo)
        bodies.append(Body(f"lower{i}", R_lo, p_lo, *inert(f"lower{i}")))
        site_local = model.site(f"foot_site_{i}").pos.copy()
        chain_sites_sym.append(p_lo + R_lo * sp.Matrix(site_local))

    prop_frames = []
    for i in (1, 2, 3):
        Ra, pa = fixed(f"proparm{i}")
        R_arm = R_b * sp.Matrix(Ra)
        p_arm = p_b + R_b * sp.Matrix(pa)
        Rp, pp = fixed(f"prop{i}")
        R_pr = R_arm * sp.Matrix(Rp) * rot_axis(jaxis(f"servo_{i}"), sig[i - 1])
        p_pr = p_arm + R_arm * sp.Matrix(pp)
        bodies.append(Body(f"prop{i}", R_pr, p_pr, *inert(f"prop{i}")))
        prop_frames.append((R_pr, p_pr))

    p_f = sp.Matrix([xf, yf, zf])
    R_f = euler_zyx(phf, thf, psf)
    bodies.append(Body("foot", R_f, p_f, *inert("foot")))

    # ---- loop closure Phi and its Jacobian ----
    Phi = sp.Matrix.vstack(*[s - p_f for s in chain_sites_sym])  # 9x1
    G = Phi.jacobian(q)                                          # 9x24

    # ---- mass matrix and bias (per-body geometric Jacobians) ----
    print("assembling M and h symbolically ...")
    t0 = time.time()
    M = sp.zeros(24, 24)
    h = sp.zeros(24, 1)
    g_vec = sp.Matrix([0, 0, -GRAV])
    for b in bodies:
        Jv = b.p_com.jacobian(q)                    # 3x24
        Rd = sum((b.R_w.diff(q[k]) * qd[k] for k in range(24)), sp.zeros(3, 3))
        w = unskew(Rd * b.R_w.T)                    # world angular velocity
        Jw = w.jacobian(qd)                         # 3x24
        v = Jv * qd
        # velocity-product accelerations (qdd = 0)
        a_vp = sum((v.diff(q[k]) * qd[k] for k in range(24)), sp.zeros(3, 1))
        al_vp = sum((w.diff(q[k]) * qd[k] for k in range(24)), sp.zeros(3, 1))
        Iw = b.I_world
        M += b.mass * Jv.T * Jv + Jw.T * Iw * Jw
        h += Jv.T * (b.mass * (a_vp - g_vec)) + Jw.T * (Iw * al_vp + w.cross(Iw * w))
        print(f"  {b.name:8s} done  ({time.time()-t0:.1f} s)")

    # rotor armature on the 12 hinge coordinates (identity-mapped dofs)
    dof_armature = np.zeros(24)
    for jname in ([f"ctrl_joint_{i}" for i in (1, 2, 3)]
                  + [f"cross_pin_{i}" for i in (1, 2, 3)]
                  + [f"lower_pin_{i}" for i in (1, 2, 3)]
                  + [f"servo_{i}" for i in (1, 2, 3)]):
        j = model.joint(jname)
        # euler coordinate index == mujoco dof index for hinges (see mapping)
        dof_armature[model.jnt_dofadr[j.id]] = model.dof_armature[model.jnt_dofadr[j.id]]
    M += sp.diag(*[float(a) for a in dof_armature])

    # ---- thrust input matrix B_p: tau_gen = B_p(q) u,  u = rotor thrusts ----
    # Each rotor: force u*d_i and reaction drag torque km_i*u*d_i, with
    # d_i = prop frame +X in world (thrust axis, tilted by servo sigma_i).
    B_p = sp.zeros(24, 3)
    for i, (R_pr, p_pr) in enumerate(prop_frames):
        gear = model.actuator(f"thrust_{i+1}").gear
        km = float(gear[3])                     # +-0.018 m drag/thrust ratio
        d = R_pr[:, 0]                          # world thrust direction
        Jv = p_pr.jacobian(q)
        Rd_p = sum((R_pr.diff(q[k]) * qd[k] for k in range(24)), sp.zeros(3, 3))
        Jw = unskew(Rd_p * R_pr.T).jacobian(qd)
        B_p[:, i] = Jv.T * d + km * (Jw.T * d)

    # ---- velocity map E: v_mujoco = E(q) qdot_euler ----
    # world angular velocity of base/foot as function of euler rates:
    Rd_b = sum((R_b.diff(q[k]) * qd[k] for k in range(24)), sp.zeros(3, 3))
    Ww_b = unskew(Rd_b * R_b.T).jacobian(qd)[:, 3:6]     # 3x3
    Rd_f = sum((R_f.diff(q[k]) * qd[k] for k in range(24)), sp.zeros(3, 3))
    Ww_f = unskew(Rd_f * R_f.T).jacobian(qd)[:, 21:24]
    E_world = sp.eye(24)
    E_world[3:6, 3:6] = Ww_b
    E_world[21:24, 21:24] = Ww_f
    E_body = sp.eye(24)
    E_body[3:6, 3:6] = R_b.T * Ww_b
    E_body[21:24, 21:24] = R_f.T * Ww_f

    return model, q, qd, M, h, G, Phi, B_p, E_world, E_body, R_b, R_f


# ---------------------------------------------------------------------------
# verification against MuJoCo
# ---------------------------------------------------------------------------
def verify(model, q, qd, M, h, G, Phi, B_p, E_world, E_body, R_b, R_f, ntest):
    print("lambdifying (cse) ...")
    t0 = time.time()
    f_M = sp.lambdify((q,), M, "numpy", cse=True)
    f_h = sp.lambdify((q, qd), h, "numpy", cse=True)
    f_G = sp.lambdify((q,), G, "numpy", cse=True)
    f_B = sp.lambdify((q,), B_p, "numpy", cse=True)
    f_Ew = sp.lambdify((q,), E_world, "numpy", cse=True)
    f_Eb = sp.lambdify((q,), E_body, "numpy", cse=True)
    f_Rb = sp.lambdify((q,), R_b, "numpy", cse=True)
    f_Rf = sp.lambdify((q,), R_f, "numpy", cse=True)
    # Edot*qd term: d(E qd)/dq * qd
    Eqd = E_world * qd
    f_Edqd_w = sp.lambdify((q, qd), Eqd.jacobian(q) * qd, "numpy", cse=True)
    Eqd_b = E_body * qd
    f_Edqd_b = sp.lambdify((q, qd), Eqd_b.jacobian(q) * qd, "numpy", cse=True)
    print(f"  lambdify done ({time.time()-t0:.1f} s)")

    data = mujoco.MjData(model)
    hip_names = [f"ctrl_joint_{i}" for i in (1, 2, 3)]
    site_ids = [model.site(f"foot_site_{i}").id for i in (1, 2, 3)]
    foot_sid = model.site("foot_center").id
    rng = np.random.default_rng(7)

    thrust_act = [model.actuator(f"thrust_{i}").id for i in (1, 2, 3)]
    errs = {"M_w": [], "M_b": [], "tau_w": [], "tau_b": [], "G_w": [], "G_b": [],
            "B_w": [], "B_b": []}
    for _ in range(ntest):
        qe = np.zeros(24)
        qe[0:3] = rng.uniform(-0.5, 0.5, 3) + [0, 0, 2.0]   # base high: no contact
        qe[3:6] = rng.uniform(-1.0, 1.0, 3)
        qe[6:15] = rng.uniform(-0.6, 0.6, 9)
        qe[15:18] = rng.uniform(-1.0, 1.0, 3)
        qe[18:21] = rng.uniform(-0.5, 0.5, 3) + [0, 0, 1.0]
        qe[21:24] = rng.uniform(-1.0, 1.0, 3)
        qde = rng.uniform(-3.0, 3.0, 24)
        qdde = rng.uniform(-20.0, 20.0, 24)

        Rb = np.asarray(f_Rb(qe), dtype=float)
        Rf = np.asarray(f_Rf(qe), dtype=float)
        quat_b = np.zeros(4)
        mujoco.mju_mat2Quat(quat_b, Rb.flatten())
        quat_f = np.zeros(4)
        mujoco.mju_mat2Quat(quat_f, Rf.flatten())
        qpos = np.concatenate([qe[0:3], quat_b, qe[6:18], qe[18:21], quat_f])
        assert qpos.size == model.nq

        for conv, f_E, f_Edqd in (("w", f_Ew, f_Edqd_w), ("b", f_Eb, f_Edqd_b)):
            E = np.asarray(f_E(qe), dtype=float)
            v_mj = E @ qde
            data.qpos[:] = qpos
            data.qvel[:] = v_mj
            mujoco.mj_forward(model, data)

            M_mj = np.zeros((model.nv, model.nv))
            mujoco.mj_fullM(model, M_mj, data.qM)
            M_e = np.asarray(f_M(qe), dtype=float)
            errs[f"M_{conv}"].append(np.abs(M_e - E.T @ M_mj @ E).max())

            vdot_mj = E @ qdde + np.asarray(f_Edqd(qe, qde), dtype=float).flatten()
            tau_mj = M_mj @ vdot_mj + data.qfrc_bias
            tau_e = (np.asarray(f_M(qe), dtype=float) @ qdde
                     + np.asarray(f_h(qe, qde), dtype=float).flatten())
            errs[f"tau_{conv}"].append(np.abs(tau_e - E.T @ tau_mj).max())

            jf = np.zeros((3, model.nv))
            mujoco.mj_jacSite(model, data, jf, None, foot_sid)
            G_mj = np.zeros((9, model.nv))
            js = np.zeros((3, model.nv))
            for k, sid in enumerate(site_ids):
                js[:] = 0
                mujoco.mj_jacSite(model, data, js, None, sid)
                G_mj[3 * k:3 * k + 3] = js - jf
            G_e = np.asarray(f_G(qe), dtype=float)
            errs[f"G_{conv}"].append(np.abs(G_e - G_mj @ E).max())

            # thrust input matrix: qfrc_actuator is linear in u; isolate it
            # by differencing u = e_i against u = 0 at fixed state.
            data.ctrl[:] = 0.0
            mujoco.mj_forward(model, data)
            f0 = data.qfrc_actuator.copy()
            B_mj = np.zeros((model.nv, 3))
            for k, aid in enumerate(thrust_act):
                data.ctrl[:] = 0.0
                data.ctrl[aid] = 1.0
                mujoco.mj_forward(model, data)
                B_mj[:, k] = data.qfrc_actuator - f0
            data.ctrl[:] = 0.0
            B_e = np.asarray(f_B(qe), dtype=float)
            errs[f"B_{conv}"].append(np.abs(B_e - E.T @ B_mj).max())

    print(f"\nverification over {ntest} random states (max abs error):")
    for conv, label in (("w", "free-joint ang vel = WORLD frame"),
                        ("b", "free-joint ang vel = BODY frame")):
        print(f"  assuming {label}:")
        print(f"    M   : {max(errs[f'M_{conv}']):.3e}")
        print(f"    tau : {max(errs[f'tau_{conv}']):.3e}   (full inverse dynamics)")
        print(f"    G   : {max(errs[f'G_{conv}']):.3e}")
        print(f"    B_p : {max(errs[f'B_{conv}']):.3e}   (thrust input matrix)")
    return errs


# ---------------------------------------------------------------------------
# code generation
# ---------------------------------------------------------------------------
def _emit(fh, fname, args, mats):
    """Write one numpy function computing the given dict of sympy matrices.

    args: list of (argname, list-of-symbols) -- flat arrays unpacked to the
    original symbol names so the CSE'd expressions resolve.
    """
    from sympy.printing.numpy import NumPyPrinter
    pr = NumPyPrinter()
    all_exprs, shapes = [], []
    for name, m in mats.items():
        shapes.append((name, m.shape))
        all_exprs.extend(list(m))
    # plain cse: the "basic" optimization pass is O(minutes) on ~1e6-op
    # expression sets for negligible runtime gain
    repl, red = sp.cse(all_exprs)
    fh.write(f"\n\ndef {fname}({', '.join(a for a, _ in args)}):\n")
    for a, syms in args:
        fh.write(f"    ({', '.join(str(s) for s in syms)},) = {a}\n")
    for s, e in repl:
        fh.write(f"    {s} = {pr.doprint(e)}\n")
    k = 0
    outs = []
    for name, shape in shapes:
        fh.write(f"    {name} = numpy.zeros({shape})\n")
        for r in range(shape[0]):
            for c in range(shape[1]):
                if red[k] != 0:
                    fh.write(f"    {name}[{r},{c}] = {pr.doprint(red[k])}\n")
                k += 1
        outs.append(name)
    fh.write(f"    return {', '.join(outs)}\n")


def generate(q, qd, M, h, G, Phi, B_p, E_world, E_body):
    print("generating gen_full_order_dyn.py ...")
    t0 = time.time()
    # acceleration-level constraint term: Gdot(q,qd) qd = d(G qd)/dq qd
    Gd_qd = (G * qd).jacobian(q) * qd
    qs = [str(s) for s in q]
    tmp = GEN + ".tmp"
    with open(tmp, "w") as fh:
        fh.write('"""AUTO-GENERATED by derive_full_order_sympy.py -- do not edit.\n\n'
                 "Closed-form full-order dynamics of the 3RSR hopper (24 Euler\n"
                 "coordinates, see FULL_ORDER_3RSR.md).  q order:\n"
                 f"  {qs}\n"
                 "qd = matching rates.  All functions take flat float arrays.\n\n"
                 "  mass_matrix(q)      -> M  (24x24), includes rotor armature\n"
                 "  bias_force(q, qd)   -> h  (24x1) = C(q,qd) qd + g(q)\n"
                 "  constraint_jac(q)   -> G  (9x24) loop closure dPhi/dq\n"
                 "  chain_phi(q)        -> Phi (9x1) loop closure residual\n"
                 "  constraint_gdot_qd(q, qd) -> (9x1) Gdot(q,qd) qd\n"
                 "  input_matrix(q)     -> B  (24x3) rotor thrust -> gen. force\n"
                 "  vel_map_body(q)     -> E  (24x24), v_mujoco = E qd (MuJoCo\n"
                 "                         free-joint convention: lin world, ang body)\n"
                 '"""\nimport numpy\n')
        aq = ("q", list(q))
        aqd = ("qd", list(qd))
        for fname, arglist, mats in (
                ("mass_matrix", [aq], {"M": M}),
                ("bias_force", [aq, aqd], {"h": h}),
                ("constraint_jac", [aq], {"G": G}),
                ("chain_phi", [aq], {"Phi": Phi}),
                ("constraint_gdot_qd", [aq, aqd], {"Gdqd": Gd_qd}),
                ("input_matrix", [aq], {"B": B_p}),
                ("vel_map_world", [aq], {"E": E_world}),
                ("vel_map_body", [aq], {"E": E_body})):
            _emit(fh, fname, arglist, mats)
            print(f"  {fname} emitted ({time.time()-t0:.1f} s)")
    os.replace(tmp, GEN)
    print(f"  wrote {GEN} ({os.path.getsize(GEN)/1024:.0f} KB, {time.time()-t0:.1f} s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ntest", type=int, default=20)
    ap.add_argument("--no-gen", action="store_true")
    args = ap.parse_args()

    t0 = time.time()
    model, q, qd, M, h, G, Phi, B_p, E_w, E_b, R_b, R_f = build_symbolic_model()
    print(f"symbolic model built in {time.time()-t0:.1f} s")
    ops = sum(e.count_ops() for e in M) + sum(e.count_ops() for e in h)
    print(f"expression size: M+h = {ops} ops before CSE")

    verify(model, q, qd, M, h, G, Phi, B_p, E_w, E_b, R_b, R_f, args.ntest)
    if not args.no_gen:
        generate(q, qd, M, h, G, Phi, B_p, E_w, E_b)


if __name__ == "__main__":
    main()
