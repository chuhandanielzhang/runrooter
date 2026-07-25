#!/usr/bin/env python3
"""Numerical tests for the controller-side full-order stance feedforward.

1. closure FK reproduces the MuJoCo keyframe passive angles + foot point,
2. controller-side stance_tau (FRD/LCM inputs) matches the true actuator
   torque during a simulated squat motion, given the measured GRF.

Usage: python3 test_full_order_ff.py
"""
import os
import sys

import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "upper_controller_pc", "hopper_controller"))
from modee.controllers.full_order_ff import (  # noqa: E402
    FullOrderFF, HOME_SIM, Q0_LCM, M_FRD)
from full_order_id import FullOrderID  # noqa: E402

XML = os.path.join(HERE, "three_leg_3rsr_closed.xml")
PERM = (0, 2, 1)  # MFR_PERM_BRANCH=0


def quat_to_R(q):
    m = np.zeros(9)
    mujoco.mju_quat2Mat(m, np.asarray(q, dtype=float))
    return m.reshape(3, 3)


def main():
    model = mujoco.MjModel.from_xml_path(XML)
    data = mujoco.MjData(model)
    ff = FullOrderFF(perm_branch=0)

    # ---- test 1: closure FK vs keyframe ----
    mujoco.mj_resetDataKeyframe(model, data, model.key("home").id)
    theta_sim = np.full(3, data.qpos[7])
    w = ff.closure_fk(theta_sim)
    assert w is not None, "closure FK did not converge"
    al_ref, be_ref = data.qpos[8], data.qpos[9]
    pf_b_ref = data.qpos[19:22] - data.qpos[0:3]   # foot free joint qpos adr 19
    e_ab = max(abs(w[0] - al_ref), abs(w[1] - be_ref))
    e_pf = np.abs(w[6:9] - pf_b_ref).max()
    print(f"test1 closure FK: |d(alpha,beta)|={e_ab:.2e}  |d p_f|={e_pf:.2e}")
    # keyframe values are rounded to 6 decimals in the XML
    assert e_ab < 1e-5 and e_pf < 1e-5

    # ---- test 2: stance tau vs ground truth during squat ----
    fid = FullOrderID(model)
    hip_act = [model.actuator(f"ctrl_motor_{i}").id for i in (1, 2, 3)]
    q_home = data.ctrl[hip_act[0]]
    errs, taus = [], []
    for step in range(int(5.0 / model.opt.timestep)):
        t = step * model.opt.timestep
        amp = 0.30 if t > 1.0 else 0.0
        q_cmd = q_home + amp * np.sin(2 * np.pi * 1.5 * (t - 1.0))
        for a in hip_act:
            data.ctrl[a] = q_cmd
        mujoco.mj_step(model, data)
        if step % 8 != 0 or t < 1.2:
            continue
        mujoco.mj_forward(model, data)
        _, f_meas = fid.contact_jacobian_and_force(data)
        if f_meas is None:
            continue

        # controller-side (FRD/LCM) inputs from the sim state
        q_sim = data.qpos[[7, 10, 13]]
        qd_sim = data.qvel[[6, 9, 12]]
        q_lcm = np.zeros(3)
        qd_lcm = np.zeros(3)
        for i in range(3):
            q_lcm[i] = Q0_LCM - (q_sim[PERM[i]] - HOME_SIM)
            qd_lcm[i] = -qd_sim[PERM[i]]
        R_sim = quat_to_R(data.qpos[3:7])
        R_frd = M_FRD @ R_sim @ M_FRD.T
        om_frd = M_FRD @ data.qvel[3:6]
        v_frd = M_FRD @ data.qvel[0:3]
        f_frd = M_FRD @ f_meas

        # accelerometer specific force in FRD from the sim ground truth
        R_sim3 = quat_to_R(data.qpos[3:7])
        spec_sim_b = R_sim3.T @ (data.qacc[0:3] + np.array([0.0, 0.0, 9.81]))
        acc_frd = M_FRD @ spec_sim_b
        tau_lcm = ff.stance_tau(q_lcm=q_lcm, qd_lcm=qd_lcm, R_wb_frd=R_frd,
                                omega_b_frd=om_frd, v_w_frd=v_frd,
                                f_ref_w_frd=f_frd,
                                dt=8 * model.opt.timestep, acc_b_frd=acc_frd)
        if tau_lcm is None:
            continue
        tau_true_sim = data.qfrc_actuator[fid.hip_dofs]
        tau_true_lcm = np.array([-tau_true_sim[PERM[i]] for i in range(3)])
        errs.append(tau_lcm - tau_true_lcm)
        taus.append(tau_true_lcm)

    errs = np.array(errs)
    taus = np.array(taus)
    rms = float(np.sqrt((errs ** 2).mean()))
    rng = float(np.abs(taus).max())
    print(f"test2 stance tau: {len(errs)} stance samples, "
          f"true |tau| max={rng:.2f} Nm, FF RMS err={rms:.3f} Nm, "
          f"max err={np.abs(errs).max():.3f} Nm")
    assert rms < 0.5, "full-order FF deviates too much from ground truth"
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
