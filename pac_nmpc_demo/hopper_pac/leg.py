"""
Serial-equivalent leg kinematics (roll / pitch / shift).

Math copied from ModeECore._serial_leg_fk_jac (core.py) so the demo controller
sees the leg exactly like the runtime stack does.

Base frame: +X forward, +Y left, +Z up.  Foot hangs below the base (z < 0).
q_shift > 0 SHORTENS the leg.
"""

from __future__ import annotations

import numpy as np

from .conventions import SERIAL_HIP_Z_OFF_M, SERIAL_FOOT_Z_M, LEG_L0_M


def fk_jac(q_roll: float, q_pitch: float, q_shift: float) -> tuple[np.ndarray, np.ndarray]:
    """Foot position in base frame + Jacobian (qdot -> foot velocity in base frame)."""
    p0 = np.array([0.0, 0.0, -SERIAL_HIP_Z_OFF_M])
    cr, sr = np.cos(q_roll), np.sin(q_roll)
    cp, sp = np.cos(q_pitch), np.sin(q_pitch)
    Rr = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    Rp = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    R = Rr @ Rp

    v = np.array([0.0, 0.0, float(q_shift) - SERIAL_FOOT_Z_M])
    foot_rel = R @ v
    foot_b = p0 + foot_rel

    axis_roll = np.array([1.0, 0.0, 0.0])
    axis_pitch = Rr @ np.array([0.0, 1.0, 0.0])
    axis_shift = R[:, 2]
    J = np.stack(
        [np.cross(axis_roll, foot_rel), np.cross(axis_pitch, foot_rel), axis_shift],
        axis=1,
    )
    return foot_b, J


def leg_length(q: np.ndarray) -> float:
    foot_b, _ = fk_jac(float(q[0]), float(q[1]), float(q[2]))
    return float(np.linalg.norm(foot_b))


def ik(foot_b_des: np.ndarray) -> np.ndarray:
    """
    Inverse kinematics for the serial leg.

    Geometry: foot_b - p0 = -L_eff * [sin(pitch), -sin(roll)cos(pitch), cos(roll)cos(pitch)]
    with L_eff = SERIAL_FOOT_Z_M - q_shift.
    """
    p0 = np.array([0.0, 0.0, -SERIAL_HIP_Z_OFF_M])
    w = np.asarray(foot_b_des, dtype=float).reshape(3) - p0
    L_eff = float(np.linalg.norm(w))
    L_eff = max(L_eff, 1e-6)
    sp = float(np.clip(-w[0] / L_eff, -1.0, 1.0))
    pitch = float(np.arcsin(sp))
    cp = float(np.cos(pitch))
    sr = float(np.clip(w[1] / (L_eff * max(cp, 1e-6)), -1.0, 1.0))
    roll = float(np.arcsin(sr))
    q_shift = SERIAL_FOOT_Z_M - L_eff
    return np.array([roll, pitch, q_shift])


def grf_to_tau(f_c_w: np.ndarray, R_wb: np.ndarray, q: np.ndarray) -> np.ndarray:
    """
    Map desired world-frame ground reaction force ON THE BODY to joint torques.
    tau = J^T f_leg_b with f_leg_b = -R_wb^T f_c_w   (Eq. leg_mapping in the CASE paper).
    """
    _, J = fk_jac(float(q[0]), float(q[1]), float(q[2]))
    f_leg_b = -(np.asarray(R_wb).T @ np.asarray(f_c_w, dtype=float).reshape(3))
    return J.T @ f_leg_b


__all__ = ["fk_jac", "leg_length", "ik", "grf_to_tau", "LEG_L0_M"]
