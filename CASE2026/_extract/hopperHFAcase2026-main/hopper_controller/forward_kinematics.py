from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DeltaLegConfig:
    """
    Minimal delta-leg geometry parameters required by the closed-form kinematics in this file.

    NOTE: These defaults were previously provided by `hopper_config.HopperConfig`, which we removed
    when slimming the repo to ModeE-only. Keep these numbers consistent with your real robot.
    """

    D: float = 0.158   # upper arm length (m)
    d: float = 0.398   # rod length (m)
    r: float = 0.02424 # base radius (m)


class SimulinkVelocityFilter:
    """Simple EWMA velocity estimate from position (kept for backward compatibility)."""

    def __init__(self, dt: float = 0.001, forgetting_factor: float = 0.98):
        self.dt = float(dt)
        self.forgetting_factor = float(forgetting_factor)
        self.pos_prev: float | None = None
        self.ema_state: float = 0.0

    def update(self, pos: float) -> float:
        if self.pos_prev is None:
            self.pos_prev = float(pos)
            return 0.0
        raw_velocity = (float(pos) - float(self.pos_prev)) / float(self.dt)
        self.pos_prev = float(pos)
        a = float(self.forgetting_factor)
        self.ema_state = a * float(self.ema_state) + (1.0 - a) * float(raw_velocity)
        return float(self.ema_state)

    def reset(self) -> None:
        self.pos_prev = None
        self.ema_state = 0.0


class SimulinkVelocityFilterVector:
    """3-DOF wrapper for SimulinkVelocityFilter (kept for backward compatibility)."""

    def __init__(self, dt: float = 0.001, forgetting_factor: float = 0.98):
        self.filters = [
            SimulinkVelocityFilter(dt, forgetting_factor),
            SimulinkVelocityFilter(dt, forgetting_factor),
            SimulinkVelocityFilter(dt, forgetting_factor),
        ]

    def update(self, q_curr: np.ndarray) -> np.ndarray:
        q_curr = np.asarray(q_curr, dtype=float).reshape(3)
        qd_filtered = np.zeros(3, dtype=float)
        for i in range(3):
            qd_filtered[i] = float(self.filters[i].update(float(q_curr[i])))
        return qd_filtered

    def reset(self) -> None:
        for f in self.filters:
            f.reset()

# ===== Frame alignment: Delta kinematics frame -> IMU/body frame (fixed yaw offset) =====
# The analytic closed-form delta kinematics in this file uses a legacy XY frame which can be rotated
# w.r.t. the robot's desired IMU/body frame. We align the kinematics outputs to the IMU/body axes by
# applying a fixed yaw rotation about +Z.
#
# Clockwise (viewed from +Z) is negative yaw in right-handed convention.
KIN_YAW_OFFSET_DEG = -30.0

# ===== Motor index convention alignment (physical motors -> analytic model) =====
# The closed-form delta kinematics below was derived with a specific motor numbering.
# We permute the input motor vector BEFORE evaluating the analytic formulas.
# "model <- physical" means: theta_model[i] = theta_physical[PERM_MODEL_FROM_PHYS[i]]
PERM_MODEL_FROM_PHYS = np.array([1, 2, 0], dtype=int)   # [model0, model1, model2] <- [phys1, phys2, phys0]
PERM_PHYS_FROM_MODEL = np.array([2, 0, 1], dtype=int)   # inverse permutation (rows back to physical order)


def _Rz(rad: float) -> np.ndarray:
    c = float(np.cos(float(rad)))
    s = float(np.sin(float(rad)))
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)


_R_NEW_FROM_OLD = _Rz(np.deg2rad(KIN_YAW_OFFSET_DEG))     # old -> new (apply to positions/vels)
_R_OLD_FROM_NEW = _R_NEW_FROM_OLD.T                      # new -> old (orthonormal)

class ForwardKinematics:
    
    def __init__(self, config=None):
        if config is None:
            config = DeltaLegConfig()
        self.D = float(getattr(config, "D"))
        self.d = float(getattr(config, "d"))
        self.r = float(getattr(config, "r"))
        
    def forward_kinematics(self, theta):
        theta = np.asarray(theta, dtype=float).reshape(3)
        # Reorder to the analytic model's motor numbering
        theta = theta[PERM_MODEL_FROM_PHYS]
        s1 = np.sin(theta[0])
        s2 = np.sin(theta[1])
        s3 = np.sin(theta[2])
        c1 = np.cos(theta[0])
        c2 = np.cos(theta[1])
        c3 = np.cos(theta[2])
        
        pos = np.zeros(3)
        D, d, r = self.D, self.d, self.r
        

        pos[0] = (3 ** (1 / 2) * (3 * r ** 3 + D ** 3 * c1 * c2 ** 2 + D ** 3 * c2 ** 2 * c3 + 2 * D * r ** 2 * c1 + 3 * D * r ** 2 * c2 + 4 * D * r ** 2 * c3 + 2 * D ** 2 * r * c2 ** 2 + D ** 2 * r * c1 * c2 + 3 * D ** 2 * r * c1 * c3 + 3 * D ** 2 * r * c2 * c3 + D ** 3 * c1 * c2 * c3)) / (2 * (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3)) - (3 ** (1 / 2) * r) / 2 - (3 ** (1 / 2) * D * c2) / 2 + (3 ** (1 / 2) * D * (((2 * D * s2 + (((c2 * s1 + c3 * s1 - c2 * s3 - c3 * s2) * D ** 2 - r * (s2 - 2 * s1 + s3) * D) * (3 * r ** 3 + D ** 3 * c1 * c2 ** 2 + D ** 3 * c2 ** 2 * c3 + 6 * D * r ** 2 * c1 + 3 * D * r ** 2 * c2 + 2 * D ** 2 * r * c2 ** 2 + 5 * D ** 2 * r * c1 * c2 + 3 * D ** 2 * r * c1 * c3 - D ** 2 * r * c2 * c3 + D ** 3 * c1 * c2 * c3)) / (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3) ** 2 - (D * (3 * D * np.sin(theta[0] + theta[1]) - 3 * D * np.sin(theta[0] + theta[2]) + 6 * r * s2 - 6 * r * s3 - D * np.sin(theta[0] - theta[1]) + D * np.sin(theta[0] - theta[2]) + 2 * D * np.sin(theta[1] - theta[2])) * (3 * r ** 3 + D ** 3 * c1 * c2 ** 2 + D ** 3 * c2 ** 2 * c3 + 2 * D * r ** 2 * c1 + 3 * D * r ** 2 * c2 + 4 * D * r ** 2 * c3 + 2 * D ** 2 * r * c2 ** 2 + D ** 2 * r * c1 * c2 + 3 * D ** 2 * r * c1 * c3 + 3 * D ** 2 * r * c2 * c3 + D ** 3 * c1 * c2 * c3)) / (2 * (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3) ** 2)) ** 2 - ((4 * ((c2 * s1 + c3 * s1 - c2 * s3 - c3 * s2) * D ** 2 - r * (s2 - 2 * s1 + s3) * D) ** 2) / (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3) ** 2 + (D ** 2 * (3 * D * np.sin(theta[0] + theta[1]) - 3 * D * np.sin(theta[0] + theta[2]) + 6 * r * s2 - 6 * r * s3 - D * np.sin(theta[0] - theta[1]) + D * np.sin(theta[0] - theta[2]) + 2 * D * np.sin(theta[1] - theta[2])) ** 2) / (3 * (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3) ** 2) + 4) * ((3 * r ** 3 + D ** 3 * c1 * c2 ** 2 + D ** 3 * c2 ** 2 * c3 + 6 * D * r ** 2 * c1 + 3 * D * r ** 2 * c2 + 2 * D ** 2 * r * c2 ** 2 + 5 * D ** 2 * r * c1 * c2 + 3 * D ** 2 * r * c1 * c3 - D ** 2 * r * c2 * c3 + D ** 3 * c1 * c2 * c3) ** 2 / (4 * (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3) ** 2) - D ** 2 * (c2 ** 2 - 1) - d ** 2 + (3 * (3 * r ** 3 + D ** 3 * c1 * c2 ** 2 + D ** 3 * c2 ** 2 * c3 + 2 * D * r ** 2 * c1 + 3 * D * r ** 2 * c2 + 4 * D * r ** 2 * c3 + 2 * D ** 2 * r * c2 ** 2 + D ** 2 * r * c1 * c2 + 3 * D ** 2 * r * c1 * c3 + 3 * D ** 2 * r * c2 * c3 + D ** 3 * c1 * c2 * c3) ** 2) / (4 * (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3) ** 2))) ** (1 / 2) / 2 + D * s2 + (((c2 * s1 + c3 * s1 - c2 * s3 - c3 * s2) * D ** 2 - r * (s2 - 2 * s1 + s3) * D) * (3 * r ** 3 + D ** 3 * c1 * c2 ** 2 + D ** 3 * c2 ** 2 * c3 + 6 * D * r ** 2 * c1 + 3 * D * r ** 2 * c2 + 2 * D ** 2 * r * c2 ** 2 + 5 * D ** 2 * r * c1 * c2 + 3 * D ** 2 * r * c1 * c3 - D ** 2 * r * c2 * c3 + D ** 3 * c1 * c2 * c3)) / (2 * (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3) ** 2) - (D * (3 * D * np.sin(theta[0] + theta[1]) - 3 * D * np.sin(theta[0] + theta[2]) + 6 * r * s2 - 6 * r * s3 - D * np.sin(theta[0] - theta[1]) + D * np.sin(theta[0] - theta[2]) + 2 * D * np.sin(theta[1] - theta[2])) * (3 * r ** 3 + D ** 3 * c1 * c2 ** 2 + D ** 3 * c2 ** 2 * c3 + 2 * D * r ** 2 * c1 + 3 * D * r ** 2 * c2 + 4 * D * r ** 2 * c3 + 2 * D ** 2 * r * c2 ** 2 + D ** 2 * r * c1 * c2 + 3 * D ** 2 * r * c1 * c3 + 3 * D ** 2 * r * c2 * c3 + D ** 3 * c1 * c2 * c3)) / (4 * (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3) ** 2)) * (3 * D * np.sin(theta[0] + theta[1]) - 3 * D * np.sin(theta[0] + theta[2]) + 6 * r * s2 - 6 * r * s3 - D * np.sin(theta[0] - theta[1]) + D * np.sin(theta[0] - theta[2]) + 2 * D * np.sin(theta[1] - theta[2]))) / (6 * (((c2 * s1 + c3 * s1 - c2 * s3 - c3 * s2) * D ** 2 - r * (s2 - 2 * s1 + s3) * D) ** 2 / (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3) ** 2 + (D ** 2 * (3 * D * np.sin(theta[0] + theta[1]) - 3 * D * np.sin(theta[0] + theta[2]) + 6 * r * s2 - 6 * r * s3 - D * np.sin(theta[0] - theta[1]) + D * np.sin(theta[0] - theta[2]) + 2 * D * np.sin(theta[1] - theta[2])) ** 2) / (12 * (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3) ** 2) + 1) * (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3))
        

        pos[1] = (3 * r ** 3 + D ** 3 * c1 * c2 ** 2 + D ** 3 * c2 ** 2 * c3 + 6 * D * r ** 2 * c1 + 3 * D * r ** 2 * c2 + 2 * D ** 2 * r * c2 ** 2 + 5 * D ** 2 * r * c1 * c2 + 3 * D ** 2 * r * c1 * c3 - D ** 2 * r * c2 * c3 + D ** 3 * c1 * c2 * c3) / (2 * (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3)) - r / 2 - (D * c2) / 2 - (((c2 * s1 + c3 * s1 - c2 * s3 - c3 * s2) * D ** 2 - r * (s2 - 2 * s1 + s3) * D) * (((2 * D * s2 + (((c2 * s1 + c3 * s1 - c2 * s3 - c3 * s2) * D ** 2 - r * (s2 - 2 * s1 + s3) * D) * (3 * r ** 3 + D ** 3 * c1 * c2 ** 2 + D ** 3 * c2 ** 2 * c3 + 6 * D * r ** 2 * c1 + 3 * D * r ** 2 * c2 + 2 * D ** 2 * r * c2 ** 2 + 5 * D ** 2 * r * c1 * c2 + 3 * D ** 2 * r * c1 * c3 - D ** 2 * r * c2 * c3 + D ** 3 * c1 * c2 * c3)) / (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3) ** 2 - (D * (3 * D * np.sin(theta[0] + theta[1]) - 3 * D * np.sin(theta[0] + theta[2]) + 6 * r * s2 - 6 * r * s3 - D * np.sin(theta[0] - theta[1]) + D * np.sin(theta[0] - theta[2]) + 2 * D * np.sin(theta[1] - theta[2])) * (3 * r ** 3 + D ** 3 * c1 * c2 ** 2 + D ** 3 * c2 ** 2 * c3 + 2 * D * r ** 2 * c1 + 3 * D * r ** 2 * c2 + 4 * D * r ** 2 * c3 + 2 * D ** 2 * r * c2 ** 2 + D ** 2 * r * c1 * c2 + 3 * D ** 2 * r * c1 * c3 + 3 * D ** 2 * r * c2 * c3 + D ** 3 * c1 * c2 * c3)) / (2 * (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3) ** 2)) ** 2 - ((4 * ((c2 * s1 + c3 * s1 - c2 * s3 - c3 * s2) * D ** 2 - r * (s2 - 2 * s1 + s3) * D) ** 2) / (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3) ** 2 + (D ** 2 * (3 * D * np.sin(theta[0] + theta[1]) - 3 * D * np.sin(theta[0] + theta[2]) + 6 * r * s2 - 6 * r * s3 - D * np.sin(theta[0] - theta[1]) + D * np.sin(theta[0] - theta[2]) + 2 * D * np.sin(theta[1] - theta[2])) ** 2) / (3 * (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3) ** 2) + 4) * ((3 * r ** 3 + D ** 3 * c1 * c2 ** 2 + D ** 3 * c2 ** 2 * c3 + 6 * D * r ** 2 * c1 + 3 * D * r ** 2 * c2 + 2 * D ** 2 * r * c2 ** 2 + 5 * D ** 2 * r * c1 * c2 + 3 * D ** 2 * r * c1 * c3 - D ** 2 * r * c2 * c3 + D ** 3 * c1 * c2 * c3) ** 2 / (4 * (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3) ** 2) - D ** 2 * (c2 ** 2 - 1) - d ** 2 + (3 * (3 * r ** 3 + D ** 3 * c1 * c2 ** 2 + D ** 3 * c2 ** 2 * c3 + 2 * D * r ** 2 * c1 + 3 * D * r ** 2 * c2 + 4 * D * r ** 2 * c3 + 2 * D ** 2 * r * c2 ** 2 + D ** 2 * r * c1 * c2 + 3 * D ** 2 * r * c1 * c3 + 3 * D ** 2 * r * c2 * c3 + D ** 3 * c1 * c2 * c3) ** 2) / (4 * (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3) ** 2))) ** (1 / 2) / 2 + D * s2 + (((c2 * s1 + c3 * s1 - c2 * s3 - c3 * s2) * D ** 2 - r * (s2 - 2 * s1 + s3) * D) * (3 * r ** 3 + D ** 3 * c1 * c2 ** 2 + D ** 3 * c2 ** 2 * c3 + 6 * D * r ** 2 * c1 + 3 * D * r ** 2 * c2 + 2 * D ** 2 * r * c2 ** 2 + 5 * D ** 2 * r * c1 * c2 + 3 * D ** 2 * r * c1 * c3 - D ** 2 * r * c2 * c3 + D ** 3 * c1 * c2 * c3)) / (2 * (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3) ** 2) - (D * (3 * D * np.sin(theta[0] + theta[1]) - 3 * D * np.sin(theta[0] + theta[2]) + 6 * r * s2 - 6 * r * s3 - D * np.sin(theta[0] - theta[1]) + D * np.sin(theta[0] - theta[2]) + 2 * D * np.sin(theta[1] - theta[2])) * (3 * r ** 3 + D ** 3 * c1 * c2 ** 2 + D ** 3 * c2 ** 2 * c3 + 2 * D * r ** 2 * c1 + 3 * D * r ** 2 * c2 + 4 * D * r ** 2 * c3 + 2 * D ** 2 * r * c2 ** 2 + D ** 2 * r * c1 * c2 + 3 * D ** 2 * r * c1 * c3 + 3 * D ** 2 * r * c2 * c3 + D ** 3 * c1 * c2 * c3)) / (4 * (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3) ** 2))) / ((((c2 * s1 + c3 * s1 - c2 * s3 - c3 * s2) * D ** 2 - r * (s2 - 2 * s1 + s3) * D) ** 2 / (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3) ** 2 + (D ** 2 * (3 * D * np.sin(theta[0] + theta[1]) - 3 * D * np.sin(theta[0] + theta[2]) + 6 * r * s2 - 6 * r * s3 - D * np.sin(theta[0] - theta[1]) + D * np.sin(theta[0] - theta[2]) + 2 * D * np.sin(theta[1] - theta[2])) ** 2) / (12 * (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3) ** 2) + 1) * (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3))
        


        pos[2] = (((2 * D * s2 + (((c2 * s1 + c3 * s1 - c2 * s3 - c3 * s2) * D ** 2 - r * (s2 - 2 * s1 + s3) * D) * (3 * r ** 3 + D ** 3 * c1 * c2 ** 2 + D ** 3 * c2 ** 2 * c3 + 6 * D * r ** 2 * c1 + 3 * D * r ** 2 * c2 + 2 * D ** 2 * r * c2 ** 2 + 5 * D ** 2 * r * c1 * c2 + 3 * D ** 2 * r * c1 * c3 - D ** 2 * r * c2 * c3 + D ** 3 * c1 * c2 * c3)) / (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3) ** 2 - (D * (3 * D * np.sin(theta[0] + theta[1]) - 3 * D * np.sin(theta[0] + theta[2]) + 6 * r * s2 - 6 * r * s3 - D * np.sin(theta[0] - theta[1]) + D * np.sin(theta[0] - theta[2]) + 2 * D * np.sin(theta[1] - theta[2])) * (3 * r ** 3 + D ** 3 * c1 * c2 ** 2 + D ** 3 * c2 ** 2 * c3 + 2 * D * r ** 2 * c1 + 3 * D * r ** 2 * c2 + 4 * D * r ** 2 * c3 + 2 * D ** 2 * r * c2 ** 2 + D ** 2 * r * c1 * c2 + 3 * D ** 2 * r * c1 * c3 + 3 * D ** 2 * r * c2 * c3 + D ** 3 * c1 * c2 * c3)) / (2 * (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3) ** 2)) ** 2 - ((4 * ((c2 * s1 + c3 * s1 - c2 * s3 - c3 * s2) * D ** 2 - r * (s2 - 2 * s1 + s3) * D) ** 2) / (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3) ** 2 + (D ** 2 * (3 * D * np.sin(theta[0] + theta[1]) - 3 * D * np.sin(theta[0] + theta[2]) + 6 * r * s2 - 6 * r * s3 - D * np.sin(theta[0] - theta[1]) + D * np.sin(theta[0] - theta[2]) + 2 * D * np.sin(theta[1] - theta[2])) ** 2) / (3 * (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3) ** 2) + 4) * ((3 * r ** 3 + D ** 3 * c1 * c2 ** 2 + D ** 3 * c2 ** 2 * c3 + 6 * D * r ** 2 * c1 + 3 * D * r ** 2 * c2 + 2 * D ** 2 * r * c2 ** 2 + 5 * D ** 2 * r * c1 * c2 + 3 * D ** 2 * r * c1 * c3 - D ** 2 * r * c2 * c3 + D ** 3 * c1 * c2 * c3) ** 2 / (4 * (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3) ** 2) - D ** 2 * (c2 ** 2 - 1) - d ** 2 + (3 * (3 * r ** 3 + D ** 3 * c1 * c2 ** 2 + D ** 3 * c2 ** 2 * c3 + 2 * D * r ** 2 * c1 + 3 * D * r ** 2 * c2 + 4 * D * r ** 2 * c3 + 2 * D ** 2 * r * c2 ** 2 + D ** 2 * r * c1 * c2 + 3 * D ** 2 * r * c1 * c3 + 3 * D ** 2 * r * c2 * c3 + D ** 3 * c1 * c2 * c3) ** 2) / (4 * (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3) ** 2))) ** (1 / 2) / 2 + D * s2 + (((c2 * s1 + c3 * s1 - c2 * s3 - c3 * s2) * D ** 2 - r * (s2 - 2 * s1 + s3) * D) * (3 * r ** 3 + D ** 3 * c1 * c2 ** 2 + D ** 3 * c2 ** 2 * c3 + 6 * D * r ** 2 * c1 + 3 * D * r ** 2 * c2 + 2 * D ** 2 * r * c2 ** 2 + 5 * D ** 2 * r * c1 * c2 + 3 * D ** 2 * r * c1 * c3 - D ** 2 * r * c2 * c3 + D ** 3 * c1 * c2 * c3)) / (2 * (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3) ** 2) - (D * (3 * D * np.sin(theta[0] + theta[1]) - 3 * D * np.sin(theta[0] + theta[2]) + 6 * r * s2 - 6 * r * s3 - D * np.sin(theta[0] - theta[1]) + D * np.sin(theta[0] - theta[2]) + 2 * D * np.sin(theta[1] - theta[2])) * (3 * r ** 3 + D ** 3 * c1 * c2 ** 2 + D ** 3 * c2 ** 2 * c3 + 2 * D * r ** 2 * c1 + 3 * D * r ** 2 * c2 + 4 * D * r ** 2 * c3 + 2 * D ** 2 * r * c2 ** 2 + D ** 2 * r * c1 * c2 + 3 * D ** 2 * r * c1 * c3 + 3 * D ** 2 * r * c2 * c3 + D ** 3 * c1 * c2 * c3)) / (4 * (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3) ** 2)) / (((c2 * s1 + c3 * s1 - c2 * s3 - c3 * s2) * D ** 2 - r * (s2 - 2 * s1 + s3) * D) ** 2 / (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3) ** 2 + (D ** 2 * (3 * D * np.sin(theta[0] + theta[1]) - 3 * D * np.sin(theta[0] + theta[2]) + 6 * r * s2 - 6 * r * s3 - D * np.sin(theta[0] - theta[1]) + D * np.sin(theta[0] - theta[2]) + 2 * D * np.sin(theta[1] - theta[2])) ** 2) / (12 * (3 * r ** 2 + D ** 2 * c1 * c2 + D ** 2 * c1 * c3 + D ** 2 * c2 * c3 + 2 * D * r * c1 + 2 * D * r * c2 + 2 * D * r * c3) ** 2) + 1)
        pos[0] = pos[0]
        pos[2] = pos[2]
        pos[1] = pos[1]





        knee_positions = np.zeros((3, 3))
        for i in range(3):
            angle_deg = 120 * i
            R_i = np.array([[np.cos(np.deg2rad(angle_deg)), -np.sin(np.deg2rad(angle_deg)), 0],
                           [np.sin(np.deg2rad(angle_deg)), np.cos(np.deg2rad(angle_deg)), 0],
                           [0, 0, 1]])

            knee_base = np.array([0, self.r, 0])
            knee_offset = np.array([0, self.D*np.cos(theta[i]), self.D*np.sin(theta[i])])
            knee_positions[:, i] = R_i @ (knee_base + knee_offset)

        check = np.zeros(3)
        for i in range(3):
            check[i] = np.linalg.norm(pos - knee_positions[:, i]) - self.d
        
        # Align XY axes to IMU/body frame (fixed yaw about +Z). Z is unchanged.
        pos_aligned = (_R_NEW_FROM_OLD @ np.asarray(pos, dtype=float).reshape(3)).reshape(3)
        return pos_aligned, check

class InverseJacobian:
    
    def __init__(self, config=None, use_simulink_filter=True, forgetting_factor=0.95, dt=0.001):
        if config is None:
            config = DeltaLegConfig()
        self.D = float(getattr(config, "D"))
        self.d = float(getattr(config, "d"))
        self.r = float(getattr(config, "r"))
        

        self.forward_kinematics_obj = ForwardKinematics(config)
        

        self.use_simulink_filter = use_simulink_filter
        if use_simulink_filter:
            self.simulink_filter = SimulinkVelocityFilterVector(dt=dt, forgetting_factor=forgetting_factor)
        else:
            self.simulink_filter = None
    
    def forward_kinematics(self, theta):
        return self.forward_kinematics_obj.forward_kinematics(theta)
        
    def inverse_jacobian(self, x, thetadot, theta=None):
        # `x` is expected to be in the aligned kinematics frame (IMU/body-aligned XY).
        # Internally, the analytic Jacobian below is derived in the legacy (old) frame,
        # so we rotate x into the old frame before evaluating the expressions.
        x = np.asarray(x, dtype=float).reshape(3)
        x_old = (_R_OLD_FROM_NEW @ x.reshape(3)).reshape(3)

        if self.use_simulink_filter and theta is not None:
            thetadot_filtered = self.simulink_filter.update(theta)
        else:
            thetadot_filtered = thetadot
        thetadot_filtered = np.asarray(thetadot_filtered, dtype=float).reshape(3)
        # Reorder thetadot into the analytic model's motor numbering
        thetadot_model = thetadot_filtered[PERM_MODEL_FROM_PHYS]
        

        D, d, r = self.D, self.d, self.r
        

        j = np.zeros((3, 3))
        
        sqrt3 = np.sqrt(3)
        
        # NOTE: all expressions below are in the legacy (old) frame.
        x = x_old

        sqrt_arg1 = 1 - ((r - x[1])**2/2 + D**2/2 - d**2/2 + x[0]**2/2 + x[2]**2/2)**2/(D**2*((r - x[1])**2 + x[2]**2))
        sqrt_arg1 = max(sqrt_arg1, 1e-12)
        denom1 = D * np.sqrt((r - x[1])**2 + x[2]**2) * np.sqrt(sqrt_arg1)
        j[0, 0] = x[0] / denom1
        

        term1 = (r - x[1])/(D*np.sqrt((r - x[1])**2 + x[2]**2))
        term2 = ((2*r - 2*x[1])*((r - x[1])**2/2 + D**2/2 - d**2/2 + x[0]**2/2 + x[2]**2/2))/(2*D*((r - x[1])**2 + x[2]**2)**(3/2))
        sqrt_arg2 = 1 - ((r - x[1])**2/2 + D**2/2 - d**2/2 + x[0]**2/2 + x[2]**2/2)**2/(D**2*((r - x[1])**2 + x[2]**2))
        sqrt_arg2 = max(sqrt_arg2, 1e-12)
        denom2 = np.sqrt(sqrt_arg2)
        j[0, 1] = -(term1 - term2)/denom2 - x[2]/((r - x[1])**2 + x[2]**2)
        

        term3 = x[2]/(D*np.sqrt((r - x[1])**2 + x[2]**2))
        term4 = (x[2]*((r - x[1])**2/2 + D**2/2 - d**2/2 + x[0]**2/2 + x[2]**2/2))/(D*((r - x[1])**2 + x[2]**2)**(3/2))
        j[0, 2] = (term3 - term4)/denom2 - (r - x[1])/((r - x[1])**2 + x[2]**2)
        


        temp_x1 = x[0]/4 - sqrt3*x[1]/4
        temp_x2 = r + x[1]/2 + sqrt3*x[0]/2
        temp_x3 = x[0]/2 - sqrt3*x[1]/2
        temp_denom = np.sqrt((temp_x2)**2 + x[2]**2)
        temp_arg = (temp_x2**2/2 + D**2/2 - d**2/2 + temp_x3**2/2 + x[2]**2/2)
        sqrt_arg_temp = 1 - temp_arg**2/(D**2*((temp_x2)**2 + x[2]**2))
        sqrt_arg_temp = max(sqrt_arg_temp, 1e-12)
        temp_sqrt_denom = np.sqrt(sqrt_arg_temp)
        

        numerator1 = temp_x1 + sqrt3*temp_x2/2
        term1_1 = numerator1 / (D * temp_denom)
        term2_1 = (sqrt3 * temp_x2 * temp_arg) / (2 * D * ((temp_x2)**2 + x[2]**2)**(3/2))
        j[1, 0] = (term1_1 - term2_1) / temp_sqrt_denom + sqrt3*x[2] / (2*((temp_x2)**2 + x[2]**2))
        

        numerator2 = r/2 + x[1]/4 - sqrt3*temp_x3/2 + sqrt3*x[0]/4
        term1_2 = numerator2 / (D * temp_denom)
        term2_2 = (temp_x2 * temp_arg) / (2 * D * ((temp_x2)**2 + x[2]**2)**(3/2))
        j[1, 1] = x[2] / (2*((temp_x2)**2 + x[2]**2)) + (term1_2 - term2_2) / temp_sqrt_denom
        

        temp_big_expr = -2*D**2 + 2*d**2 + 2*r**2 + 2*sqrt3*r*x[0] + 2*r*x[1] + x[0]**2 + 2*sqrt3*x[0]*x[1] - x[1]**2 + 2*x[2]**2
        temp_denom_big = r*x[1] + D**2 - d**2 + r**2 + x[0]**2 + x[1]**2 + x[2]**2 + sqrt3*r*x[0]
        temp_final_denom = (2*r + x[1] + sqrt3*x[0])**2 + 4*x[2]**2
        sqrt_arg_final = 1 - temp_denom_big**2/(D**2 * temp_final_denom)
        sqrt_arg_final = max(sqrt_arg_final, 1e-12)
        temp_sqrt_final = np.sqrt(sqrt_arg_final)
        
        j[1, 2] = (2*x[2]*temp_big_expr) / (D * temp_sqrt_final * temp_final_denom**(3/2)) - (4*temp_x2) / temp_final_denom
        


        temp2_x1 = x[0]/4 + sqrt3*x[1]/4
        temp2_x2 = r + x[1]/2 - sqrt3*x[0]/2
        temp2_x3 = x[0]/2 + sqrt3*x[1]/2
        temp2_denom = np.sqrt((temp2_x2)**2 + x[2]**2)
        temp2_arg = (temp2_x2**2/2 + D**2/2 - d**2/2 + temp2_x3**2/2 + x[2]**2/2)
        sqrt_arg_temp2 = 1 - temp2_arg**2/(D**2*((temp2_x2)**2 + x[2]**2))
        sqrt_arg_temp2 = max(sqrt_arg_temp2, 1e-12)
        temp2_sqrt_denom = np.sqrt(sqrt_arg_temp2)
        

        numerator3 = temp2_x1 - sqrt3*temp2_x2/2
        term1_3 = numerator3 / (D * temp2_denom)
        term2_3 = (sqrt3 * temp2_x2 * temp2_arg) / (2 * D * ((temp2_x2)**2 + x[2]**2)**(3/2))
        j[2, 0] = (term1_3 + term2_3) / temp2_sqrt_denom - sqrt3*x[2] / (2*((temp2_x2)**2 + x[2]**2))
        

        numerator4 = r/2 + x[1]/4 + sqrt3*temp2_x3/2 - sqrt3*x[0]/4
        term1_4 = numerator4 / (D * temp2_denom)
        term2_4 = (temp2_x2 * temp2_arg) / (2 * D * ((temp2_x2)**2 + x[2]**2)**(3/2))
        j[2, 1] = x[2] / (2*((temp2_x2)**2 + x[2]**2)) + (term1_4 - term2_4) / temp2_sqrt_denom
        

        temp2_big_expr = -2*D**2 + 2*d**2 + 2*r**2 - 2*sqrt3*r*x[0] + 2*r*x[1] + x[0]**2 - 2*sqrt3*x[0]*x[1] - x[1]**2 + 2*x[2]**2
        temp2_denom_big = r*x[1] + D**2 - d**2 + r**2 + x[0]**2 + x[1]**2 + x[2]**2 - sqrt3*r*x[0]
        temp2_final_denom = (2*r + x[1] - sqrt3*x[0])**2 + 4*x[2]**2
        sqrt_arg_final2 = 1 - temp2_denom_big**2/(D**2 * temp2_final_denom)
        sqrt_arg_final2 = max(sqrt_arg_final2, 1e-12)
        temp2_sqrt_final = np.sqrt(sqrt_arg_final2)
        
        j[2, 2] = (2*x[2]*temp2_big_expr) / (D * temp2_sqrt_final * temp2_final_denom**(3/2)) - (4*temp2_x2) / temp2_final_denom
        

        try:
            xdot_old = np.linalg.solve(j, thetadot_model[:3])
        except np.linalg.LinAlgError:
            xdot_old = np.linalg.lstsq(j, thetadot_model[:3], rcond=None)[0]
        
        # Convert Jacobian and velocity back to the aligned (new) frame.
        # thetadot = J_inv_old * xdot_old
        # xdot_old = R_old_from_new * xdot_new  =>  thetadot = (J_inv_old * R_old_from_new) * xdot_new
        J_inv_new_model = j @ _R_OLD_FROM_NEW
        xdot_new = (_R_NEW_FROM_OLD @ np.asarray(xdot_old, dtype=float).reshape(3)).reshape(3)
        # Return J_inv in PHYSICAL motor order: thetadot_phys = J_inv_phys * xdot_new
        J_inv_new_phys = J_inv_new_model[PERM_PHYS_FROM_MODEL, :]
        return J_inv_new_phys, xdot_new
