"""
PWM <-> thrust motor table.

SNAPSHOT COPY of `hopperHFAcase2026/hopper_controller/modee/controllers/motor_utils.py`
(MotorTableModel only). Copied so this demo folder is fully isolated from the
Jetson/PC runtime tree. Do NOT edit the original; edit here freely.
"""

from __future__ import annotations

import numpy as np


class MotorTableModel:
    """PWM-based motor model using a measured thrust table (1950KV characterization)."""

    def __init__(
        self,
        pwm_us_bp: np.ndarray,
        thrust_n_bp: np.ndarray,
        *,
        pwm_min_us: float = 1000.0,
        pwm_max_us: float = 2000.0,
    ):
        self.pwm_us_bp = np.asarray(pwm_us_bp, dtype=float).reshape(-1)
        self.thrust_n_bp = np.asarray(thrust_n_bp, dtype=float).reshape(-1)
        if self.pwm_us_bp.shape != self.thrust_n_bp.shape:
            raise ValueError("pwm_us_bp and thrust_n_bp must have the same shape")
        self.pwm_min_us = float(pwm_min_us)
        self.pwm_max_us = float(pwm_max_us)

    @staticmethod
    def default_from_table() -> "MotorTableModel":
        throttle_pct = np.array([0.0, 20.0, 40.0, 60.0, 80.0, 100.0], dtype=float)
        pwm_us = 1000.0 + (throttle_pct / 100.0) * 1000.0
        thrust_g = np.array([0.0, 269.8, 663.2, 1060.8, 1610.7, 2032.9], dtype=float)
        thrust_n = (thrust_g / 1000.0) * 9.81
        return MotorTableModel(pwm_us, thrust_n)

    def clamp_pwm(self, pwm_us: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(pwm_us, dtype=float), self.pwm_min_us, self.pwm_max_us)

    def thrust_from_pwm(self, pwm_us: np.ndarray) -> np.ndarray:
        p = self.clamp_pwm(pwm_us)
        return np.interp(p, self.pwm_us_bp, self.thrust_n_bp)

    def pwm_from_thrust(self, thrust_n: np.ndarray) -> np.ndarray:
        t = np.asarray(thrust_n, dtype=float)
        t_clamped = np.clip(t, float(np.min(self.thrust_n_bp)), float(np.max(self.thrust_n_bp)))
        pwm = np.interp(t_clamped, self.thrust_n_bp, self.pwm_us_bp)
        return self.clamp_pwm(pwm)
