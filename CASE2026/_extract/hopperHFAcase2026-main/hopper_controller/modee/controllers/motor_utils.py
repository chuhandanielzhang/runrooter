import numpy as np


class MotorModel:
    """Simple motor model: thrust = Ct * omega^2, reaction torque = Cd * omega^2."""

    def __init__(self, ct: float, cd: float, max_speed: float):
        self.ct = float(ct)
        self.cd = float(cd)
        self.max_speed = float(max_speed)  # krpm or rad/s depending on upstream convention

    def clamp_speed(self, motor_speeds: np.ndarray) -> np.ndarray:
        s = np.asarray(motor_speeds, dtype=float)
        return np.clip(s, 0.0, self.max_speed)

    def thrusts_from_speeds(self, motor_speeds: np.ndarray) -> np.ndarray:
        s = np.asarray(motor_speeds, dtype=float)
        return self.ct * s * s

    def torques_from_speeds(self, motor_speeds: np.ndarray) -> np.ndarray:
        s = np.asarray(motor_speeds, dtype=float)
        return self.cd * s * s


class MotorTableModel:
    """
    PWM-based motor model using a measured thrust table.

    This is intended for simulation realism:
    - Controller / QP can output desired per-motor thrust (N)
    - We convert thrust -> PWM (1000-2000us) using the inverse table (ESC-like)
    - Then convert PWM -> actual thrust using the forward table
    - Reaction torque is also generated (if no torque table is provided, we use a constant ratio)
    """

    def __init__(
        self,
        pwm_us_bp: np.ndarray,
        thrust_n_bp: np.ndarray,
        *,
        rpm_bp: np.ndarray | None = None,
        power_w_bp: np.ndarray | None = None,
        torque_nm_bp: np.ndarray | None = None,
        tau_per_thrust: float | None = None,
        pwm_min_us: float = 1000.0,
        pwm_max_us: float = 2000.0,
    ):
        self.pwm_us_bp = np.asarray(pwm_us_bp, dtype=float).reshape(-1)
        self.thrust_n_bp = np.asarray(thrust_n_bp, dtype=float).reshape(-1)
        if self.pwm_us_bp.shape != self.thrust_n_bp.shape:
            raise ValueError("pwm_us_bp and thrust_n_bp must have the same shape")

        self.pwm_min_us = float(pwm_min_us)
        self.pwm_max_us = float(pwm_max_us)

        self.rpm_bp = None if rpm_bp is None else np.asarray(rpm_bp, dtype=float).reshape(-1)
        if (self.rpm_bp is not None) and (self.rpm_bp.shape != self.pwm_us_bp.shape):
            raise ValueError("rpm_bp must have the same shape as pwm_us_bp")
        self.power_w_bp = None if power_w_bp is None else np.asarray(power_w_bp, dtype=float).reshape(-1)
        if (self.power_w_bp is not None) and (self.power_w_bp.shape != self.pwm_us_bp.shape):
            raise ValueError("power_w_bp must have the same shape as pwm_us_bp")

        self.torque_nm_bp = None if torque_nm_bp is None else np.asarray(torque_nm_bp, dtype=float).reshape(-1)
        if (self.torque_nm_bp is not None) and (self.torque_nm_bp.shape != self.pwm_us_bp.shape):
            raise ValueError("torque_nm_bp must have the same shape as pwm_us_bp")

        # If no torque table, approximate: tau_z = tau_per_thrust * thrust
        self.tau_per_thrust = None if tau_per_thrust is None else float(tau_per_thrust)

    @staticmethod
    def default_from_table(*, tau_per_thrust: float | None = None) -> "MotorTableModel":
        """
        Default table (1950KV) from your motor characterization screenshot:

        Columns: throttle%, voltage(V), current(A), rpm, thrust(g), power(W), efficiency(g/W)

        We interpret it as **per-motor** data and build:
        - PWM(us) from throttle% using a linear ESC map (1000-2000us)
        - Thrust(N) from thrust(g)
        - Reaction torque magnitude (Nm) from power/rpm:
            tau ≈ P / omega,  omega = 2*pi*rpm/60

        Note: power in the table is electrical; tau computed this way is an approximation but captures the scale.
        """
        throttle_pct = np.array([0.0, 20.0, 40.0, 60.0, 80.0, 100.0], dtype=float)
        pwm_us = 1000.0 + (throttle_pct / 100.0) * 1000.0
        thrust_g = np.array([0.0, 269.8, 663.2, 1060.8, 1610.7, 2032.9], dtype=float)
        thrust_n = (thrust_g / 1000.0) * 9.81
        rpm = np.array([0.0, 12081.1, 18434.1, 22987.9, 28369.3, 32523.4], dtype=float)
        power_w = np.array([0.0, 70.9, 236.1, 475.7, 845.9, 1353.9], dtype=float)
        omega = rpm * (2.0 * np.pi / 60.0)
        torque_nm = np.zeros_like(omega)
        mask = omega > 1e-9
        torque_nm[mask] = power_w[mask] / omega[mask]
        # Scale down torque by 0.01 (user requirement: 电功率->机械扭矩的转换系数)
        torque_nm = torque_nm * 0.01
        return MotorTableModel(
            pwm_us,
            thrust_n,
            rpm_bp=rpm,
            power_w_bp=power_w,
            torque_nm_bp=torque_nm,
            tau_per_thrust=tau_per_thrust,
        )

    def clamp_pwm(self, pwm_us: np.ndarray) -> np.ndarray:
        p = np.asarray(pwm_us, dtype=float)
        return np.clip(p, self.pwm_min_us, self.pwm_max_us)

    def thrust_from_pwm(self, pwm_us: np.ndarray) -> np.ndarray:
        p = self.clamp_pwm(pwm_us)
        return np.interp(p, self.pwm_us_bp, self.thrust_n_bp)

    def rpm_from_pwm(self, pwm_us: np.ndarray) -> np.ndarray:
        p = self.clamp_pwm(pwm_us)
        if self.rpm_bp is None:
            return np.zeros_like(p, dtype=float)
        return np.interp(p, self.pwm_us_bp, self.rpm_bp)

    def torque_from_pwm(self, pwm_us: np.ndarray) -> np.ndarray:
        p = self.clamp_pwm(pwm_us)
        if self.torque_nm_bp is not None:
            return np.interp(p, self.pwm_us_bp, self.torque_nm_bp)
        if self.tau_per_thrust is not None:
            return self.tau_per_thrust * self.thrust_from_pwm(p)
        # default: no torque info
        return np.zeros_like(p, dtype=float)

    def pwm_from_thrust(self, thrust_n: np.ndarray) -> np.ndarray:
        t = np.asarray(thrust_n, dtype=float)
        # invert using monotonic interp on the provided table
        t_clamped = np.clip(t, float(np.min(self.thrust_n_bp)), float(np.max(self.thrust_n_bp)))
        pwm = np.interp(t_clamped, self.thrust_n_bp, self.pwm_us_bp)
        return self.clamp_pwm(pwm)


