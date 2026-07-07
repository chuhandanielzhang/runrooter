#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import sys
import time

import lcm

# Make LCM python types importable (same pattern as modee/lcm_controller.py)
_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
_LCM_TYPES_DIR = os.path.join(_CUR_DIR, "..", "hopper_lcm_types", "lcm_types")
sys.path.append(_LCM_TYPES_DIR)

from python.gamepad_lcmt import gamepad_lcmt  # type: ignore
from python.motor_pwm_lcmt import motor_pwm_lcmt  # type: ignore


class RightStickPwm1Passthrough:
    def __init__(self, *, lc: lcm.LCM, pwm_on: float, deadband: float):
        self.lc = lc
        self.pwm_on = float(pwm_on)
        self.deadband = float(deadband)
        self._last_active: bool | None = None
        self.lc.subscribe("gamepad_lcmt", self._on_gamepad)

    def _on_gamepad(self, channel: str, data: bytes) -> None:
        try:
            msg = gamepad_lcmt.decode(data)
        except Exception:
            return

        try:
            rx = float(msg.rightStickAnalog[0])
            ry = float(msg.rightStickAnalog[1])
        except Exception:
            rx = 0.0
            ry = 0.0

        active = (abs(rx) > float(self.deadband)) or (abs(ry) > float(self.deadband))

        pwm = [0.0] * 6
        if bool(active):
            pwm[1] = float(self.pwm_on)

        out = motor_pwm_lcmt()
        out.timestamp = int(time.time() * 1e6)
        out.pwm_values = [float(v) for v in pwm]
        out.roll_error = 0.0
        out.pitch_error = 0.0
        out.roll_output = 0.0
        out.pitch_output = 0.0
        out.control_mode = 1
        self.lc.publish("motor_pwm_lcmt", out.encode())

        if self._last_active is None or bool(active) != bool(self._last_active):
            print(f"[pwm_test] rightStick=({rx:+.3f},{ry:+.3f}) active={int(active)} -> pwm1={pwm[1]:.1f}")
            self._last_active = bool(active)


def main() -> None:
    ap = argparse.ArgumentParser(description="Right stick -> pwm[1]=1100, all other PWM=0 (upper-level test)")
    ap.add_argument(
        "--lcm-url",
        type=str,
        default="udpm://239.255.76.67:7667?ttl=255",
        help='LCM URL (default: "udpm://239.255.76.67:7667?ttl=255")',
    )
    ap.add_argument("--pwm-on", type=float, default=1100.0, help="PWM value written to pwm[1] when stick is nonzero")
    ap.add_argument(
        "--deadband",
        type=float,
        default=0.0,
        help="Right stick deadband. 0.0 means any nonzero value triggers.",
    )
    args = ap.parse_args()

    lc = lcm.LCM(str(args.lcm_url))
    RightStickPwm1Passthrough(lc=lc, pwm_on=float(args.pwm_on), deadband=float(args.deadband))

    while True:
        lc.handle_timeout(100)


if __name__ == "__main__":
    main()







