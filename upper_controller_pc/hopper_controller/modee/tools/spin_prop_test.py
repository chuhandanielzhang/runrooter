#!/usr/bin/env python3
"""Sequentially spin prop motors via motor_pwm_lcmt (bring-up / wiring check).

Each motor runs at a fixed PWM (default 1100 us) for a fixed duration (default 2 s),
one at a time. Other channels stay at pwm_min (idle).

Requires px4_bridge (or equivalent) subscribed on the same LCM network with
control_mode == prop_arm_mode (default 3).

Usage:
  cd upper_controller_pc/hopper_controller
  python3 modee/tools/spin_prop_test.py
  python3 modee/tools/spin_prop_test.py --pwm 1100 --duration 2 --motors 1,2,3
"""
from __future__ import annotations

import argparse
import os
import sys
import time

import lcm

HERE = os.path.dirname(os.path.abspath(__file__))
CTRL = os.path.abspath(os.path.join(HERE, "..", ".."))
_LCM_TYPES_DIR = os.path.join(CTRL, "..", "hopper_lcm_types", "lcm_types")
for p in (CTRL, _LCM_TYPES_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from python.motor_pwm_lcmt import motor_pwm_lcmt  # noqa: E402

# pwm_values index -> 120° Y arm (body FRD: +X fwd, +Y right).
# 2026-07-18 confirmed physical mapping (M1 moved MAIN1 -> MAIN8):
# M1->MAIN8 at +30deg, M2->MAIN2 at -90deg, M3->MAIN3 at +150deg.
# Bridge map: pwm[1]->set1, pwm[2]->set2, pwm[3]->set3 (sets, not pins).
# --motors here still takes the PWM INDICES 1,2,3 (unchanged).
MOTOR_LABELS = {
    1: "M1 pwm[1]->MAIN8 @ +30deg (+x,+y), bridge direction inverted",
    2: "M2 pwm[2]->MAIN2 @ -90deg (0,-L), bridge direction inverted",
    3: "M3 pwm[3]->MAIN3 @ +150deg (-x,+y)",
}


def _publish(lc: lcm.LCM, pwm_us: list[float], control_mode: int) -> None:
    msg = motor_pwm_lcmt()
    msg.timestamp = int(time.time() * 1e6)
    msg.pwm_values = [float(v) for v in pwm_us]
    msg.roll_error = 0.0
    msg.pitch_error = 0.0
    msg.roll_output = 0.0
    msg.pitch_output = 0.0
    msg.control_mode = int(control_mode)
    lc.publish("motor_pwm_lcmt", msg.encode())


def _stream(lc: lcm.LCM, pwm_us: list[float], control_mode: int, duration_s: float, hz: float) -> None:
    dt = 1.0 / max(1.0, float(hz))
    end = time.time() + float(duration_s)
    while time.time() < end:
        _publish(lc, pwm_us, control_mode)
        time.sleep(dt)
    _publish(lc, pwm_us, control_mode)


def main() -> None:
    ap = argparse.ArgumentParser(description="Sequential prop motor spin test (motor_pwm_lcmt)")
    ap.add_argument("--lcm-url", default="udpm://239.255.76.67:7667?ttl=1")
    ap.add_argument("--pwm", type=float, default=1100.0, help="PWM pulse width (us) while spinning")
    ap.add_argument("--pwm-min", type=float, default=1000.0, help="Idle PWM (us) for off channels")
    ap.add_argument("--duration", type=float, default=2.0, help="Spin time per motor (s)")
    ap.add_argument("--motors", default="1,2,3",
                    help="pwm_values indices to test, comma-separated (default: 1,2,3)")
    ap.add_argument("--hz", type=float, default=50.0, help="Publish rate (Hz)")
    ap.add_argument("--control-mode", type=int, default=3,
                    help="motor_pwm_lcmt.control_mode when props armed (px4 default: 3)")
    ap.add_argument("--pause", type=float, default=0.5,
                    help="Idle pause between motors (s)")
    ap.add_argument("--together", action="store_true",
                    help="spin ALL listed motors at the same time (default: one by one)")
    ap.add_argument("--wire", action="store_true",
                    help="interpret --pwm in WIRE scale (1500=stop, 1000=full rev, "
                         "2000=full fwd) instead of the internal LCM scale (1000=stop)")
    args = ap.parse_args()

    if args.wire:
        # wire 1500=stop, +/-500 span  ->  LCM 1000=stop, +/-1000 span
        args.pwm = 1000.0 + (float(args.pwm) - 1500.0) * 2.0

    motor_idxs = [int(x.strip()) for x in str(args.motors).split(",") if x.strip()]
    for idx in motor_idxs:
        if not (0 <= idx <= 5):
            raise SystemExit(f"invalid motor index {idx}; pwm_values is length 6 (0..5)")

    # A spin shorter than a few bridge periods (bridge re-streams at ~250 Hz and
    # the ESC ramps over ~100 ms) does nothing visible; catch typos like 0.001.
    if float(args.duration) < 0.2:
        raise SystemExit(
            f"--duration {args.duration} s is too short to observe (ESC ramp ~0.1 s); "
            f"use >= 0.2, e.g. --duration 1")

    lc = lcm.LCM(str(args.lcm_url))
    idle = [float(args.pwm_min)] * 6
    spin_pwm = float(args.pwm)
    cm = int(args.control_mode)

    def wire_us(pwm_lcm: float) -> float:
        """LCM scale (1000=stop) -> actual AUX wire PWM (1500=stop, 3D ESC)."""
        act = max(-1.0, min(1.0, (pwm_lcm - 1000.0) / 1000.0))
        return 1500.0 + act * 500.0

    def pct(pwm_lcm: float) -> str:
        act = max(-1.0, min(1.0, (pwm_lcm - 1000.0) / 1000.0))
        if act == 0.0:
            return "stop"
        return f"{abs(act) * 100.0:.0f}% {'FWD' if act > 0 else 'REV'}"

    print(f"LCM {args.lcm_url}")
    print(f"Motors {motor_idxs}: cmd {spin_pwm:.0f} (LCM scale, 1000=stop) "
          f"= wire {wire_us(spin_pwm):.0f} us (1500=stop) = {pct(spin_pwm)}")
    print(f"Duration {args.duration:.1f} s each, control_mode={cm}, "
          f"idle cmd {args.pwm_min:.0f} = wire {wire_us(args.pwm_min):.0f} us")
    print("Ctrl+C to abort.\n")

    # Hold idle briefly so bridge sees a live stream before arming.
    print("[idle] streaming pwm_min ...")
    _stream(lc, idle, cm, 0.5, args.hz)

    try:
        if args.together:
            pwm = list(idle)
            for idx in motor_idxs:
                pwm[idx] = spin_pwm
            print(f"[ALL {motor_idxs}] cmd {spin_pwm:.0f} (wire {wire_us(spin_pwm):.0f} us, "
                  f"{pct(spin_pwm)}) for {args.duration:.1f} s (together)")
            _stream(lc, pwm, cm, float(args.duration), args.hz)
            print(f"[ALL {motor_idxs}] done -> idle")
            _stream(lc, idle, cm, float(args.pause), args.hz)
        else:
            for idx in motor_idxs:
                pwm = list(idle)
                pwm[idx] = spin_pwm
                label = MOTOR_LABELS.get(idx, f"pwm[{idx}]")
                print(f"[{label}] cmd {spin_pwm:.0f} (wire {wire_us(spin_pwm):.0f} us, "
                      f"{pct(spin_pwm)}) for {args.duration:.1f} s")
                _stream(lc, pwm, cm, float(args.duration), args.hz)
                print(f"[{label}] done -> idle")
                _stream(lc, idle, cm, float(args.pause), args.hz)
    except KeyboardInterrupt:
        print("\n[abort]")

    # Disarm: idle PWM + control_mode off.
    print("[disarm] pwm_min, control_mode=1")
    for _ in range(10):
        _publish(lc, idle, 1)
        time.sleep(0.02)
    print("Finished.")


if __name__ == "__main__":
    main()
