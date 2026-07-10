#!/usr/bin/env python3
"""Bench system-ID for the tri-rotor (thrust calibration / motor lag / attitude FRF).

Publishes motor_pwm_lcmt (same path as spin_prop_test.py, px4_bridge must run)
and logs hopper_imu_lcmt (gyro/acc/rpy) + the commanded PWM to a CSV at full
IMU rate. Three experiment modes:

  ladder  Collective PWM staircase (all 3 arms same PWM). Robot sits on a
          SCALE. Each level is held hold_s seconds and announced on stdout ->
          write down the scale reading (and battery voltage) per level.
          Gives the per-robot thrust map  T_total(pwm, V).
  step    Differential torque steps (one arm up, opposite pair down around a
          baseline) while the robot hangs in the GIMBAL. The gyro response
          rise gives the motor time constant tau_m; the initial angular
          acceleration gives control effectiveness (-> J with the thrust map).
  chirp   Differential sine sweep f0->f1 Hz around the baseline (GIMBAL).
          Frequency response gyro/pwm -> bandwidth, phase lag, J cross-check.

SAFETY: requires --yes-spin to actually spin. Always ends with an idle ramp +
control_mode=1 disarm (also on Ctrl+C). Secure the robot before running.

Usage (from upper_controller_pc/hopper_controller):
  python3 modee/tools/bench_prop_id.py ladder --pwm-max 1500 --yes-spin
  python3 modee/tools/bench_prop_id.py step  --base 1150 --delta 150 --yes-spin
  python3 modee/tools/bench_prop_id.py chirp --base 1150 --delta 80 \
      --f0 0.5 --f1 12 --sweep-s 40 --yes-spin
"""
from __future__ import annotations

import argparse
import csv
import math
import os
import sys
import threading
import time

import lcm

HERE = os.path.dirname(os.path.abspath(__file__))
CTRL = os.path.abspath(os.path.join(HERE, "..", ".."))
_LCM_TYPES_DIR = os.path.join(CTRL, "..", "hopper_lcm_types", "lcm_types")
for p in (CTRL, _LCM_TYPES_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from python.motor_pwm_lcmt import motor_pwm_lcmt  # noqa: E402
from python.hopper_imu_lcmt import hopper_imu_lcmt  # noqa: E402

# Physical arm mapping (2026-07-06, body FRD): pwm[1]=M1 @ +30deg (+x,+y),
# pwm[2]=M2 @ +150deg (-x,+y), pwm[3]=M3 @ -90deg (0,-L).
ARM_IDXS = (1, 2, 3)
PWM_STOP = 1000.0


class ImuLogger:
    """Subscribes hopper_imu_lcmt and writes rows tagged with the live PWM cmd."""

    def __init__(self, lc: lcm.LCM, csv_path: str):
        self.lc = lc
        self._fp = open(csv_path, "w", newline="")
        self._w = csv.writer(self._fp)
        self._w.writerow(
            ["t_s", "pwm1", "pwm2", "pwm3", "phase",
             "gx", "gy", "gz", "ax", "ay", "az",
             "roll", "pitch", "yaw"])
        self._lock = threading.Lock()
        self._pwm = [PWM_STOP, PWM_STOP, PWM_STOP]
        self._phase = "idle"
        self._t0 = time.time()
        self.n_rows = 0
        lc.subscribe("hopper_imu_lcmt", self._on_imu)

    def set_cmd(self, pwm123, phase: str) -> None:
        with self._lock:
            self._pwm = [float(v) for v in pwm123]
            self._phase = str(phase)

    def _on_imu(self, _ch, data) -> None:
        m = hopper_imu_lcmt.decode(data)
        with self._lock:
            p1, p2, p3 = self._pwm
            ph = self._phase
        self._w.writerow(
            [f"{time.time() - self._t0:.6f}", f"{p1:.1f}", f"{p2:.1f}", f"{p3:.1f}", ph,
             *(f"{float(v):.6f}" for v in m.gyro),
             *(f"{float(v):.6f}" for v in m.acc),
             *(f"{float(v):.6f}" for v in m.rpy)])
        self.n_rows += 1

    def close(self) -> None:
        self._fp.flush()
        self._fp.close()


def _publish(lc: lcm.LCM, pwm6, control_mode: int) -> None:
    msg = motor_pwm_lcmt()
    msg.timestamp = int(time.time() * 1e6)
    msg.pwm_values = [float(v) for v in pwm6]
    msg.roll_error = 0.0
    msg.pitch_error = 0.0
    msg.roll_output = 0.0
    msg.pitch_output = 0.0
    msg.control_mode = int(control_mode)
    lc.publish("motor_pwm_lcmt", msg.encode())


class Runner:
    def __init__(self, args, log: ImuLogger, lc: lcm.LCM):
        self.args = args
        self.log = log
        self.lc = lc
        self._stop = threading.Event()
        # LCM handle thread (drains IMU into the CSV).
        self._rx = threading.Thread(target=self._rx_loop, daemon=True)
        self._rx.start()

    def _rx_loop(self) -> None:
        while not self._stop.is_set():
            self.lc.handle_timeout(100)

    def stop(self) -> None:
        self._stop.set()

    def send(self, p1: float, p2: float, p3: float, phase: str) -> None:
        a = self.args
        cap = float(a.pwm_max)
        floor = float(a.pwm_floor)
        p = [min(cap, max(floor, float(v))) for v in (p1, p2, p3)]
        pwm6 = [PWM_STOP] * 6
        for k, idx in enumerate(ARM_IDXS):
            pwm6[idx] = p[k]
        self.log.set_cmd(p, phase)
        _publish(self.lc, pwm6, int(a.control_mode))

    def hold(self, p1: float, p2: float, p3: float, phase: str, dur_s: float) -> None:
        dt = 1.0 / float(self.args.hz)
        end = time.time() + float(dur_s)
        while time.time() < end:
            self.send(p1, p2, p3, phase)
            time.sleep(dt)

    def ramp_down(self, from_pwm: float, dur_s: float = 1.5) -> None:
        dt = 1.0 / float(self.args.hz)
        n = max(1, int(dur_s / dt))
        for i in range(n):
            v = from_pwm + (PWM_STOP - from_pwm) * (i + 1) / n
            self.send(v, v, v, "rampdown")
            time.sleep(dt)

    def disarm(self) -> None:
        for _ in range(10):
            _publish(self.lc, [PWM_STOP] * 6, 1)
            time.sleep(0.02)


def run_ladder(r: Runner, a) -> None:
    levels = list(range(int(a.ladder_start), int(a.pwm_max) + 1, int(a.ladder_step)))
    print(f"LADDER: {levels} us, hold {a.hold_s}s each. Robot on the SCALE.")
    print("Write down per level: scale reading [g or N] + battery voltage [V].")
    r.hold(PWM_STOP, PWM_STOP, PWM_STOP, "idle", 1.0)
    for lv in levels:
        print(f"\n>>> LEVEL {lv} us  (read the scale + voltage now)")
        r.hold(lv, lv, lv, f"ladder{lv}", float(a.hold_s))
    r.ramp_down(levels[-1])


def run_step(r: Runner, a) -> None:
    base = float(a.base)
    d = float(a.delta)
    print(f"STEP: base {base} us, differential +/-{d} us, {a.reps} doublets. GIMBAL.")
    r.hold(PWM_STOP, PWM_STOP, PWM_STOP, "idle", 1.0)
    print(">>> spool to baseline")
    r.hold(base, base, base, "base", 3.0)
    # Pitch-ish doublet: M3 (rear/-Y? actually -90deg arm) vs M1+M2 pair.
    for k in range(int(a.reps)):
        print(f">>> doublet {k + 1}/{a.reps}")
        r.hold(base - d / 2, base - d / 2, base + d, f"step_up{k}", float(a.step_s))
        r.hold(base + d / 2, base + d / 2, base - d, f"step_dn{k}", float(a.step_s))
        r.hold(base, base, base, f"base{k}", float(a.step_s))
    r.ramp_down(base)


def run_chirp(r: Runner, a) -> None:
    base = float(a.base)
    d = float(a.delta)
    f0, f1, T = float(a.f0), float(a.f1), float(a.sweep_s)
    print(f"CHIRP: base {base} us, +/-{d} us, {f0}->{f1} Hz over {T}s. GIMBAL.")
    r.hold(PWM_STOP, PWM_STOP, PWM_STOP, "idle", 1.0)
    print(">>> spool to baseline")
    r.hold(base, base, base, "base", 3.0)
    dt = 1.0 / float(a.hz)
    t0 = time.time()
    while True:
        t = time.time() - t0
        if t >= T:
            break
        # Exponential (log) chirp: equal time per octave.
        kr = (f1 / f0) ** (1.0 / T)
        phi = 2.0 * math.pi * f0 * ((kr ** t - 1.0) / math.log(kr))
        s = d * math.sin(phi)
        r.send(base - s / 2, base - s / 2, base + s, "chirp")
        time.sleep(dt)
    r.hold(base, base, base, "base_end", 2.0)
    r.ramp_down(base)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["ladder", "step", "chirp"])
    ap.add_argument("--lcm-url", default="udpm://239.255.76.67:7667?ttl=1")
    ap.add_argument("--hz", type=float, default=100.0, help="PWM publish rate")
    ap.add_argument("--control-mode", type=int, default=3)
    ap.add_argument("--pwm-max", type=float, default=1500.0, help="hard PWM cap (us)")
    ap.add_argument("--pwm-floor", type=float, default=1000.0,
                    help="hard PWM floor (us); lower only if you want reverse")
    ap.add_argument("--log", default=None, help="CSV path (default logs/bench_<mode>_<ts>.csv)")
    ap.add_argument("--yes-spin", action="store_true", help="REQUIRED to actually spin props")
    # ladder
    ap.add_argument("--ladder-start", type=float, default=1050.0)
    ap.add_argument("--ladder-step", type=float, default=50.0)
    ap.add_argument("--hold-s", type=float, default=5.0)
    # step
    ap.add_argument("--base", type=float, default=1150.0)
    ap.add_argument("--delta", type=float, default=150.0)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--step-s", type=float, default=1.5)
    # chirp
    ap.add_argument("--f0", type=float, default=0.5)
    ap.add_argument("--f1", type=float, default=12.0)
    ap.add_argument("--sweep-s", type=float, default=40.0)
    a = ap.parse_args()

    if not a.yes_spin:
        raise SystemExit("Refusing to spin: add --yes-spin after securing the robot.")

    log_path = a.log
    if log_path is None:
        os.makedirs(os.path.join(CTRL, "logs"), exist_ok=True)
        log_path = os.path.join(CTRL, "logs", f"bench_{a.mode}_{time.strftime('%m%d_%H%M%S')}.csv")

    lc = lcm.LCM(str(a.lcm_url))
    log = ImuLogger(lc, log_path)
    r = Runner(a, log, lc)
    print(f"LCM {a.lcm_url}  ->  log {log_path}")
    try:
        {"ladder": run_ladder, "step": run_step, "chirp": run_chirp}[a.mode](r, a)
    except KeyboardInterrupt:
        print("\n[abort] ramping down")
        r.ramp_down(float(a.base if a.mode != "ladder" else a.pwm_max), 1.0)
    finally:
        r.disarm()
        r.stop()
        log.close()
        print(f"done. IMU rows logged: {log.n_rows}  ->  {log_path}")


if __name__ == "__main__":
    main()
