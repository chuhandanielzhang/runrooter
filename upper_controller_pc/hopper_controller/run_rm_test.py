#!/usr/bin/env python3
"""Standalone RM (M2006) trigger test — NO legs, NO propellers.

Control law: current-mode PD (a PID without the integral term):
    iq = kp * (q_target - rm_q) - kd * rm_qd,  clipped to +/-iq_max.

Gamepad:
  LT  -> PRE-ZERO at the current position (rm_set_zero pulse; the JETSON DRIVER
         latches it, so the drive always starts from a fresh rm_q = 0), then
         drive 0 -> -11 rad; on arrival hold --hold s, then RE-ZERO again at
         the target. Motors then go idle (0 A). NO drive back.
  RT  -> IDENTICAL sequence, target +11 rad.
  B   -> abort at any time: stage reset, 0 A. (The Jetson driver also cuts the
         current in DAMP mode by itself.)

Every run writes a session log to logs/rm_test_YYYYMMDD_HHMMSS.csv:
  - '#' header lines: start time + all parameters;
  - 100 Hz data rows: t, stage, q, qd, iq_des, iq_fb, online, zero offset;
  - event rows (LT/RT/B presses, stage changes, re-zero, check results) carry
    the message in the last 'event' column. All events also print to stdout.

The Jetson driver only FORWARDS rm_iq_des in PD/PWMPD mode, so the gamepad X
button still arms, exactly like the full controller.

Usage (on the PC, robot driver + DDS bridge already running on the Jetson):

    cd upper_controller_pc/hopper_controller
    python3 run_rm_test.py                     # defaults: +/-11 rad, cap 3 A
    python3 run_rm_test.py --iq-max 1.0        # gentler first test
    python3 run_rm_test.py --range 5.0         # shorter +/-5 rad motion

Then on the gamepad:  X (arm) -> LT / RT -> B (stop).
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from datetime import datetime

import numpy as np
import lcm

_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_CUR_DIR, "..", "hopper_lcm_types", "lcm_types"))

from python.hopper_data_lcmt import hopper_data_lcmt  # type: ignore
from python.hopper_cmd_lcmt import hopper_cmd_lcmt  # type: ignore
from python.gamepad_lcmt import gamepad_lcmt  # type: ignore

# Stages (LT and RT are the SAME sequence, only the target sign differs:
# PRE-ZERO here -> drive to +/-range -> hold -> re-zero there -> idle)
IDLE, PREZERO, DRIVE, HOLD = 0, 1, 2, 3
STAGE_NAMES = {IDLE: "IDLE", PREZERO: "PRE0", DRIVE: "DRIVE", HOLD: "HOLD"}


class RmTriggerTest:
    def __init__(self, args) -> None:
        self.args = args
        self.lc = lcm.LCM("udpm://239.255.76.67:7667?ttl=255")
        self.rm_q_raw = np.zeros(3)   # as reported by the driver
        self.rm_qd = np.zeros(3)
        self.rm_iq = np.zeros(3)
        self.rm_online = 0
        self.have_data = False

        self.stage = IDLE
        self.stage_t0 = 0.0
        self.target = 0.0            # set on LT (-range) / RT (+range)
        self.iq_des = np.zeros(3)
        # rm_set_zero pulse: publish nonzero until this deadline (driver latches
        # the new zero on the 0->1 edge; a short pulse survives packet loss).
        self.set_zero_until = 0.0
        self._last_lt = False
        self._last_rt = False
        self._last_b = False

        # ---- session log (one file per program start) ----
        log_dir = os.path.join(_CUR_DIR, "logs")
        os.makedirs(log_dir, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_path = os.path.join(log_dir, "rm_test_%s.csv" % stamp)
        self._log_fp = open(self.log_path, "w", newline="")
        self._log = csv.writer(self._log_fp)
        self._log_fp.write("# rm_test session start %s\n"
                           % datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
        self._log_fp.write("# params: %s\n" % vars(args))
        self._log.writerow(["t", "stage",
                            "q0", "q1", "q2", "qd0", "qd1", "qd2",
                            "iq_des0", "iq_des1", "iq_des2",
                            "iq_fb0", "iq_fb1", "iq_fb2",
                            "online", "set_zero", "event"])
        self._t0 = time.time()
        self._last_row_t = 0.0

        self.lc.subscribe("hopper_data_lcmt", self._on_data)
        self.lc.subscribe("gamepad_lcmt", self._on_gamepad)
        self.event("session start, log: %s" % self.log_path)

    # -------- helpers --------
    @property
    def q(self) -> np.ndarray:
        # rm_q as reported by the driver (the driver owns the zero offset).
        return self.rm_q_raw

    def event(self, msg: str) -> None:
        """Print AND write an event row into the CSV."""
        print("[rm] " + msg)
        self._write_row(event=msg)
        self._log_fp.flush()

    def _write_row(self, event: str = "") -> None:
        q = self.q
        self._log.writerow(
            ["%.4f" % (time.time() - self._t0), STAGE_NAMES[self.stage]]
            + ["%.4f" % v for v in q] + ["%.3f" % v for v in self.rm_qd]
            + ["%.3f" % v for v in self.iq_des] + ["%.3f" % v for v in self.rm_iq]
            + [self.rm_online, int(time.time() < self.set_zero_until)] + [event])

    # -------- LCM --------
    def _on_data(self, channel: str, data: bytes) -> None:
        msg = hopper_data_lcmt.decode(data)
        self.rm_q_raw = np.array(msg.rm_q, dtype=float)
        self.rm_qd = np.array(msg.rm_qd, dtype=float)
        self.rm_iq = np.array(msg.rm_iq, dtype=float)
        self.rm_online = int(msg.rm_online)
        self.have_data = True

    def _on_gamepad(self, channel: str, data: bytes) -> None:
        msg = gamepad_lcmt.decode(data)
        lt = float(getattr(msg, "leftTriggerAnalog", 0.0)) > 0.5
        rt = float(getattr(msg, "rightTriggerAnalog", 0.0)) > 0.5
        b = bool(getattr(msg, "b", 0))
        if lt and not self._last_lt:
            if self.stage == IDLE:
                self._start(-abs(self.args.range), "LT")
            else:
                self.event("LT ignored (busy; B to abort)")
        if rt and not self._last_rt:
            if self.stage == IDLE:
                self._start(+abs(self.args.range), "RT")
            else:
                self.event("RT ignored (busy; B to abort)")
        if b and not self._last_b:
            self.stage = IDLE
            self.iq_des = np.zeros(3)
            self.event("B -> abort, 0 A")
        self._last_lt = lt
        self._last_rt = rt
        self._last_b = b

    # -------- control --------
    def _start(self, target: float, name: str) -> None:
        """PRE-ZERO at the current position, then drive to target & re-zero there."""
        self.target = float(target)
        self.stage = PREZERO
        self.stage_t0 = time.time()
        self.set_zero_until = time.time() + 0.1   # pre-zero pulse NOW
        self.event("%s -> PRE-ZERO here, then drive to %+.1f rad (cap %.1f A)"
                   % (name, self.target, self.args.iq_max))

    def _pd_to(self, target: float) -> np.ndarray:
        a = self.args
        return np.clip(a.kp * (target - self.q) - a.kd * self.rm_qd,
                       -a.iq_max, a.iq_max)

    def _update(self) -> None:
        a = self.args
        now = time.time()
        if self.stage == IDLE:
            self.iq_des = np.zeros(3)
            return
        if self.rm_online != 7:
            self.iq_des = np.zeros(3)   # feedback lost: coast, stage stays
            return

        if self.stage == PREZERO:
            # 0 A while the driver latches the new zero and fresh rm_q flows back.
            self.iq_des = np.zeros(3)
            if (now - self.stage_t0) >= 0.3:
                self.stage = DRIVE
                self.event("pre-zero done (q=[%+.3f %+.3f %+.3f]) -> drive to %+.1f rad"
                           % (*self.q, self.target))
            return
        if self.stage == DRIVE:
            if bool(np.all(np.abs(self.q - self.target) <= a.tol)):
                self.stage = HOLD
                self.stage_t0 = now
                self.event("reached %+.1f rad -> hold %.1f s, then re-zero here"
                           % (self.target, a.hold))
            self.iq_des = self._pd_to(self.target)
        elif self.stage == HOLD:
            if (now - self.stage_t0) >= a.hold:
                # Ask the JETSON DRIVER to latch this position as the new zero:
                # pulse rm_set_zero for 0.1 s (edge-triggered on the driver side).
                self.set_zero_until = now + 0.1
                self.stage = IDLE
                self.iq_des = np.zeros(3)
                self.event("RE-ZERO: rm_set_zero pulse sent, q was [%+.3f %+.3f %+.3f] "
                           "-> driver sets rm_q=0 here; idle (0 A)" % tuple(self.q))
                return
            self.iq_des = self._pd_to(self.target)

    # -------- IO --------
    def _publish_cmd(self) -> None:
        msg = hopper_cmd_lcmt()
        msg.tau_ff = [0.0, 0.0, 0.0]
        msg.q_des = [0.0, 0.0, 0.0]
        msg.qd_des = [0.0, 0.0, 0.0]
        msg.kp_joint = [0.0, 0.0, 0.0]   # legs FREE even in PD mode
        msg.kd_joint = [0.0, 0.0, 0.0]
        msg.rm_iq_des = [float(x) for x in self.iq_des]
        msg.rm_set_zero = 1 if time.time() < self.set_zero_until else 0
        self.lc.publish("hopper_cmd_lcmt", msg.encode())

    def run(self) -> None:
        try:
            self._run_loop()
        finally:
            # Always leave the motors coasting: stream 0 A for a short burst so
            # the last command the Jetson driver holds is zero (its own 100/200ms
            # freshness timeouts are the backstop if these packets are lost).
            self.stage = IDLE
            self.iq_des = np.zeros(3)
            for _ in range(50):   # 0.1 s @ 500 Hz
                self._publish_cmd()
                time.sleep(0.002)
            self.event("exit -> streamed 0 A stop command")
            self._log_fp.close()
            print("[rm] log saved: %s" % self.log_path)

    def _run_loop(self) -> None:
        dt = 0.002  # 500 Hz, same as the full controller
        print("Waiting for hopper_data_lcmt... (is hopper_driver running on the Jetson?)")
        print("Gamepad: X = arm  |  LT = go -%0.f & re-zero  |  RT = go +%0.f & re-zero  |  B = stop"
              % (abs(self.args.range), abs(self.args.range)))
        next_t = time.time()
        last_print = 0.0
        while True:
            # Drain LCM without blocking the 500 Hz command stream.
            while self.lc.handle_timeout(0) > 0:
                pass
            self._update()
            self._publish_cmd()

            now = time.time()
            # 100 Hz CSV data rows
            if self.have_data and (now - self._last_row_t) >= 0.01:
                self._last_row_t = now
                self._write_row()
            if now - last_print >= 0.2:
                last_print = now
                if self.have_data:
                    print("[%s] q=[%7.3f %7.3f %7.3f]  qd=[%6.2f %6.2f %6.2f]  "
                          "iq_des=[%5.2f %5.2f %5.2f]  iq_fb=[%5.2f %5.2f %5.2f]  online=%d"
                          % (STAGE_NAMES[self.stage],
                             *self.q, *self.rm_qd, *self.iq_des, *self.rm_iq,
                             self.rm_online))
                else:
                    print("(no hopper_data_lcmt yet)")
                self._log_fp.flush()

            next_t += dt
            sleep = next_t - time.time()
            if sleep > 0:
                time.sleep(sleep)
            else:
                next_t = time.time()


def main() -> None:
    ap = argparse.ArgumentParser(description="Standalone RM M2006 LT/RT test (no legs, no props)")
    ap.add_argument("--range", type=float, default=11.0, help="motion amplitude (rad): LT -> -range, RT -> +range (re-zero at the target). Default 11")
    ap.add_argument("--iq-max", dest="iq_max", type=float, default=3.0, help="current cap (A), default 3")
    ap.add_argument("--kp", type=float, default=2.0, help="PD kp (A/rad), default 2")
    ap.add_argument("--kd", type=float, default=0.05, help="PD kd (A/(rad/s)), default 0.05")
    ap.add_argument("--tol", type=float, default=0.3, help="in-place / zero-check tolerance (rad), default 0.3")
    ap.add_argument("--hold", type=float, default=1.0, help="hold time at the target before the re-zero (s), default 1")
    args = ap.parse_args()
    try:
        RmTriggerTest(args).run()
    except KeyboardInterrupt:
        pass  # zero-current stop burst already sent in run()'s finally block


if __name__ == "__main__":
    main()
