#!/usr/bin/env python3
"""Subscribe to live LCM and write a CSV in a separate process.

This must NOT run inside hopper-upper. The 500 Hz controller only publishes
LCM; this process records those messages so disk flush cannot stall hopping.

Recorded channels (already on the bus):
  hopper_data_lcmt  q/qd/tau_meas + fold-arm RM (NOT kiwi wheels)
  hopper_cmd_lcmt   tau_ff / q_des
  hopper_imu_lcmt   rpy / gyro / acc
  motor_pwm_lcmt    pwm_values / control_mode
  wheel_cmd_lcmt    kiwi enable + speed_des (Jetson can1)

rm_* is the Pixhawk fold-arm M2006. Kiwi hubs are wheel_cmd_lcmt.
To see "wheels actually moved": wheel_en==1 AND |wheel_w*| > 0.
Driver still gates on PD/PWMPD + 200 ms freshness; command-only log
cannot prove CAN current, but a zero enable/speed is enough to explain
a dead chassis.

Not recorded (never published): ModeE internals such as phase, Tau_prop_des.
"""
from __future__ import annotations

import argparse
import csv
import os
import queue
import signal
import sys
import threading
import time
from datetime import datetime, timedelta, timezone

_HERE = os.path.dirname(os.path.abspath(__file__))
_CTRL = os.path.dirname(_HERE)
for cand in (
    os.path.join(_CTRL, "..", "hopper_lcm_types", "lcm_types"),
    os.path.join(_HERE, "..", "..", "hopper_lcm_types", "lcm_types"),
    "/home/nvidia/hopper_upper/hopper_lcm_types/lcm_types",
    "/home/nvidia/Hopper_srbRL/hopper_lcm_types/lcm_types",
):
    if os.path.isdir(cand):
        sys.path.insert(0, cand)
        break

import lcm  # noqa: E402
from python.hopper_cmd_lcmt import hopper_cmd_lcmt  # type: ignore  # noqa: E402
from python.hopper_data_lcmt import hopper_data_lcmt  # type: ignore  # noqa: E402
from python.hopper_imu_lcmt import hopper_imu_lcmt  # type: ignore  # noqa: E402
from python.motor_pwm_lcmt import motor_pwm_lcmt  # type: ignore  # noqa: E402
from python.wheel_cmd_lcmt import wheel_cmd_lcmt  # type: ignore  # noqa: E402

BJ = timezone(timedelta(hours=8))
HEADER = [
    "wall_time_s",
    "wall_bj",
    "q0", "q1", "q2",
    "qd0", "qd1", "qd2",
    "tau_meas0", "tau_meas1", "tau_meas2",
    "tau_cmd0", "tau_cmd1", "tau_cmd2",
    "q_des0", "q_des1", "q_des2",
    "pwm0", "pwm1", "pwm2", "pwm3", "pwm4", "pwm5",
    "control_mode",
    "roll", "pitch", "yaw",
    "gyro0", "gyro1", "gyro2",
    "acc0", "acc1", "acc2",
    "rm_q0", "rm_q1", "rm_q2",
    "rm_iq0", "rm_iq1", "rm_iq2",
    # Kiwi wheels (Jetson can1). Do not confuse with rm_* fold arms.
    "wheel_en",
    "wheel_w0", "wheel_w1", "wheel_w2",
]


def _bj(ts: float) -> str:
    return datetime.fromtimestamp(ts, tz=BJ).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]


def _vec(msg, name: str, n: int) -> list[float]:
    if msg is None:
        return [float("nan")] * n
    try:
        v = getattr(msg, name)
        return [float(v[i]) for i in range(n)]
    except Exception:
        return [float("nan")] * n


class LcmCsvLogger:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.running = True
        self.lock = threading.Lock()
        self.data = None
        self.cmd = None
        self.imu = None
        self.pwm = None
        self.wheel = None
        self.rows: queue.Queue = queue.Queue(maxsize=4000)
        self.n_written = 0
        self.path = ""

    def _on_data(self, _ch: str, raw: bytes) -> None:
        try:
            msg = hopper_data_lcmt.decode(raw)
        except Exception:
            return
        with self.lock:
            self.data = msg

    def _on_cmd(self, _ch: str, raw: bytes) -> None:
        try:
            msg = hopper_cmd_lcmt.decode(raw)
        except Exception:
            return
        with self.lock:
            self.cmd = msg

    def _on_imu(self, _ch: str, raw: bytes) -> None:
        try:
            msg = hopper_imu_lcmt.decode(raw)
        except Exception:
            return
        with self.lock:
            self.imu = msg

    def _on_pwm(self, _ch: str, raw: bytes) -> None:
        try:
            msg = motor_pwm_lcmt.decode(raw)
        except Exception:
            return
        with self.lock:
            self.pwm = msg

    def _on_wheel(self, _ch: str, raw: bytes) -> None:
        try:
            msg = wheel_cmd_lcmt.decode(raw)
        except Exception:
            return
        with self.lock:
            self.wheel = msg

    def _snapshot_row(self, now: float) -> list:
        with self.lock:
            data, cmd, imu, pwm, wheel = (
                self.data, self.cmd, self.imu, self.pwm, self.wheel
            )
        pwm_vals = _vec(pwm, "pwm_values", 6)
        try:
            cm = int(pwm.control_mode) if pwm is not None else -1
        except Exception:
            cm = -1
        try:
            wheel_en = int(wheel.enable) if wheel is not None else -1
        except Exception:
            wheel_en = -1
        return [
            now,
            _bj(now),
            *_vec(data, "q", 3),
            *_vec(data, "qd", 3),
            *_vec(data, "tauIq", 3),
            *_vec(cmd, "tau_ff", 3),
            *_vec(cmd, "q_des", 3),
            *pwm_vals,
            cm,
            *_vec(imu, "rpy", 3),
            *_vec(imu, "gyro", 3),
            *_vec(imu, "acc", 3),
            *_vec(data, "rm_q", 3),
            *_vec(data, "rm_iq", 3),
            wheel_en,
            *_vec(wheel, "speed_des_rad_s", 3),
        ]

    def _open_csv(self) -> None:
        os.makedirs(self.args.log_dir, exist_ok=True)
        stamp = datetime.now(BJ).strftime("%Y%m%d_%H%M%S")
        self.path = os.path.join(self.args.log_dir, f"lcm_{stamp}.csv")
        latest = os.path.join(self.args.log_dir, "lcm_latest.csv")
        self._fp = open(self.path, "w", newline="")
        self._writer = csv.writer(self._fp)
        self._writer.writerow(HEADER)
        self._fp.flush()
        try:
            if os.path.islink(latest) or os.path.exists(latest):
                os.remove(latest)
            os.symlink(os.path.basename(self.path), latest)
        except Exception:
            pass
        print(f"[lcm-csv] START -> {self.path}  ({self.args.hz:.0f} Hz)", flush=True)

    def _writer_loop(self) -> None:
        last_flush = time.time()
        while self.running or not self.rows.empty():
            try:
                row = self.rows.get(timeout=0.1)
            except queue.Empty:
                continue
            self._writer.writerow(row)
            self.n_written += 1
            now = time.time()
            if (now - last_flush) >= 0.25:
                self._fp.flush()
                last_flush = now

    def run(self) -> None:
        self._open_csv()
        lc = lcm.LCM(self.args.lcm_url)
        lc.subscribe("hopper_data_lcmt", self._on_data)
        lc.subscribe("hopper_cmd_lcmt", self._on_cmd)
        lc.subscribe("hopper_imu_lcmt", self._on_imu)
        lc.subscribe("motor_pwm_lcmt", self._on_pwm)
        lc.subscribe("wheel_cmd_lcmt", self._on_wheel)
        wt = threading.Thread(target=self._writer_loop, daemon=True)
        wt.start()
        dt = 1.0 / max(1.0, float(self.args.hz))
        next_t = time.time()
        try:
            while self.running:
                lc.handle_timeout(5)
                now = time.time()
                if now < next_t:
                    continue
                if now - next_t > 2.0 * dt:
                    next_t = now + dt
                else:
                    next_t += dt
                row = self._snapshot_row(now)
                try:
                    self.rows.put_nowait(row)
                except queue.Full:
                    try:
                        self.rows.get_nowait()
                    except queue.Empty:
                        pass
                    try:
                        self.rows.put_nowait(row)
                    except queue.Full:
                        pass
        finally:
            self.running = False
            wt.join(timeout=2.0)
            try:
                self._fp.flush()
                self._fp.close()
            except Exception:
                pass
            print(f"[lcm-csv] STOP -> {self.path}  ({self.n_written} rows)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Off-loop LCM -> CSV logger")
    ap.add_argument(
        "--lcm-url",
        default=os.environ.get("LCM_DEFAULT_URL", "udpm://239.255.76.67:7667?ttl=255"),
    )
    ap.add_argument(
        "--log-dir",
        default=os.environ.get(
            "LCM_CSV_DIR",
            os.path.join(_CTRL, "logs", "lcm_csv"),
        ),
    )
    ap.add_argument("--hz", type=float, default=200.0)
    args = ap.parse_args()
    log = LcmCsvLogger(args)

    def _stop(_sig=None, _frm=None) -> None:
        log.running = False

    signal.signal(signal.SIGINT, _stop)
    signal.signal(signal.SIGTERM, _stop)
    log.run()


if __name__ == "__main__":
    main()
