#!/usr/bin/env python3
"""
Batch analysis for ModeE CSV logs.

Goal:
- Quantify "jitter" (qd/tau/f_tau_delta) and correlate with fall/instability.
- Summarize per-log stats and sort by severity.

This script is intentionally dependency-free (stdlib only).
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


def _f(row: dict, k: str) -> float:
    try:
        return float(row.get(k, "nan"))
    except Exception:
        return float("nan")


def _i(row: dict, k: str) -> int:
    try:
        return int(float(row.get(k, "0")))
    except Exception:
        return 0


def _finite(x: float) -> bool:
    return math.isfinite(x)


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return float("nan")
    v = sorted(values)
    if len(v) == 1:
        return float(v[0])
    p = max(0.0, min(1.0, float(p)))
    idx = p * (len(v) - 1)
    lo = int(math.floor(idx))
    hi = int(math.ceil(idx))
    if lo == hi:
        return float(v[lo])
    w = idx - lo
    return float((1.0 - w) * v[lo] + w * v[hi])


@dataclass
class LogSummary:
    path: Path
    duration_s: float
    n_rows: int
    n_touchdowns: int
    n_liftoffs: int
    max_abs_roll_deg: float
    max_abs_pitch_deg: float
    fall_time_s: float  # nan if never exceeded threshold
    qd_rms: float
    tau_raw_rms: float
    f_tau_delta_rms: float
    tau_out_scale_p10: float

    def severity(self) -> float:
        # A simple heuristic score: earlier fall + bigger attitude.
        fall_pen = 0.0 if not _finite(self.fall_time_s) else 1.0 / max(1e-3, self.fall_time_s)
        att = max(self.max_abs_roll_deg, self.max_abs_pitch_deg)
        sat = 0.0 if not _finite(self.tau_out_scale_p10) else max(0.0, 1.0 - self.tau_out_scale_p10)
        jitter = 0.0
        if _finite(self.tau_raw_rms):
            jitter += 0.5 * self.tau_raw_rms
        if _finite(self.qd_rms):
            jitter += 0.1 * self.qd_rms
        if _finite(self.f_tau_delta_rms):
            jitter += 0.05 * self.f_tau_delta_rms
        return float(10.0 * fall_pen + 2.0 * att + 5.0 * sat + 0.02 * jitter)


def summarize_csv(path: Path, *, fall_deg: float = 35.0) -> LogSummary:
    rows: list[dict] = []
    with path.open() as f:
        r = csv.DictReader(f)
        for row in r:
            rows.append(row)

    n = len(rows)
    if n == 0:
        return LogSummary(
            path=path,
            duration_s=0.0,
            n_rows=0,
            n_touchdowns=0,
            n_liftoffs=0,
            max_abs_roll_deg=float("nan"),
            max_abs_pitch_deg=float("nan"),
            fall_time_s=float("nan"),
            qd_rms=float("nan"),
            tau_raw_rms=float("nan"),
            f_tau_delta_rms=float("nan"),
            tau_out_scale_p10=float("nan"),
        )

    # time
    ts = [_f(r, "t_s") for r in rows]
    ts_f = [t for t in ts if _finite(t)]
    t0 = ts_f[0] if ts_f else 0.0
    t1 = ts_f[-1] if ts_f else 0.0
    dur = float(max(0.0, t1 - t0))

    # events
    n_td = sum(1 for r in rows if _i(r, "touchdown") == 1)
    n_lo = sum(1 for r in rows if _i(r, "liftoff") == 1)

    # attitude
    roll = [_f(r, "rpy_hat_roll") for r in rows]
    pitch = [_f(r, "rpy_hat_pitch") for r in rows]
    roll_f = [abs(x) for x in roll if _finite(x)]
    pitch_f = [abs(x) for x in pitch if _finite(x)]
    max_roll_deg = float(max(roll_f) * 180.0 / math.pi) if roll_f else float("nan")
    max_pitch_deg = float(max(pitch_f) * 180.0 / math.pi) if pitch_f else float("nan")

    # fall time (first time roll/pitch exceeds threshold)
    fall_time = float("nan")
    for r in rows:
        t = _f(r, "t_s")
        rr = abs(_f(r, "rpy_hat_roll")) * 180.0 / math.pi
        pp = abs(_f(r, "rpy_hat_pitch")) * 180.0 / math.pi
        if _finite(t) and _finite(rr) and _finite(pp) and (max(rr, pp) >= float(fall_deg)):
            fall_time = float(t - t0)
            break

    # jitter metrics: compute RMS on stance rows (where Jacobian/torque mapping is active)
    qd = []
    tau_raw = []
    f_tau = []
    tau_scale = []
    for r in rows:
        if _i(r, "stance") != 1:
            continue
        qd.extend([_f(r, "qd0"), _f(r, "qd1"), _f(r, "qd2")])
        tau_raw.extend([_f(r, "tau_raw0"), _f(r, "tau_raw1"), _f(r, "tau_raw2")])
        f_tau.extend([_f(r, "f_tau_delta0"), _f(r, "f_tau_delta1"), _f(r, "f_tau_delta2")])
        tau_scale.append(_f(r, "tau_out_scale_applied"))

    def rms(v: list[float]) -> float:
        v = [x for x in v if _finite(x)]
        if not v:
            return float("nan")
        return float(math.sqrt(sum(x * x for x in v) / len(v)))

    qd_rms = rms(qd)
    tau_raw_rms = rms(tau_raw)
    f_tau_rms = rms(f_tau)
    tau_scale_f = [x for x in tau_scale if _finite(x)]
    tau_scale_p10 = _percentile(tau_scale_f, 0.10) if tau_scale_f else float("nan")

    return LogSummary(
        path=path,
        duration_s=dur,
        n_rows=n,
        n_touchdowns=n_td,
        n_liftoffs=n_lo,
        max_abs_roll_deg=max_roll_deg,
        max_abs_pitch_deg=max_pitch_deg,
        fall_time_s=fall_time,
        qd_rms=qd_rms,
        tau_raw_rms=tau_raw_rms,
        f_tau_delta_rms=f_tau_rms,
        tau_out_scale_p10=tau_scale_p10,
    )


def iter_csv_files(root: Path) -> Iterable[Path]:
    if root.is_file() and root.suffix.lower() == ".csv":
        yield root
        return
    for p in root.rglob("*.csv"):
        # skip analysis outputs if any
        if "analysis_" in p.parts:
            continue
        yield p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, default="/home/abc/hopper_logs/modee_csv", help="CSV root directory")
    ap.add_argument("--fall-deg", type=float, default=35.0, help="Attitude threshold to mark a 'fall' (deg)")
    ap.add_argument("--top", type=int, default=30, help="Show top-N severe logs")
    args = ap.parse_args()

    root = Path(args.root).expanduser()
    files = sorted(iter_csv_files(root))
    if not files:
        print(f"No CSV files found under: {root}")
        return 2

    sums: list[LogSummary] = []
    for p in files:
        try:
            sums.append(summarize_csv(p, fall_deg=float(args.fall_deg)))
        except Exception as e:
            print(f"[WARN] failed to parse {p}: {e}")

    sums.sort(key=lambda s: s.severity(), reverse=True)

    def fmt(x: float, unit: str = "") -> str:
        if not _finite(x):
            return "nan"
        return f"{x:.3f}{unit}"

    print(f"Parsed {len(sums)} CSV logs from {root}")
    print("Top severe logs:")
    print(
        "score\tfall_s\tmaxR\tmaxP\ttd\tlo\ttauScaleP10\tqdRMS\ttauRawRMS\tfTauRMS\tfile"
    )
    for s in sums[: int(args.top)]:
        print(
            f"{s.severity():.2f}\t"
            f"{fmt(s.fall_time_s,'s')}\t"
            f"{fmt(s.max_abs_roll_deg,'deg')}\t"
            f"{fmt(s.max_abs_pitch_deg,'deg')}\t"
            f"{s.n_touchdowns}\t{s.n_liftoffs}\t"
            f"{fmt(s.tau_out_scale_p10)}\t"
            f"{fmt(s.qd_rms)}\t{fmt(s.tau_raw_rms)}\t{fmt(s.f_tau_delta_rms)}\t"
            f"{s.path.name}"
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())





