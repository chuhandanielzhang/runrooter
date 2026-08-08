#!/usr/bin/env python3
"""Replay the fall-rate rescue model against saved ModeE CSV logs.

This is a model/event replay, not a plant resimulation: logged liftoff,
touchdown, takeoff speed, and prop-assist share drive the same
``fall_rescue_model_step`` function used by the real-time controller.
"""

from __future__ import annotations

import argparse
import csv
import glob
import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from modee.core import ModeEConfig, fall_rescue_model_step  # noqa: E402


def _finite(row: dict[str, str], key: str, default: float) -> float:
    try:
        value = float(row.get(key, ""))
        return value if math.isfinite(value) else float(default)
    except (TypeError, ValueError):
        return float(default)


def replay(path: Path, cfg: ModeEConfig) -> dict:
    base = float(cfg.prop_base_thrust_ratio)
    mg = float(cfg.mass_kg * cfg.gravity)
    drop_nom = 0.0
    drop_valid = False
    timing_errors: list[float] = []
    flight: dict | None = None
    flights = 0
    eligible = 0
    triggered = 0
    trigger_rows = 0
    max_rho = 0.0
    max_residual = float("-inf")

    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            t = _finite(row, "t_s", float("nan"))
            if not math.isfinite(t):
                continue

            if int(_finite(row, "liftoff", 0.0)) == 1:
                flight = {
                    "t_lo": t,
                    "vz_lo": _finite(row, "vz_lo_m_s", 0.0),
                    "rho_ascent": base,
                    "active": False,
                    "v_model": 0.0,
                    "rho_prev": 0.0,
                    "triggered": False,
                }

            if flight is not None and row.get("phase") == "FLIGHT":
                prop_fz = max(0.0, _finite(row, "prop_energy_fz", 0.0))
                flight["rho_ascent"] = max(
                    float(flight["rho_ascent"]), base + prop_fz / max(1e-6, mg)
                )
                calib_n = max(
                    1, int(cfg.prop_fall_rescue_calib_hops)
                )
                ready = bool(drop_valid and len(timing_errors) >= calib_n)
                margin = float(max(
                    0.0, cfg.prop_fall_rescue_t_margin_min_s
                ))
                if timing_errors:
                    values = np.asarray(timing_errors, dtype=float)
                    std = float(np.std(values, ddof=1)) if values.size >= 2 else 0.0
                    margin = max(
                        margin,
                        float(np.mean(values))
                        + float(cfg.prop_fall_rescue_t_sigma) * std,
                    )
                result = fall_rescue_model_step(
                    t_flight_s=t - float(flight["t_lo"]),
                    vz_lo_mps=float(flight["vz_lo"]),
                    rho_ascent=float(flight["rho_ascent"]),
                    rho_base=base,
                    gravity=float(cfg.gravity),
                    dt=float(cfg.dt),
                    was_active=bool(flight["active"]),
                    v_model_prev_mps=float(flight["v_model"]),
                    rho_prev=float(flight["rho_prev"]),
                    ratio_max=float(cfg.prop_fall_rescue_ratio_max),
                    band_mps=float(cfg.prop_fall_rescue_band_mps),
                    window_s=float(cfg.prop_fall_rescue_window_s),
                    nominal_drop_m=drop_nom,
                    timing_margin_s=margin,
                    baseline_ready=ready,
                )
                flight["active"] = bool(result["active"])
                flight["v_model"] = float(result["v_model_mps"])
                flight["rho_prev"] = float(result["rho"])
                max_residual = max(
                    max_residual, float(result["residual_t_s"])
                )
                if float(result["rho"]) > 1e-9:
                    trigger_rows += 1
                    max_rho = max(max_rho, float(result["rho"]))
                    flight["triggered"] = True

            if (flight is not None
                    and int(_finite(row, "touchdown", 0.0)) == 1):
                flights += 1
                duration = t - float(flight["t_lo"])
                was_triggered = bool(flight["triggered"])
                if was_triggered:
                    triggered += 1
                if drop_valid and len(timing_errors) >= int(
                    cfg.prop_fall_rescue_calib_hops
                ):
                    eligible += 1

                if (not was_triggered and 0.12 <= duration <= 1.5
                        and float(flight["vz_lo"]) > 1e-3):
                    g = float(cfg.gravity)
                    rho_up = min(0.8, max(0.0, float(flight["rho_ascent"])))
                    g_up = max(1e-3, g * (1.0 - rho_up))
                    g_dn = max(1e-3, g * (1.0 - base))
                    vz_lo = float(flight["vz_lo"])
                    t_apex = vz_lo / g_up
                    h_apex = vz_lo * vz_lo / (2.0 * g_up)
                    if drop_valid:
                        t_pred = t_apex + math.sqrt(
                            2.0 * max(0.0, h_apex + drop_nom) / g_dn
                        )
                        timing_errors.append(duration - t_pred)
                        timing_errors = timing_errors[
                            -max(5, int(cfg.prop_fall_rescue_history_n)):
                        ]
                    t_down = max(0.0, duration - t_apex)
                    observed_drop = min(
                        0.5,
                        max(0.0, 0.5 * g_dn * t_down * t_down - h_apex),
                    )
                    if drop_valid:
                        alpha = min(
                            1.0,
                            max(0.0, float(cfg.prop_fall_rescue_drop_alpha)),
                        )
                        drop_nom = (
                            (1.0 - alpha) * drop_nom
                            + alpha * observed_drop
                        )
                    else:
                        drop_nom = observed_drop
                        drop_valid = True
                flight = None

    return {
        "flights": flights,
        "eligible": eligible,
        "triggered": triggered,
        "trigger_rows": trigger_rows,
        "max_rho": max_rho,
        "max_residual_s": (
            max_residual if math.isfinite(max_residual) else float("nan")
        ),
        "calib_n": len(timing_errors),
        "drop_nom_m": drop_nom,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "logs",
        nargs="*",
        help="CSV files/globs; default: logs/exp1*.csv",
    )
    args = parser.parse_args()
    patterns = args.logs or [str(ROOT / "logs" / "exp1*.csv")]
    paths = sorted({
        Path(name).resolve()
        for pattern in patterns
        for name in glob.glob(pattern)
    })
    if not paths:
        parser.error("no CSV logs matched")

    cfg = ModeEConfig()
    total_flights = total_eligible = total_triggered = 0
    print("file,flights,eligible,triggered,max_rho,drop_nom_m")
    for path in paths:
        result = replay(path, cfg)
        total_flights += int(result["flights"])
        total_eligible += int(result["eligible"])
        total_triggered += int(result["triggered"])
        print(
            f"{path.name},{result['flights']},{result['eligible']},"
            f"{result['triggered']},{result['max_rho']:.4f},"
            f"{result['drop_nom_m']:.4f}"
        )
    print(
        f"TOTAL,{total_flights},{total_eligible},{total_triggered},"
        f"false_trigger_rate="
        f"{(total_triggered / max(1, total_eligible)):.6f}"
    )
    return 1 if total_triggered else 0


if __name__ == "__main__":
    raise SystemExit(main())
