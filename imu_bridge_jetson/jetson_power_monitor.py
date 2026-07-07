#!/usr/bin/env python3
"""Jetson input-rail power monitor -> LCM (jetson_power_lcmt).

Reads the onboard INA3221 (VDD_IN channel) through hwmon sysfs, samples fast
(default 100Hz) to catch brownout dips between publishes, and publishes at a
slower rate (default 10Hz) with min/max voltage seen since the last publish.

Shows up in lcm-spy as channel "jetson_power_lcmt":
    vin_v      latest input voltage (V)
    iin_a      latest input current (A)
    pin_w      latest input power (W)
    vin_min_v  lowest V seen in the last publish window  <- brownout detector
    vin_max_v  highest V seen in the last publish window
"""

import argparse
import glob
import os
import sys
import time


def _add_lcm_type_paths():
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = [
        os.environ.get("HOPPER_LCM_PY", ""),
        os.path.join(here, "..", "hopper_lcm_types", "lcm_types", "python"),
        "/home/nvidia/hopper_upper/hopper_lcm_types/lcm_types/python",
        os.path.expanduser("~/hopper_upper/hopper_lcm_types/lcm_types/python"),
        os.path.expanduser("~/Hopper_srbRL/hopper_lcm_types/lcm_types/python"),
    ]
    for c in candidates:
        if c and os.path.isdir(c) and c not in sys.path:
            sys.path.insert(0, c)


_add_lcm_type_paths()
import lcm  # noqa: E402
from jetson_power_lcmt import jetson_power_lcmt  # noqa: E402


def find_vdd_in():
    """Locate the INA3221 VDD_IN channel in hwmon sysfs.

    Returns (voltage_path_mV, current_path_mA) or (None, None).
    Jetson Orin: label "VDD_IN"; older boards: "POM_5V_IN" / "VDD_IN 5V".
    """
    preferred = ("VDD_IN", "POM_5V_IN", "VDD_5V_IN", "SYS 5V", "IN")
    found = []  # (rank, vpath, cpath, label)
    for lab_path in glob.glob("/sys/class/hwmon/hwmon*/in*_label"):
        try:
            with open(lab_path) as f:
                label = f.read().strip()
        except OSError:
            continue
        base = lab_path[: -len("_label")]          # .../inN
        idx = base.rsplit("in", 1)[-1]             # N
        vpath = base + "_input"                    # mV
        cpath = os.path.join(os.path.dirname(lab_path), f"curr{idx}_input")  # mA
        if not os.path.exists(vpath):
            continue
        rank = len(preferred)
        for i, key in enumerate(preferred):
            if key.lower() in label.lower():
                rank = i
                break
        found.append((rank, vpath, cpath if os.path.exists(cpath) else None, label))
    if not found:
        return None, None, None
    found.sort(key=lambda t: t[0])
    rank, vpath, cpath, label = found[0]
    return vpath, cpath, label


def read_mv(path):
    try:
        with open(path) as f:
            return float(f.read().strip())
    except (OSError, ValueError):
        return float("nan")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--url", default="udpm://239.255.76.67:7667?ttl=1")
    ap.add_argument("--channel", default="jetson_power_lcmt")
    ap.add_argument("--sample-hz", type=float, default=100.0,
                    help="sysfs sampling rate (dip detection)")
    ap.add_argument("--publish-hz", type=float, default=10.0)
    args = ap.parse_args()

    vpath, cpath, label = find_vdd_in()
    if vpath is None:
        print("!! no INA3221 hwmon voltage channel found under /sys/class/hwmon/")
        print("   (this carrier board may not have an input power monitor)")
        sys.exit(1)
    print(f"monitoring '{label}': V={vpath}" + (f" I={cpath}" if cpath else " (no current channel)"))

    lc = lcm.LCM(args.url)
    dt = 1.0 / max(args.sample_hz, 1.0)
    pub_every = max(1, int(round(args.sample_hz / max(args.publish_hz, 0.1))))

    vmin = float("inf")
    vmax = float("-inf")
    n = 0
    while True:
        v = read_mv(vpath) * 1e-3
        if v == v:  # not NaN
            vmin = min(vmin, v)
            vmax = max(vmax, v)
        n += 1
        if n >= pub_every:
            i = read_mv(cpath) * 1e-3 if cpath else float("nan")
            msg = jetson_power_lcmt()
            msg.timestamp = int(time.time() * 1e6)
            msg.vin_v = v
            msg.iin_a = i
            msg.pin_w = v * i if i == i else float("nan")
            msg.vin_min_v = vmin if vmin != float("inf") else float("nan")
            msg.vin_max_v = vmax if vmax != float("-inf") else float("nan")
            lc.publish(args.channel, msg.encode())
            vmin = float("inf")
            vmax = float("-inf")
            n = 0
        time.sleep(dt)


if __name__ == "__main__":
    main()
