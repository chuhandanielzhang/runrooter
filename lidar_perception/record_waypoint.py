#!/usr/bin/env python3
"""Append the robot's CURRENT lidar position to waypoints.yaml.

Usage: walk/carry the robot to the desired spot, then run:
  python3 record_waypoint.py
"""
from __future__ import annotations

import os
import sys
import time

import yaml
import lcm

_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_CUR_DIR, "..", "upper_controller_pc", "hopper_lcm_types", "lcm_types"))
from python.hopper_odom_lcmt import hopper_odom_lcmt  # type: ignore


def main() -> None:
    with open(os.path.join(_CUR_DIR, "perception_config.yaml"), "r") as f:
        cfg = yaml.safe_load(f)
    wp_file = str(cfg.get("patrol", {}).get("waypoints_file", os.path.join(_CUR_DIR, "waypoints.yaml")))

    got = {}

    def on_odom(channel, data):
        msg = hopper_odom_lcmt.decode(data)
        if int(msg.quality) == 1:
            got["xy"] = [round(float(msg.pos[0]), 3), round(float(msg.pos[1]), 3)]

    lc = lcm.LCM(str(cfg.get("lcm_url", "udpm://239.255.76.67:7667?ttl=255")))
    lc.subscribe("hopper_odom_lcmt", on_odom)
    t0 = time.time()
    while "xy" not in got and time.time() - t0 < 3.0:
        lc.handle_timeout(200)
    if "xy" not in got:
        raise SystemExit("no healthy hopper_odom_lcmt within 3s -- is the bridge running?")

    doc = {"waypoints": []}
    if os.path.isfile(wp_file):
        with open(wp_file, "r") as f:
            doc = yaml.safe_load(f) or {"waypoints": []}
        doc.setdefault("waypoints", [])
    doc["waypoints"].append(got["xy"])
    with open(wp_file, "w") as f:
        yaml.safe_dump(doc, f, default_flow_style=None, allow_unicode=True)
    print(f"recorded waypoint #{len(doc['waypoints'])}: {got['xy']} -> {wp_file}")


if __name__ == "__main__":
    main()
