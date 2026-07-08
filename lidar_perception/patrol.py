#!/usr/bin/env python3
"""
Waypoint patrol node (pure LCM, no ROS).

hopper_odom_lcmt (from odom_lcm_bridge.py) -> P controller over waypoints
  -> hopper_nav_cmd_lcmt (desired XY velocity in the map frame)

The ModeE LCM controller decides whether to USE this command (SELECT button
toggles patrol; stick input / B always wins back). This node just keeps
publishing; `active=0` is sent whenever odom is stale/degraded so the
controller falls back to the stick automatically.

Run:  python3 patrol.py            (any python with lcm + numpy + yaml)
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np
import yaml

import lcm

_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_CUR_DIR, "..", "upper_controller_pc", "hopper_lcm_types", "lcm_types"))
from python.hopper_odom_lcmt import hopper_odom_lcmt  # type: ignore
from python.hopper_nav_cmd_lcmt import hopper_nav_cmd_lcmt  # type: ignore


class Patrol:
    def __init__(self, cfg: dict):
        p = cfg.get("patrol", {})
        self.rate_hz = float(p.get("rate_hz", 50.0))
        self.kp = float(p.get("kp", 0.6))
        self.v_max = float(p.get("v_max_mps", 0.4))
        self.arrive_r = float(p.get("arrive_radius_m", 0.35))
        self.loop = bool(p.get("loop", True))
        self.odom_stale_s = float(p.get("odom_stale_s", 0.3))

        wp_file = str(p.get("waypoints_file", os.path.join(_CUR_DIR, "waypoints.yaml")))
        with open(wp_file, "r") as f:
            wps = yaml.safe_load(f).get("waypoints", [])
        self.waypoints = [np.array([float(w[0]), float(w[1])], dtype=float) for w in wps]
        if not self.waypoints:
            raise SystemExit(f"no waypoints in {wp_file}")

        self.lc = lcm.LCM(str(cfg.get("lcm_url", "udpm://239.255.76.67:7667?ttl=255")))
        self.lc.subscribe("hopper_odom_lcmt", self._on_odom)

        self._pos_xy: np.ndarray | None = None
        self._odom_quality = 0
        self._odom_t = 0.0
        self._wp_i = 0
        self._done = False

    def _on_odom(self, channel: str, data: bytes) -> None:
        try:
            msg = hopper_odom_lcmt.decode(data)
        except Exception:
            return
        self._pos_xy = np.array([float(msg.pos[0]), float(msg.pos[1])], dtype=float)
        self._odom_quality = int(msg.quality)
        self._odom_t = time.time()

    def _step(self) -> tuple[np.ndarray, bool, float]:
        """Returns (v_cmd_xy, active, dist_to_wp)."""
        now = time.time()
        fresh = (now - self._odom_t) < self.odom_stale_s
        if (self._pos_xy is None) or (not fresh) or (self._odom_quality != 1) or self._done:
            return np.zeros(2, dtype=float), False, float("nan")

        wp = self.waypoints[self._wp_i]
        err = wp - self._pos_xy
        dist = float(np.linalg.norm(err))

        if dist < self.arrive_r:
            if self._wp_i + 1 < len(self.waypoints):
                self._wp_i += 1
            elif self.loop:
                self._wp_i = 0
            else:
                self._done = True
                return np.zeros(2, dtype=float), True, dist
            wp = self.waypoints[self._wp_i]
            err = wp - self._pos_xy
            dist = float(np.linalg.norm(err))

        v = self.kp * err
        n = float(np.linalg.norm(v))
        if n > self.v_max:
            v *= self.v_max / n
        return v.astype(float), True, dist

    def run(self) -> None:
        dt = 1.0 / self.rate_hz
        t_print = 0.0
        print(f"[patrol] {len(self.waypoints)} waypoints, v_max={self.v_max} m/s, loop={self.loop}")
        while True:
            self.lc.handle_timeout(0)
            v, active, dist = self._step()

            out = hopper_nav_cmd_lcmt()
            out.utime = int(time.time() * 1e6)
            out.v_xy_w = [float(v[0]), float(v[1])]
            out.active = 1 if active else 0
            out.wp_index = int(self._wp_i)
            out.dist_to_wp = float(dist) if math.isfinite(dist) else -1.0
            self.lc.publish("hopper_nav_cmd_lcmt", out.encode())

            now = time.time()
            if now - t_print > 1.0:
                st = "RUN" if active else ("DONE" if self._done else "WAIT-ODOM")
                pos = "?" if self._pos_xy is None else f"[{self._pos_xy[0]:+.2f} {self._pos_xy[1]:+.2f}]"
                print(f"[patrol] {st} wp{self._wp_i} dist={dist:.2f} pos={pos} v=[{v[0]:+.2f} {v[1]:+.2f}]")
                t_print = now
            time.sleep(dt)


def main() -> None:
    with open(os.path.join(_CUR_DIR, "perception_config.yaml"), "r") as f:
        cfg = yaml.safe_load(f)
    Patrol(cfg).run()


if __name__ == "__main__":
    main()
