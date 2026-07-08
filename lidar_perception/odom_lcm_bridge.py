#!/usr/bin/env python3
"""
Point-LIO odometry -> LCM bridge.

Subscribes /aft_mapped_to_init (nav_msgs/Odometry, Point-LIO):
  - pose:  lidar frame pose in the gravity-aligned odom frame "camera_init"
           (z-UP; x = lidar forward at power-on)
  - twist.linear:  lidar velocity in the odom frame
  - twist.angular: lidar angular velocity in the LIDAR body frame

Publishes "hopper_odom_lcmt":
  - body(FRD) pose/vel in the hopper map frame (NED-style, +Z DOWN),
    lever-arm + mount-rotation extrinsics applied,
  - optional relocalization transform T_map_odom (from relocalize.py),
  - quality gate: finite / jump / stale checks.

Run with SYSTEM python3 (ROS Humble), NOT conda:
  source /opt/ros/humble/setup.bash && source ~/Hopper/lidar_ws/install/setup.bash
  /usr/bin/python3 odom_lcm_bridge.py
"""
from __future__ import annotations

import math
import os
import sys
import time

import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from nav_msgs.msg import Odometry

import lcm

_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(_CUR_DIR, "..", "upper_controller_pc", "hopper_lcm_types", "lcm_types"))
from python.hopper_odom_lcmt import hopper_odom_lcmt  # type: ignore

# odom frame (z-up) -> hopper map frame (z-down): 180 deg about +X
_C_MAP_ODOM = np.diag([1.0, -1.0, -1.0])


def _quat_xyzw_to_R(q) -> np.ndarray:
    x, y, z, w = float(q.x), float(q.y), float(q.z), float(q.w)
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-12:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=float,
    )


def _R_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=float)
    tr = float(np.trace(R))
    if tr > 0.0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=float)
    q /= max(1e-12, float(np.linalg.norm(q)))
    if q[0] < 0.0:
        q = -q
    return q


def _R_to_rpy_zyx(R: np.ndarray) -> np.ndarray:
    """Aerospace ZYX extraction, same convention as core.py `_R_to_rpy_xyz`."""
    roll = math.atan2(R[2, 1], R[2, 2])
    pitch = -math.asin(max(-1.0, min(1.0, R[2, 0])))
    yaw = math.atan2(R[1, 0], R[0, 0])
    return np.array([roll, pitch, yaw], dtype=float)


def _rpy_deg_to_R(rpy_deg) -> np.ndarray:
    r, p, y = [math.radians(float(v)) for v in rpy_deg]
    cr, sr = math.cos(r), math.sin(r)
    cp, sp = math.cos(p), math.sin(p)
    cy, sy = math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]], dtype=float)
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]], dtype=float)
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]], dtype=float)
    return Rz @ Ry @ Rx


class OdomLcmBridge(Node):
    def __init__(self, cfg: dict):
        super().__init__("hopper_odom_lcm_bridge")
        self.cfg = cfg
        self.lc = lcm.LCM(str(cfg.get("lcm_url", "udpm://239.255.76.67:7667?ttl=255")))

        ext = cfg.get("extrinsic", {})
        # R_bl: lidar frame -> body FRD frame; t_bl: lidar origin in body frame (m)
        self.R_bl = _rpy_deg_to_R(ext.get("rpy_deg", [180.0, 0.0, 0.0]))
        self.t_bl = np.asarray(ext.get("t_bl_m", [0.0, 0.0, 0.0]), dtype=float).reshape(3)
        # body origin expressed in the lidar frame (lever arm used every packet)
        self.r_b_in_l = (-self.R_bl.T @ self.t_bl).reshape(3)

        hl = cfg.get("health", {})
        self.max_jump = float(hl.get("max_pos_jump_m", 0.5))

        # Optional relocalization transform (odom frame -> saved-map frame, z-up
        # space, 4x4). Produced by relocalize.py; absence = odometry-only mode.
        self.T_map_odom: np.ndarray | None = None
        self.localized = 0
        reloc_file = str(cfg.get("reloc_transform_file", ""))
        if reloc_file and os.path.isfile(reloc_file):
            try:
                T = np.load(reloc_file).reshape(4, 4)
                if np.all(np.isfinite(T)):
                    self.T_map_odom = T
                    self.localized = 1
                    self.get_logger().info(f"reloc transform loaded: {reloc_file}")
            except Exception as e:
                self.get_logger().warn(f"reloc transform load failed: {e}")

        self._prev_p_map: np.ndarray | None = None
        self._n_rx = 0
        self._n_bad = 0
        self._t_rate = time.time()

        self.sub = self.create_subscription(
            Odometry, str(cfg.get("odom_topic", "/aft_mapped_to_init")), self._on_odom, 50
        )
        self.get_logger().info(
            f"bridge up: {cfg.get('odom_topic')} -> hopper_odom_lcmt | "
            f"R_bl rpy_deg={ext.get('rpy_deg')} t_bl={self.t_bl.tolist()} localized={self.localized}"
        )

    def _on_odom(self, msg: Odometry) -> None:
        p_l = np.array(
            [msg.pose.pose.position.x, msg.pose.pose.position.y, msg.pose.pose.position.z],
            dtype=float,
        )
        R_ol = _quat_xyzw_to_R(msg.pose.pose.orientation)  # lidar -> odom
        v_l = np.array(
            [msg.twist.twist.linear.x, msg.twist.twist.linear.y, msg.twist.twist.linear.z],
            dtype=float,
        )  # odom frame
        w_l = np.array(
            [msg.twist.twist.angular.x, msg.twist.twist.angular.y, msg.twist.twist.angular.z],
            dtype=float,
        )  # lidar body frame

        # ---- body pose/vel in odom frame (z-up), lever arm applied ----
        p_b = p_l + R_ol @ self.r_b_in_l
        R_ob = R_ol @ self.R_bl.T  # body -> odom
        v_b = v_l + R_ol @ np.cross(w_l, self.r_b_in_l)

        # ---- optional reloc: odom frame -> saved-map frame (still z-up) ----
        if self.T_map_odom is not None:
            Tm = self.T_map_odom
            p_b = Tm[0:3, 0:3] @ p_b + Tm[0:3, 3]
            R_ob = Tm[0:3, 0:3] @ R_ob
            v_b = Tm[0:3, 0:3] @ v_b

        # ---- z-up odom/map space -> hopper map frame (+Z DOWN) ----
        p_map = _C_MAP_ODOM @ p_b
        v_map = _C_MAP_ODOM @ v_b
        R_mb = _C_MAP_ODOM @ R_ob  # body FRD -> hopper map

        quality = 1
        if not (np.all(np.isfinite(p_map)) and np.all(np.isfinite(v_map)) and np.all(np.isfinite(R_mb))):
            quality = 0
        elif self._prev_p_map is not None and float(np.linalg.norm(p_map - self._prev_p_map)) > self.max_jump:
            quality = 0
            self._n_bad += 1
        if quality == 1:
            self._prev_p_map = p_map.copy()

        out = hopper_odom_lcmt()
        stamp = msg.header.stamp
        out.utime = int(stamp.sec) * 1_000_000 + int(stamp.nanosec) // 1_000
        out.pos = [float(v) for v in p_map]
        out.vel = [float(v) for v in v_map]
        out.quat = [float(v) for v in _R_to_quat_wxyz(R_mb)]
        out.rpy = [float(v) for v in _R_to_rpy_zyx(R_mb)]
        out.localized = int(self.localized)
        out.quality = int(quality)
        self.lc.publish("hopper_odom_lcmt", out.encode())

        self._n_rx += 1
        now = time.time()
        if now - self._t_rate >= 5.0:
            hz = self._n_rx / (now - self._t_rate)
            rpy = _R_to_rpy_zyx(R_mb)
            self.get_logger().info(
                f"odom {hz:5.1f} Hz | pos_map=[{p_map[0]:+.2f} {p_map[1]:+.2f} {p_map[2]:+.2f}] "
                f"yaw={math.degrees(rpy[2]):+.1f}deg | bad={self._n_bad}"
            )
            self._n_rx = 0
            self._n_bad = 0
            self._t_rate = now


def main() -> None:
    cfg_path = os.path.join(_CUR_DIR, "perception_config.yaml")
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    rclpy.init()
    node = OdomLcmBridge(cfg)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
