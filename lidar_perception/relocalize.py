#!/usr/bin/env python3
"""
One-shot relocalization against a saved map (M4).

Workflow:
  1. Mapping session: run Point-LIO with pcd_save_en:=true, walk the robot
     around, Ctrl-C -> map saved; copy it to lidar_perception/maps/map.pcd
     (save_map.sh does this).
  2. Later sessions: start driver + Point-LIO, keep the robot STILL, run:
       /usr/bin/python3 relocalize.py            # (system python, ROS sourced)
     It accumulates ~3 s of /cloud_registered (odom frame), registers it to
     the saved map (FPFH+RANSAC global init, ICP refine), and writes
     maps/T_map_odom.npy (4x4, map <- odom, z-up space).
  3. Restart odom_lcm_bridge.py -- it loads the file and publishes
     localized=1 poses in the SAVED map frame, so waypoints recorded in that
     map stay valid across power cycles.

Requires: open3d (pip install --user open3d), rclpy, sensor_msgs.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import PointCloud2
import sensor_msgs_py.point_cloud2 as pc2

try:
    import open3d as o3d
except ImportError:
    raise SystemExit("open3d missing: pip install --user open3d")

_CUR_DIR = os.path.dirname(os.path.abspath(__file__))
MAP_PCD = os.path.join(_CUR_DIR, "maps", "map.pcd")
ACCUM_S = 3.0
VOXEL = 0.2


class ScanAccumulator(Node):
    def __init__(self):
        super().__init__("reloc_scan_accum")
        self.points: list[np.ndarray] = []
        self.t0: float | None = None
        self.sub = self.create_subscription(PointCloud2, "/cloud_registered", self._on_cloud, 50)

    def _on_cloud(self, msg: PointCloud2) -> None:
        if self.t0 is None:
            self.t0 = time.time()
        arr = pc2.read_points_numpy(msg, field_names=("x", "y", "z"), skip_nans=True)
        if arr.size:
            self.points.append(arr.astype(np.float64).reshape(-1, 3))

    def done(self) -> bool:
        return self.t0 is not None and (time.time() - self.t0) > ACCUM_S


def main() -> None:
    if not os.path.isfile(MAP_PCD):
        raise SystemExit(f"no saved map at {MAP_PCD} -- run a mapping session + save_map.sh first")

    rclpy.init()
    node = ScanAccumulator()
    print(f"accumulating /cloud_registered for {ACCUM_S:.0f}s ... keep the robot still")
    while rclpy.ok() and not node.done():
        rclpy.spin_once(node, timeout_sec=0.2)
    node.destroy_node()
    rclpy.shutdown()

    if not node.points:
        raise SystemExit("no /cloud_registered received -- is Point-LIO running?")
    scan_np = np.vstack(node.points)
    print(f"scan: {scan_np.shape[0]} pts | map: {MAP_PCD}")

    scan = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(scan_np)).voxel_down_sample(VOXEL)
    mp = o3d.io.read_point_cloud(MAP_PCD).voxel_down_sample(VOXEL)
    for pc in (scan, mp):
        pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL * 4, max_nn=30))

    def fpfh(pc):
        return o3d.pipelines.registration.compute_fpfh_feature(
            pc, o3d.geometry.KDTreeSearchParamHybrid(radius=VOXEL * 10, max_nn=100))

    print("global registration (RANSAC over FPFH)...")
    res = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        scan, mp, fpfh(scan), fpfh(mp), True, VOXEL * 3,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False), 3,
        [o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
         o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(VOXEL * 3)],
        o3d.pipelines.registration.RANSACConvergenceCriteria(200000, 0.999))
    print(f"  ransac fitness={res.fitness:.3f} rmse={res.inlier_rmse:.3f}")
    if res.fitness < 0.2:
        raise SystemExit("global registration too weak (fitness < 0.2) -- move to a more distinctive spot")

    print("ICP refine...")
    icp = o3d.pipelines.registration.registration_icp(
        scan, mp, VOXEL * 1.5, res.transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    print(f"  icp fitness={icp.fitness:.3f} rmse={icp.inlier_rmse:.3f}")
    if icp.fitness < 0.3:
        raise SystemExit("ICP refine too weak (fitness < 0.3) -- not saving")

    T = np.asarray(icp.transformation, dtype=float).reshape(4, 4)

    with open(os.path.join(_CUR_DIR, "perception_config.yaml"), "r") as f:
        out_path = str(yaml.safe_load(f).get(
            "reloc_transform_file", os.path.join(_CUR_DIR, "maps", "T_map_odom.npy")))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    np.save(out_path, T)
    yaw = np.degrees(np.arctan2(T[1, 0], T[0, 0]))
    print(f"T_map_odom saved -> {out_path}")
    print(f"  translation xy=[{T[0,3]:+.2f}, {T[1,3]:+.2f}] m  yaw={yaw:+.1f} deg")
    print("restart odom_lcm_bridge.py to pick it up (localized=1)")


if __name__ == "__main__":
    main()
