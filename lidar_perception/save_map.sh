#!/bin/bash
# 把最近一次建图(Point-LIO 退出时写的 PCD/scans.pcd)归档到 maps/map.pcd。
# 归档后 relocalize.py 就能对这张图做开机重定位。
set -e
SRC="/home/abc/Hopper/lidar_ws/src/point_lio/PCD/scans.pcd"
DST_DIR="/home/abc/Hopper/robot_runtime/lidar_perception/maps"
if [ ! -f "$SRC" ]; then
  echo "XX 没找到 $SRC -- 先用 'run_lidar_stack.sh map' 建图并 Ctrl-C 退出"; exit 1
fi
mkdir -p "$DST_DIR"
STAMP=$(date +%Y%m%d_%H%M%S)
cp "$SRC" "$DST_DIR/map_${STAMP}.pcd"
cp -f "$SRC" "$DST_DIR/map.pcd"
# 旧的重定位变换随新地图作废
rm -f "$DST_DIR/T_map_odom.npy"
echo "OK 地图已归档: $DST_DIR/map.pcd (备份 map_${STAMP}.pcd)"
echo "   旧的 T_map_odom.npy 已删除 -- 下次开机在新图上跑 relocalize.py"
