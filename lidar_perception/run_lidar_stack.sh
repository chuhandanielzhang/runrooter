#!/bin/bash
# Mid-360 感知栈一键启动:驱动 + Point-LIO + odom→LCM 桥。
#
# 用法:
#   bash run_lidar_stack.sh            # 定位/巡视模式(不存图, 无 rviz)
#   bash run_lidar_stack.sh map        # 建图模式(退出时保存 PCD, 开 rviz)
#   bash run_lidar_stack.sh rviz       # 定位模式 + rviz 可视化
# Ctrl-C 一键全停。建图后用 save_map.sh 把地图归档到 lidar_perception/maps/。
set -e
MODE="${1:-run}"

# 剥离 conda / mujoco 环境(ROS 必须用系统 python)
export PATH=/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin
unset LD_LIBRARY_PATH PYTHONPATH CONDA_PREFIX

source /opt/ros/humble/setup.bash
source /home/abc/Hopper/lidar_ws/install/setup.bash

PL_SHARE=$(ros2 pkg prefix point_lio)/share/point_lio

PIDS=()
cleanup() {
  echo; echo "[lidar_stack] stopping..."
  for p in "${PIDS[@]}"; do kill -INT "$p" 2>/dev/null || true; done
  wait 2>/dev/null || true
}
trap cleanup EXIT INT TERM

echo "[lidar_stack] 1/3 livox_ros_driver2 (Mid-360s @ 192.168.1.186)"
ros2 launch livox_ros_driver2 msg_MID360s_launch.py &
PIDS+=($!)
sleep 3

PCD_SAVE=false
RVIZ=false
if [ "$MODE" = "map" ]; then PCD_SAVE=true; RVIZ=true; fi
if [ "$MODE" = "rviz" ]; then RVIZ=true; fi

echo "[lidar_stack] 2/3 Point-LIO (mode=$MODE, pcd_save=$PCD_SAVE)"
# 直接 ros2 run(不走 mapping_mid360.launch.py):launch 文件不透传 pcd_save 参数。
# inline 参数与 launch 文件保持一致。
ros2 run point_lio pointlio_mapping --ros-args \
  --params-file "$PL_SHARE/config/mid360.yaml" \
  -p use_imu_as_input:=false \
  -p prop_at_freq_of_imu:=true \
  -p check_satu:=true \
  -p init_map_size:=10 \
  -p point_filter_num:=3 \
  -p space_down_sample:=true \
  -p filter_size_surf:=0.5 \
  -p filter_size_map:=0.5 \
  -p cube_side_length:=1000.0 \
  -p runtime_pos_log_enable:=false \
  -p pcd_save.pcd_save_en:=$PCD_SAVE &
PIDS+=($!)
if [ "$PCD_SAVE" = "true" ]; then
  echo "  建图模式: Ctrl-C 退出时地图保存到 point_lio 源码目录 PCD/scans.pcd"
fi
if [ "$RVIZ" = "true" ]; then
  ros2 run rviz2 rviz2 -d "$PL_SHARE/rviz_cfg/loam_livox.rviz" >/dev/null 2>&1 &
  PIDS+=($!)
fi
sleep 3

echo "[lidar_stack] 3/3 odom -> LCM bridge (hopper_odom_lcmt)"
/usr/bin/python3 /home/abc/Hopper/robot_runtime/lidar_perception/odom_lcm_bridge.py &
PIDS+=($!)

echo "[lidar_stack] all up. Ctrl-C to stop."
wait -n || true
