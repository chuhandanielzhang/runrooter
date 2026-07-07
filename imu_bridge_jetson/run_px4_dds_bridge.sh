#!/bin/bash
# Launcher for px4_dds_bridge.py: PX4 <-> LCM over uXRCE-DDS (TELEM2).
#   DOWNLINK: vehicle_attitude + sensor_combined -> hopper_imu_lcmt
#   UPLINK  : motor_pwm_lcmt -> VehicleCommand DO_SET_ACTUATOR (props, "Plan B")
# Needs the ROS humble runtime AND the px4_msgs message package (built in ~/px4_ws).
# NOTE: no `set -u` -- ROS setup.bash references unbound vars and would abort.

source /opt/ros/humble/setup.bash
source /home/nvidia/px4_ws/install/setup.bash
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"   # must match PX4 UXRCE_DDS_DOM_ID

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# IMU channel ownership: when an Lpms IG1 is present (/dev/ttyUSB*), hopper_driver
# publishes hopper_imu_lcmt and this bridge must NOT (two publishers would fight).
# The prop uplink runs in BOTH cases -- props always go over TELEM2/DDS now.
IMU_ARGS=""
if ls /dev/ttyUSB* >/dev/null 2>&1; then
  echo ">>> Lpms detected on $(ls /dev/ttyUSB* | head -1) -> --no-imu (props uplink only)" >&2
  IMU_ARGS="--no-imu"
fi

# IMU coordinate UNCHANGED: --rot z150,y-90 makes the DDS transform EXACTLY reproduce the
# old USB px4_bridge (--raw --rot z150,y-90) frame. Only comms (DDS/TELEM2) + rate (500Hz)
# changed -- the published rpy/acc/gyro/quat are byte-for-byte the same as before.
exec python3 "$HERE/px4_dds_bridge.py" --rot z150,y-90 --publish-hz 500 --print-hz 2 $IMU_ARGS "$@"
