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

# Base mount rotation z150,y-90 reproduces the old USB px4_bridge frame.
# 2026-08-07 level calibration (robot mechanically level, 501 samples):
# measured roll=-0.413 deg, pitch=-1.917 deg.  Prepending the same measured
# rotations to R_mount makes R_wb_new = R_wb_old*Rx(+0.413)*Ry(+1.917), so
# the current pose reads roll=pitch=0 while preserving yaw.  This corrects
# quat/rpy/gyro/acc together; do not apply a separate RPY-only subtraction.
IMU_ROT="y-1.917,x-0.413,z150,y-90"
# 2026-07-23: props run UNIDIRECTIONAL (--no-bidir): ESC 3D mode kept OFF (it did not
# persist across power cycles on the Hobbywing 4-in-1). Requires DSHOT_3D_ENABLE=0.
exec python3 "$HERE/px4_dds_bridge.py" --rot "$IMU_ROT" --publish-hz 500 --print-hz 2 --no-bidir --prop-reverse "" $IMU_ARGS "$@"
