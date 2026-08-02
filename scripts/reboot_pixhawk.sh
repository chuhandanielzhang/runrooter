#!/bin/bash
# Reboot the Pixhawk (PX4 FMU-v6X) on the Jetson, FROM the PC.
#
# Plan B only: reboot over TELEM2/DDS (VehicleCommand PREFLIGHT_REBOOT_SHUTDOWN
# on /fmu/in/vehicle_command). No USB MAVLink, no px4-bridge.
#
# Order:
#   1. stop px4-bridge          - ensure retired USB bridge stays off
#   2. reboot PX4 via DDS       - TELEM2 / uXRCE-DDS
#   3. restart xrce-agent       - fresh DDS session for the rebooted FMU
#   4. wait ~16 s               - PX4 boot + DDS reconnect
#   5. restart px4-dds-bridge   - IMU + prop path
#   6. verify services
#
# Safe only when props are NOT spinning and the hop controller is not running.
#
# Usage:
#   ./scripts/reboot_pixhawk.sh
#   ./scripts/reboot_pixhawk.sh nvidia@nvidia-desktop.local
#   ./scripts/reboot_pixhawk.sh nvidia@192.168.1.123
set -e
unset LD_LIBRARY_PATH   # drop conda OpenSSL so system ssh works

JETSON=${1:-nvidia@192.168.1.100}
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=6 -o StrictHostKeyChecking=accept-new)

timeout 95 ssh "${SSH_OPTS[@]}" "$JETSON" bash -s <<'REMOTE'
set -e

echo "[1] stop px4-bridge (USB retired, keep off)"
sudo systemctl stop px4-bridge.service 2>/dev/null || true
sleep 1

echo "[2] reboot PX4 via TELEM2/DDS"
source /opt/ros/humble/setup.bash
source /home/nvidia/px4_ws/install/setup.bash
export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-0}"
python3 - <<'PY'
import time
import rclpy
from px4_msgs.msg import VehicleCommand

rclpy.init()
node = rclpy.create_node("hopper_reboot_px4")
pub = node.create_publisher(VehicleCommand, "/fmu/in/vehicle_command", 10)
time.sleep(0.4)  # let discovery settle
msg = VehicleCommand()
msg.command = int(getattr(
    VehicleCommand, "VEHICLE_CMD_PREFLIGHT_REBOOT_SHUTDOWN", 246
))
msg.param1 = 1.0  # reboot autopilot
msg.target_system = 1
msg.target_component = 1
msg.source_system = 255
msg.source_component = 190
msg.from_external = True
for _ in range(3):  # a few copies: DDS is best-effort around reboot
    msg.timestamp = int(node.get_clock().now().nanoseconds / 1000)
    pub.publish(msg)
    time.sleep(0.05)
print("   reboot command sent (TELEM2/DDS)")
node.destroy_node()
rclpy.shutdown()
PY

echo "[3] restart xrce-agent (fresh, ready for PX4 reconnect)"
sudo systemctl restart xrce-agent.service

echo "[4] wait 16s for PX4 boot + DDS reconnect"
sleep 16

echo "[5] restart px4-dds-bridge"
sudo systemctl restart px4-dds-bridge.service
sleep 3

echo "[6] verify services"
systemctl is-active xrce-agent.service px4-dds-bridge.service
systemctl is-active px4-bridge.service 2>/dev/null || echo "px4-bridge inactive (ok)"
REMOTE

echo "[done] Pixhawk rebooted over TELEM2/DDS."
