#!/bin/bash
# Robust launcher for px4_bridge.py (SPLIT path: props ONLY over USB).
#
# /dev/ttyACM* numbers swap between the Pixhawk and the CANable on every reboot, so
# we auto-detect the Pixhawk via its STABLE /dev/serial/by-id symlink, wait for it,
# then exec the bridge. Meant for the px4-bridge systemd service (Restart=always).
#
# In the split architecture the IMU comes from px4-dds-bridge (DDS/TELEM2), so this
# bridge MUST run with --no-imu (props only) -> it does NOT publish hopper_imu_lcmt.
# The service appends --no-imu; pass-through args also work standalone.
#
# Props are driven over PWM (MAV_CMD_DO_SET_ACTUATOR) on this Auterion PX4 1.14.3
# build, which has no DSHOT_CONFIG param -> NO --dshot-3d here.
#
# Usage: run_px4_bridge.sh [extra args passed through to px4_bridge.py]
set -u

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BRIDGE="$HERE/px4_bridge.py"

find_dev() {
    for pat in '*Auterion*PX4*' '*PX4*FMU*' '*Pixhawk*' '*ArduPilot*'; do
        for d in /dev/serial/by-id/$pat; do
            [ -e "$d" ] || continue
            # Skip the PX4 BOOTLOADER CDC port (name contains _BL_): it appears for a
            # few seconds on power-up, does NOT speak MAVLink, and would restart-loop.
            case "$d" in
                *_BL_*|*PX4_BL*) continue ;;
            esac
            echo "$d"; return 0
        done
    done
    return 1
}

DEV=""
for i in $(seq 1 60); do
    if DEV="$(find_dev)"; then
        break
    fi
    sleep 1
done

if [ -z "$DEV" ]; then
    echo "!! px4_bridge launcher: no Pixhawk found under /dev/serial/by-id (waited 60s)" >&2
    ls -l /dev/serial/by-id/ 2>&1 >&2 || true
    exit 1
fi

echo ">>> px4_bridge launcher: using $DEV" >&2
# Prop MAVLink re-stream is capped at 150Hz: PX4 ACKs every DO_SET_ACTUATOR on this
# same USB link, so 400Hz+ floods COMMAND_ACK and the actuator outputs FREEZE (pwm
# stuck) -- this is independent of --no-imu. The PC still publishes motor_pwm_lcmt at
# 500Hz; only this re-stream is limited. --no-imu is normally supplied by the service;
# passed-through "$@" lets it (or any override) reach px4_bridge.py.
exec python3 "$BRIDGE" --dev "$DEV" --print-hz 0 --prop-rate 150 "$@"
