#!/bin/bash
# Launch the Micro-XRCE-DDS Agent bound to the Jetson 40-pin UART that is wired to
# the Pixhawk TELEM2 port. This is the high-rate, contention-free IMU link.
#
# The agent binary links against ROS humble's Fast-DDS (built with
# -DUAGENT_USE_SYSTEM_FASTDDS=ON), so we must source the ROS env to put
# libfastrtps/libfastcdr on LD_LIBRARY_PATH or it fails with
# "libfastrtps.so.2.6: cannot open shared object file".
# NOTE: no `set -u` -- ROS setup.bash references unbound vars and would abort.

source /opt/ros/humble/setup.bash

DEV="${XRCE_DEV:-/dev/ttyTHS1}"
BAUD="${XRCE_BAUD:-3000000}"   # must match PX4 SER_TEL2_BAUD

# Wait for the UART node to exist (it always does on this board, but be safe).
for _ in $(seq 1 30); do
    [ -e "$DEV" ] && break
    sleep 1
done

echo ">>> MicroXRCEAgent serial --dev $DEV -b $BAUD" >&2
exec MicroXRCEAgent serial --dev "$DEV" -b "$BAUD"
