#!/bin/bash
# Jetson-local bring-up (runs at boot via hopper-bringup.service).
# Mirrors the Jetson half of scripts/connect_and_start.sh — no PC/rsync needed.
set -o pipefail

need_active(){ systemctl is-active "$1" >/dev/null 2>&1 || systemctl restart "$1"; }

# Multicast MUST hit the wired NIC (boot can put it on lo / WiFi).
JET_ETH=$(ip -4 addr show | awk '/inet 192\.168\.1\./{print $NF; exit}')
if [ -n "$JET_ETH" ]; then
  cur_mc=$(ip route get 239.255.76.67 2>/dev/null | head -1)
  echo "$cur_mc" | grep -q "dev ${JET_ETH}" || ip route replace 224.0.0.0/4 dev "$JET_ETH"
  echo "multicast -> $(ip route get 239.255.76.67 2>/dev/null | head -1 | awk '{print $3}')"
fi

# Max performance for the 500Hz upper (schedutil cannot hold 2ms).
nvpmodel -m 2 >/dev/null 2>&1 || true
jetson_clocks 2>/dev/null || true

need_active canable.service
if [ -e /dev/canable2 ]; then
  need_active canable2.service
  echo "wheel bus: canable2 $(systemctl is-active canable2) (can1)"
else
  echo "wheel bus: /dev/canable2 absent (hop-only)"
fi

systemctl enable --now jetson-power.service >/dev/null 2>&1 || true
systemctl restart jetson-power.service 2>/dev/null || true

# TELEM2/DDS only; retired USB MAVLink prop bridge must stay off.
systemctl stop px4-bridge.service 2>/dev/null || true
systemctl disable px4-bridge.service 2>/dev/null || true

need_active xrce-agent.service
if ls /dev/ttyUSB* >/dev/null 2>&1; then
  echo "IMU: Lpms on $(ls /dev/ttyUSB* | head -1) -> dds bridge = props only"
else
  echo "IMU: no Lpms -> Pixhawk DDS serves IMU + props"
fi
systemctl restart px4-dds-bridge.service

# DDS session race: if bridge stays at 0Hz, bounce agent then bridge.
if ! ls /dev/ttyUSB* >/dev/null 2>&1; then
  dds_ok(){ journalctl -u px4-dds-bridge -n 3 --no-pager 2>/dev/null \
            | grep -qE 'pub +[1-9][0-9]*\.[0-9]+Hz'; }
  sleep 4
  if ! dds_ok; then
    echo "DDS 0Hz -> bounce xrce-agent + bridge"
    systemctl restart xrce-agent.service
    sleep 6
    systemctl restart px4-dds-bridge.service
    sleep 4
  fi
  if dds_ok; then
    echo "DDS IMU: OK ($(journalctl -u px4-dds-bridge -n 1 --no-pager | grep -oE 'pub +[0-9.]+Hz'))"
  else
    echo "WARN: DDS IMU still 0Hz (check TELEM2 / reboot Pixhawk)"
  fi
fi

systemctl restart hopper-driver.service
sleep 3

# CAN silent -> one canable reset (bus-off recovery).
r1=$(cat /sys/class/net/can0/statistics/rx_packets 2>/dev/null||echo 0); sleep 1
r2=$(cat /sys/class/net/can0/statistics/rx_packets 2>/dev/null||echo 0)
if [ "$((r2-r1))" -le 0 ]; then
  echo "CAN silent -> reset canable + hopper-driver"
  systemctl restart canable.service && sleep 2 && systemctl restart hopper-driver.service && sleep 2
  r1=$(cat /sys/class/net/can0/statistics/rx_packets 2>/dev/null||echo 0); sleep 1
  r2=$(cat /sys/class/net/can0/statistics/rx_packets 2>/dev/null||echo 0)
fi

if [ -e /dev/canable2 ]; then
  w1=$(cat /sys/class/net/can1/statistics/rx_packets 2>/dev/null||echo 0); sleep 1
  w2=$(cat /sys/class/net/can1/statistics/rx_packets 2>/dev/null||echo 0)
  if [ "$((w2-w1))" -gt 0 ]; then
    echo "wheel CAN (can1): OK ($((w2-w1)) frames/s)"
  else
    echo "WARN: wheel CAN (can1) silent (wheel battery off?)"
  fi
fi

# Perception: failure must not abort bring-up (camera may be unplugged).
systemctl enable hopper-perception.service >/dev/null 2>&1 || true
if systemctl restart hopper-perception.service 2>/dev/null; then
  echo "perception: restarted"
else
  echo "WARN: hopper-perception failed (D435 unplugged?)"
fi

# Exactly one upper: kill stray run_modee, then start the service.
systemctl stop hopper-upper.service 2>/dev/null || true
pkill -f 'run_modee.py' 2>/dev/null || true
sleep 1
if pgrep -f 'run_modee.py' >/dev/null 2>&1; then
  echo "WARN: stray run_modee still alive; killing -9"
  pkill -9 -f 'run_modee.py' 2>/dev/null || true
  sleep 1
fi
systemctl start hopper-upper.service
echo "upper: hopper-upper started"

echo "SERVICES canable=$(systemctl is-active canable) canable2=$(systemctl is-active canable2 2>/dev/null || echo n/a) driver=$(systemctl is-active hopper-driver) upper=$(systemctl is-active hopper-upper) perception=$(systemctl is-active hopper-perception 2>/dev/null || echo n/a) xrce=$(systemctl is-active xrce-agent) dds=$(systemctl is-active px4-dds-bridge 2>/dev/null || echo stopped) CAN_RX_DELTA=$((r2-r1))"
