#!/bin/bash
# Jetson-local bring-up (runs at boot via hopper-bringup.service).
# Mirrors the Jetson half of scripts/connect_and_start.sh — no PC/rsync needed.
set -o pipefail

need_active(){ systemctl is-active "$1" >/dev/null 2>&1 || systemctl restart "$1"; }

# Multicast: wired eth when the cable has carrier (PC can sniff); else lo so
# driver/upper/SELECT keep working outdoors with no ethernet.
if [ -x /usr/local/bin/lcm_multicast_route.sh ]; then
  /usr/local/bin/lcm_multicast_route.sh || true
else
  JET_ETH=$(ip -4 addr show | awk '/inet 192\.168\.1\./{print $NF; exit}')
  if [ -n "$JET_ETH" ] && [ "$(cat /sys/class/net/${JET_ETH}/carrier 2>/dev/null || echo 0)" = "1" ]; then
    ip route replace 224.0.0.0/4 dev "$JET_ETH"
  else
    ip route replace 224.0.0.0/4 dev lo
  fi
  echo "multicast -> $(ip route get 239.255.76.67 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="dev"){print $(i+1); exit}}')"
fi

# Max performance for the 500Hz upper (schedutil cannot hold 2ms).
nvpmodel -m 2 >/dev/null 2>&1 || true
jetson_clocks 2>/dev/null || true

# 2026-08-12: candleLight/gs_usb firmware -- can0/can1 are configured by udev
# + can*-gsusb.service on enumeration. Just verify/nudge here.
if [ ! -d /sys/class/net/can0 ] || [ "$(cat /sys/class/net/can0/operstate 2>/dev/null)" = "down" ]; then
  systemctl start can0-gsusb.service 2>/dev/null || true
fi
if [ -d /sys/class/net/can1 ]; then
  [ "$(cat /sys/class/net/can1/operstate 2>/dev/null)" = "down" ] && systemctl start can1-gsusb.service 2>/dev/null
  echo "wheel bus: can1 $(cat /sys/class/net/can1/operstate 2>/dev/null)"
else
  echo "wheel bus: can1 absent (hop-only)"
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

# SELECT/Back watcher: 1x start / 2x stop hopper-upper (independent of upper).
systemctl enable hopper-gamepad-upper.service >/dev/null 2>&1 || true
systemctl restart hopper-gamepad-upper.service 2>/dev/null || true

# CAN silent -> one can0 reset (bus-off recovery).
r1=$(cat /sys/class/net/can0/statistics/rx_packets 2>/dev/null||echo 0); sleep 1
r2=$(cat /sys/class/net/can0/statistics/rx_packets 2>/dev/null||echo 0)
if [ "$((r2-r1))" -le 0 ]; then
  echo "CAN silent -> reset can0 + hopper-driver"
  systemctl restart can0-gsusb.service && sleep 2 && systemctl restart hopper-driver.service && sleep 2
  r1=$(cat /sys/class/net/can0/statistics/rx_packets 2>/dev/null||echo 0); sleep 1
  r2=$(cat /sys/class/net/can0/statistics/rx_packets 2>/dev/null||echo 0)
fi

if [ -d /sys/class/net/can1 ]; then
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

echo "SERVICES can0=$(cat /sys/class/net/can0/operstate 2>/dev/null || echo n/a) can1=$(cat /sys/class/net/can1/operstate 2>/dev/null || echo n/a) driver=$(systemctl is-active hopper-driver) upper=$(systemctl is-active hopper-upper) gamepad=$(systemctl is-active hopper-gamepad-upper 2>/dev/null || echo n/a) perception=$(systemctl is-active hopper-perception 2>/dev/null || echo n/a) xrce=$(systemctl is-active xrce-agent) dds=$(systemctl is-active px4-dds-bridge 2>/dev/null || echo stopped) CAN_RX_DELTA=$((r2-r1))"
