#!/bin/bash
# Jetson LCM multicast egress.
#
# ALWAYS use lo for the onboard stack (driver <-> upper <-> SELECT).
# Routing 224/4 via the wired NIC breaks as soon as the cable is unplugged:
# hopper_cmd goes stale and the driver force-DAMPs within 200 ms (X looks
# like damping). PC-side LCM sniff is optional; use HOPPER_LCM_TO_ETH=1 only
# while a cable is plugged in for lab monitoring.
set -e

if [ "${HOPPER_LCM_TO_ETH:-0}" = "1" ]; then
  want=""
  for ifc in $(ip -o -4 addr show | awk '/inet 192\.168\.1\./{print $2}'); do
    if [ "$(cat /sys/class/net/${ifc}/carrier 2>/dev/null || echo 0)" = "1" ]; then
      want="$ifc"
      break
    fi
  done
  if [ -z "$want" ]; then
    want="lo"
    echo "WARN: HOPPER_LCM_TO_ETH=1 but no wired carrier -> lo" >&2
  fi
else
  want="lo"
fi

cur=$(ip route get 239.255.76.67 2>/dev/null | awk '{for(i=1;i<=NF;i++) if($i=="dev"){print $(i+1); exit}}')
if [ "$cur" != "$want" ]; then
  ip route replace 224.0.0.0/4 dev "$want"
  echo "multicast -> $want (was ${cur:-none})"
else
  echo "multicast -> $want (unchanged)"
fi
