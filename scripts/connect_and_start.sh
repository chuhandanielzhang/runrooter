#!/bin/bash
# One-shot: connect to Jetson and bring up the lower layer, robustly.
#
# Solves the "every time it's a different problem" pain:
#   - Jetson just rebooted / not finished booting  -> waits for SSH (up to ~120s)
#   - hopper-driver is disabled (won't auto-start)  -> starts it
#   - CAN bus-off after power/ground changes        -> resets canable, retries
#   - PC multicast route hijacked by Mihomo         -> fixes 224.0.0.0/4 route
#   - verifies end-to-end (CAN rx + LCM on PC) before saying OK
#
# Usage (on PC):
#   bash /home/abc/Hopper/hopperleg/connect_and_start.sh           # default IP .100
#   bash /home/abc/Hopper/hopperleg/connect_and_start.sh 192.168.1.123
set -o pipefail
unset LD_LIBRARY_PATH                      # drop conda OpenSSL so system ssh works

JETSON_IP="${1:-192.168.1.100}"
PC_IFACE="${PC_IFACE:-enp44s0}"
LCM_URL="udpm://239.255.76.67:7667?ttl=255"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=6 -o StrictHostKeyChecking=accept-new nvidia@${JETSON_IP}"

say(){ printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
ok(){  printf '   \033[1;32mOK\033[0m  %s\n' "$*"; }
bad(){ printf '   \033[1;31mXX\033[0m  %s\n' "$*"; }

# ── 1. fix PC multicast route (needs sudo; harmless if already correct) ──
say "1/4  PC multicast route -> ${PC_IFACE}"
cur=$(ip route get 239.255.76.67 2>/dev/null | head -1)
if echo "$cur" | grep -q "dev ${PC_IFACE}"; then
  ok "route already via ${PC_IFACE}"
else
  echo "   route is: $cur"
  echo "   fixing (sudo password may be asked)..."
  sudo ip route replace 224.0.0.0/4 dev "${PC_IFACE}" && ok "route replaced" || bad "route fix failed (run manually)"
fi

# ── 2. wait for Jetson SSH (handles 'just rebooted, still booting') ──
say "2/4  waiting for Jetson SSH @ ${JETSON_IP} (up to 120s)"
deadline=$(( $(date +%s) + 120 )); online=0
while [ "$(date +%s)" -lt "$deadline" ]; do
  if $SSH 'echo ok' >/dev/null 2>&1; then online=1; break; fi
  printf '   ...not reachable yet, retrying\r'; sleep 4
done
if [ "$online" != 1 ]; then
  bad "Jetson never came online. Check: powered on? wired cable in? IP correct?"
  echo "   tip: if WiFi (wlo1) shares 192.168.1.x it can steal the route to .100"
  exit 1
fi
ok "SSH up  ($($SSH 'hostname; uptime -p' 2>/dev/null | paste -sd' '))"

# ── 2.5 sync upper layer (PC -> Jetson) so Jetson always runs the latest code ──
say "2.5/4  syncing upper layer to Jetson"
UPPER_SRC="/home/abc/Hopper/robot_runtime/upper_controller_pc/"
rsync -a --delete \
  --exclude 'hopper_controller/logs' \
  --exclude '__pycache__' \
  "${UPPER_SRC}" nvidia@${JETSON_IP}:/home/nvidia/hopper_upper/ \
  && ok "upper synced -> /home/nvidia/hopper_upper" || bad "upper rsync failed"

# Pixhawk bridges (DDS + retired-USB) + their systemd units. The dds bridge now
# also carries the prop uplink ("Plan B"), so the Jetson copy must track the repo.
BRIDGE_SRC="/home/abc/Hopper/robot_runtime/imu_bridge_jetson/"
rsync -a --exclude '__pycache__' \
  "${BRIDGE_SRC}" nvidia@${JETSON_IP}:/home/nvidia/Hopper_srbRL/pixhawk/ \
  && ok "bridges synced -> /home/nvidia/Hopper_srbRL/pixhawk" || bad "bridge rsync failed"
SVC_SRC="/home/abc/Hopper/robot_runtime/services"
rsync -a "${SVC_SRC}/px4-bridge.service" "${SVC_SRC}/px4-dds-bridge.service" \
  "${SVC_SRC}/canable.service" "${SVC_SRC}/canable-watchdog.service" \
  "${SVC_SRC}/canable_watchdog.sh" "${SVC_SRC}/99-canable.rules" \
  "${SVC_SRC}/jetson-power.service" \
  nvidia@${JETSON_IP}:/tmp/hopper_svc/ \
  && $SSH 'sudo install -m644 /tmp/hopper_svc/*.service /etc/systemd/system/ && sudo systemctl daemon-reload' \
  && ok "service units installed + daemon-reload" || bad "service unit install failed"
# Self-healing CAN bridge: udev rule gives the CANable a stable /dev/canable +
# dev-canable.device; canable.service is BindsTo that device (zombie slcand is
# killed on USB re-enumeration, restarted on reappearance); the watchdog covers
# the cases udev can't see and rebinds hopper-driver after every recovery.
$SSH 'sudo install -m644 /tmp/hopper_svc/99-canable.rules /etc/udev/rules.d/ \
  && sudo install -m755 /tmp/hopper_svc/canable_watchdog.sh /usr/local/bin/canable_watchdog.sh \
  && sudo udevadm control --reload-rules && sudo udevadm trigger --subsystem-match=tty --action=add \
  && sudo systemctl daemon-reload \
  && sudo systemctl reenable canable.service >/dev/null 2>&1 \
  && sudo systemctl enable --now canable-watchdog.service >/dev/null 2>&1' \
  && ok "CAN self-heal installed (udev + BindsTo + watchdog)" || bad "CAN self-heal install failed"

# ── 3. bring up lower-layer services on Jetson ──
say "3/4  starting lower + upper on Jetson"
# HOPPER_UPPER (PC env) is forwarded so the heredoc can decide pc-vs-jetson upper.
$SSH "HOPPER_UPPER='${HOPPER_UPPER:-pc}' bash -s" <<'EOF'
need_active(){ systemctl is-active "$1" >/dev/null 2>&1 || sudo systemctl restart "$1"; }
# Jetson multicast route MUST point at the wired NIC. After (re)boot it can land on
# lo or the WiFi (2026-07-06: it was on lo -> LCM stayed Jetson-local, PC saw nothing).
JET_ETH=$(ip -4 addr show | awk '/inet 192\.168\.1\./{print $NF; exit}')
if [ -n "$JET_ETH" ]; then
  cur_mc=$(ip route get 239.255.76.67 2>/dev/null | head -1)
  echo "$cur_mc" | grep -q "dev ${JET_ETH}" || sudo ip route replace 224.0.0.0/4 dev "$JET_ETH"
  echo "   jetson multicast -> $(ip route get 239.255.76.67 | head -1 | awk '{print $3}')"
fi
# Max performance mode + locked clocks: the 500Hz python upper layer needs full
# single-core speed (schedutil at 1.34GHz cannot hold the 2ms budget).
sudo nvpmodel -m 2 >/dev/null 2>&1 || true
sudo jetson_clocks 2>/dev/null || true
need_active canable.service
# Jetson supply-voltage monitor -> lcm-spy channel "jetson_power_lcmt"
sudo systemctl enable --now jetson-power.service >/dev/null 2>&1 || true
sudo systemctl restart jetson-power.service 2>/dev/null || true
# Pixhawk link is TELEM2/DDS ONLY now ("Plan B"): px4-dds-bridge carries BOTH the
# IMU downlink and the prop uplink (motor_pwm_lcmt -> DO_SET_ACTUATOR). The old
# USB MAVLink prop bridge (px4-bridge) is RETIRED -- make sure it is stopped so
# two prop streams never fight.
sudo systemctl stop px4-bridge.service 2>/dev/null || true
sudo systemctl disable px4-bridge.service 2>/dev/null || true
need_active xrce-agent.service
# Always RESTART the dds bridge (not just start): its launcher decides --no-imu
# by probing for an Lpms (/dev/ttyUSB*) at startup, so plugging/unplugging the
# Lpms between runs needs a fresh probe. Prop uplink runs in both cases.
if ls /dev/ttyUSB* >/dev/null 2>&1; then
  echo "   IMU: Lpms on $(ls /dev/ttyUSB* | head -1) -> hopper_driver publishes hopper_imu_lcmt; dds bridge = props only"
else
  echo "   IMU: no Lpms (/dev/ttyUSB*) -> Pixhawk DDS serves IMU + props"
fi
sudo systemctl restart px4-dds-bridge.service
sudo systemctl restart hopper-driver.service     # disabled by default -> always start
sleep 3
# CAN feedback check; if motors silent, try a canable reset once (clears bus-off)
r1=$(cat /sys/class/net/can0/statistics/rx_packets 2>/dev/null||echo 0); sleep 1
r2=$(cat /sys/class/net/can0/statistics/rx_packets 2>/dev/null||echo 0)
if [ "$((r2-r1))" -le 0 ]; then
  echo "   CAN silent -> resetting canable (bus-off recovery)"
  sudo systemctl restart canable.service && sleep 2 && sudo systemctl restart hopper-driver.service && sleep 2
  r1=$(cat /sys/class/net/can0/statistics/rx_packets 2>/dev/null||echo 0); sleep 1
  r2=$(cat /sys/class/net/can0/statistics/rx_packets 2>/dev/null||echo 0)
fi
# Upper layer (ModeE): there must be EXACTLY ONE running controller. During
# 2026-07-05 debugging the Jetson service and a PC-launched run_modee.py ran
# simultaneously and fought over the torque channel (violent shaking).
# Default now: STOP the Jetson upper and let the operator launch from the PC
# (python3 run_modee.py --tau-max ...). Set HOPPER_UPPER=jetson to use the
# Jetson service instead.
sudo systemctl stop hopper-upper.service 2>/dev/null || true
pkill -f 'run_modee.py' 2>/dev/null || true
sleep 1
if pgrep -f 'run_modee.py' >/dev/null 2>&1; then
  echo "   WARN: stray run_modee still alive:"; pgrep -af 'run_modee.py'
  pkill -9 -f 'run_modee.py' 2>/dev/null || true
  sleep 1
fi
if [ "${HOPPER_UPPER:-pc}" = "jetson" ]; then
  sudo systemctl start hopper-upper.service
  echo "   upper: Jetson service started (HOPPER_UPPER=jetson)"
else
  echo "   upper: NOT started on Jetson -> launch on PC: python3 run_modee.py --tau-max ..."
fi
sleep 1
echo "SERVICES canable=$(systemctl is-active canable) driver=$(systemctl is-active hopper-driver) upper=$(systemctl is-active hopper-upper) xrce=$(systemctl is-active xrce-agent) dds=$(systemctl is-active px4-dds-bridge 2>/dev/null || echo stopped) [props=TELEM2/DDS, USB px4-bridge retired]"
echo "CAN_RX_DELTA=$((r2-r1))"
EOF

# ── 4. verify PC actually receives the data ──
say "4/4  PC end-to-end LCM check (3s)"
timeout 12 python3 - <<PY
import time
from collections import Counter
try:
    import lcm
except Exception as e:
    print("   lcm import failed:", e); raise SystemExit(0)
lc = lcm.LCM("${LCM_URL}")
c = Counter()
lc.subscribe(".*", lambda ch,d: c.__setitem__(ch, c[ch]+1))
t0=time.time()
while time.time()-t0<3: lc.handle_timeout(200)
if not c:
    print("   XX  no LCM on PC -> multicast route hijacked? rerun, or: sudo ip route replace 224.0.0.0/4 dev ${PC_IFACE}")
else:
    for k,v in sorted(c.items()):
        print(f"   OK  {k:22s} ~{v/3:.0f} Hz")
PY

cat <<TXT

================ upper + lower BOTH running on Jetson ================
Watch upper status line :  ssh nvidia@${JETSON_IP} journalctl -fu hopper-upper
Restart upper           :  ssh nvidia@${JETSON_IP} sudo systemctl restart hopper-upper
Stop upper              :  ssh nvidia@${JETSON_IP} sudo systemctl stop hopper-upper
Fetch CSV logs to PC    :  rsync -a nvidia@${JETSON_IP}:hopper_upper/hopper_controller/logs/ /home/abc/Hopper/robot_runtime/upper_controller_pc/hopper_controller/logs/
Gamepad: X = enter PD mode,  A = props on,  B = full stop.
======================================================================
TXT
