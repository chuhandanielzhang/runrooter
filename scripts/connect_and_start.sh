#!/bin/bash
# Full install + force bring-up from the PC (also enables Jetson boot autostart).
# Day-to-day after boot: use scripts/connect.sh / update_upper.sh / fetch_logs.sh.
#
#   bash scripts/connect_and_start.sh              # default IP .100
#   bash scripts/connect_and_start.sh 192.168.1.123
set -o pipefail
unset LD_LIBRARY_PATH

HERE="$(cd "$(dirname "$0")/.." && pwd)"
JETSON_IP="${1:-192.168.1.100}"
PC_IFACE="${PC_IFACE:-enp44s0}"
LCM_URL="udpm://239.255.76.67:7667?ttl=255"
SSH="ssh -o BatchMode=yes -o ConnectTimeout=6 -o StrictHostKeyChecking=accept-new nvidia@${JETSON_IP}"

say(){ printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
ok(){  printf '   \033[1;32mOK\033[0m  %s\n' "$*"; }
bad(){ printf '   \033[1;31mXX\033[0m  %s\n' "$*"; }

# ── 1. PC multicast ──
say "1/5  PC multicast route -> ${PC_IFACE}"
cur=$(ip route get 239.255.76.67 2>/dev/null | head -1)
if echo "$cur" | grep -q "dev ${PC_IFACE}"; then
  ok "route already via ${PC_IFACE}"
else
  echo "   route is: $cur"
  echo "   fixing (sudo password may be asked)..."
  sudo ip route replace 224.0.0.0/4 dev "${PC_IFACE}" && ok "route replaced" || bad "route fix failed (run manually)"
fi

# ── 2. wait SSH ──
say "2/5  waiting for Jetson SSH @ ${JETSON_IP} (up to 120s)"
deadline=$(( $(date +%s) + 120 )); online=0
while [ "$(date +%s)" -lt "$deadline" ]; do
  if $SSH 'echo ok' >/dev/null 2>&1; then online=1; break; fi
  printf '   ...not reachable yet, retrying\r'; sleep 4
done
if [ "$online" != 1 ]; then
  bad "Jetson never came online. Check: powered on? wired cable in? IP correct?"
  exit 1
fi
ok "SSH up  ($($SSH 'hostname; uptime -p' 2>/dev/null | paste -sd' '))"

# ── 3. sync code + install boot bring-up ──
say "3/5  syncing upper / bridges / services + enabling boot autostart"
UPPER_SRC="${HERE}/upper_controller_pc/"
rsync -a --delete \
  --exclude 'hopper_controller/logs' \
  --exclude '__pycache__' \
  "${UPPER_SRC}" "nvidia@${JETSON_IP}:/home/nvidia/hopper_upper/" \
  && ok "upper -> /home/nvidia/hopper_upper" || bad "upper rsync failed"

BRIDGE_SRC="${HERE}/imu_bridge_jetson/"
rsync -a --exclude '__pycache__' \
  "${BRIDGE_SRC}" "nvidia@${JETSON_IP}:/home/nvidia/Hopper_srbRL/pixhawk/" \
  && ok "bridges -> /home/nvidia/Hopper_srbRL/pixhawk" || bad "bridge rsync failed"

SVC_SRC="${HERE}/services"
rsync -a \
  "${SVC_SRC}/px4-bridge.service" "${SVC_SRC}/px4-dds-bridge.service" \
  "${SVC_SRC}/canable.service" "${SVC_SRC}/canable2.service" \
  "${SVC_SRC}/canable-watchdog.service" \
  "${SVC_SRC}/canable_watchdog.sh" "${SVC_SRC}/99-canable.rules" \
  "${SVC_SRC}/jetson-power.service" "${SVC_SRC}/hopper-driver.service" \
  "${SVC_SRC}/hopper-upper.service" "${SVC_SRC}/hopper-perception.service" \
  "${SVC_SRC}/hopper-bringup.service" "${SVC_SRC}/jetson_bringup.sh" \
  "${SVC_SRC}/hopper-gamepad-upper.service" "${SVC_SRC}/gamepad_upper_toggle.py" \
  "${SVC_SRC}/lcm_multicast_route.sh" \
  "nvidia@${JETSON_IP}:/tmp/hopper_svc/" \
  && $SSH 'sudo install -m644 /tmp/hopper_svc/*.service /etc/systemd/system/ \
    && sudo install -m755 /tmp/hopper_svc/jetson_bringup.sh /usr/local/bin/jetson_bringup.sh \
    && sudo install -m755 /tmp/hopper_svc/canable_watchdog.sh /usr/local/bin/canable_watchdog.sh \
    && sudo install -m755 /tmp/hopper_svc/gamepad_upper_toggle.py /usr/local/bin/gamepad_upper_toggle.py \
    && sudo install -m755 /tmp/hopper_svc/lcm_multicast_route.sh /usr/local/bin/lcm_multicast_route.sh \
    && sudo install -m644 /tmp/hopper_svc/99-canable.rules /etc/udev/rules.d/ \
    && sudo udevadm control --reload-rules && sudo udevadm trigger --subsystem-match=tty --action=add \
    && sudo systemctl daemon-reload \
    && sudo systemctl reenable canable.service canable2.service >/dev/null 2>&1 || true \
    && sudo systemctl enable canable-watchdog.service >/dev/null 2>&1 || true \
    && sudo systemctl enable jetson-power.service xrce-agent.service px4-dds-bridge.service >/dev/null 2>&1 || true \
    && sudo systemctl enable hopper-driver.service hopper-upper.service hopper-perception.service >/dev/null 2>&1 || true \
    && sudo systemctl enable hopper-bringup.service hopper-gamepad-upper.service >/dev/null 2>&1 || true \
    && sudo systemctl restart canable-watchdog.service >/dev/null 2>&1 || true \
    && sudo systemctl restart hopper-gamepad-upper.service >/dev/null 2>&1 || true' \
  && ok "services installed; hopper-bringup ENABLED for boot" || bad "service install failed"

# ── 4. force bring-up now (same script boot will run) ──
say "4/5  running jetson_bringup on Jetson"
$SSH 'sudo /usr/local/bin/jetson_bringup.sh' \
  && ok "bring-up finished" || bad "bring-up reported errors (see above)"

# ── 5. PC LCM check ──
say "5/5  PC end-to-end LCM check (3s)"
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

================ Jetson boots the stack by itself ================
Boot unit : hopper-bringup.service -> /usr/local/bin/jetson_bringup.sh
Daily PC (plug ethernet):
  connect      :  bash ${HERE}/scripts/connect.sh ${JETSON_IP}
  update upper :  bash ${HERE}/scripts/update_upper.sh ${JETSON_IP}
  fetch logs   :  bash ${HERE}/scripts/fetch_logs.sh [label] ${JETSON_IP}
Gamepad: SELECT/Back 1x=start upper, 2x=stop upper; X=PD, A=props on, B=full stop.
==================================================================
TXT
