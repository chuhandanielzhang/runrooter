#!/bin/bash
# PC: wait for Jetson on the wired link, fix PC multicast, show stack status.
# Does NOT sync code or restart services (Jetson boots via hopper-bringup).
#
#   bash scripts/connect.sh              # default 192.168.1.100
#   bash scripts/connect.sh 192.168.1.123
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

say "1/3  PC multicast route -> ${PC_IFACE}"
cur=$(ip route get 239.255.76.67 2>/dev/null | head -1)
if echo "$cur" | grep -q "dev ${PC_IFACE}"; then
  ok "route already via ${PC_IFACE}"
else
  echo "   route is: $cur"
  echo "   fixing (sudo password may be asked)..."
  sudo ip route replace 224.0.0.0/4 dev "${PC_IFACE}" && ok "route replaced" || bad "route fix failed"
fi

say "2/3  waiting for Jetson SSH @ ${JETSON_IP} (up to 120s)"
deadline=$(( $(date +%s) + 120 )); online=0
while [ "$(date +%s)" -lt "$deadline" ]; do
  if $SSH 'echo ok' >/dev/null 2>&1; then online=1; break; fi
  printf '   ...not reachable yet, retrying\r'; sleep 4
done
if [ "$online" != 1 ]; then
  bad "Jetson never came online. Powered on? cable in? IP correct?"
  exit 1
fi
ok "SSH up  ($($SSH 'hostname; uptime -p' 2>/dev/null | paste -sd' '))"

say "3/3  Jetson stack + PC LCM"
$SSH 'for s in hopper-bringup canable hopper-driver hopper-upper hopper-perception xrce-agent px4-dds-bridge; do
        printf "%-18s %s\n" "$s" "$(systemctl is-active $s.service 2>/dev/null || echo n/a)";
      done'

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
    print("   XX  no LCM on PC -> sudo ip route replace 224.0.0.0/4 dev ${PC_IFACE}")
else:
    for k,v in sorted(c.items()):
        print(f"   OK  {k:22s} ~{v/3:.0f} Hz")
PY

cat <<TXT

================ wired link ready ================
repo: ${HERE}
update upper :  bash ${HERE}/scripts/update_upper.sh ${JETSON_IP}
fetch logs   :  bash ${HERE}/scripts/fetch_logs.sh [label] ${JETSON_IP}
force bringup:  bash ${HERE}/scripts/connect_and_start.sh ${JETSON_IP}
==================================================
TXT
