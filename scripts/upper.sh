#!/bin/bash
# One-stop upper-controller ops over the wired link (outdoor use):
#
#   bash scripts/upper.sh start    [ip]   # start hopper-upper + perception
#   bash scripts/upper.sh stop     [ip]   # stop both
#   bash scripts/upper.sh restart  [ip]   # restart both
#   bash scripts/upper.sh status   [ip]   # service states + last log lines
#   bash scripts/upper.sh logs     [ip]   # follow hopper-upper journal (Ctrl-C exits)
#   bash scripts/upper.sh sync     [ip]   # push local code, restart if running
#   bash scripts/upper.sh fetch    [ip]   # pull CSV logs + camera recordings to PC
#
# Default ip: 192.168.1.100 (plug the ethernet cable, wait ~10 s for link).
set -e
unset LD_LIBRARY_PATH   # drop conda OpenSSL so system ssh works

CMD="${1:?usage: upper.sh start|stop|restart|status|logs|sync|fetch [ip]}"
JETSON_IP="${2:-192.168.1.100}"
J="nvidia@${JETSON_IP}"
SSH="ssh -o ConnectTimeout=8 ${J}"
HERE="$(cd "$(dirname "$0")/.." && pwd)"

case "$CMD" in
  start)
    $SSH 'sudo systemctl start hopper-upper.service hopper-perception.service'
    echo "hopper-upper + hopper-perception STARTED on ${JETSON_IP}"
    ;;
  stop)
    $SSH 'sudo systemctl stop hopper-upper.service hopper-perception.service'
    echo "hopper-upper + hopper-perception STOPPED on ${JETSON_IP}"
    ;;
  restart)
    $SSH 'sudo systemctl restart hopper-upper.service hopper-perception.service'
    echo "hopper-upper + hopper-perception RESTARTED on ${JETSON_IP}"
    ;;
  status)
    $SSH 'for s in hopper-upper hopper-perception; do
            printf "%-18s %s\n" "$s" "$(systemctl is-active $s.service)";
          done;
          echo "--- hopper-upper (last 15 lines) ---";
          journalctl -u hopper-upper -n 15 --no-pager'
    ;;
  logs)
    $SSH 'journalctl -fu hopper-upper --no-pager'
    ;;
  sync)
    bash "${HERE}/scripts/sync_upper.sh" "${JETSON_IP}"
    ;;
  fetch)
    DEST="${HERE}/upper_controller_pc/hopper_controller/logs/"
    rsync -a "${J}:/home/nvidia/hopper_upper/hopper_controller/logs/" "${DEST}"
    echo "Jetson logs + camera recordings -> ${DEST}"
    ;;
  *)
    echo "unknown command: ${CMD}" >&2
    exit 1
    ;;
esac
