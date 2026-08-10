#!/bin/bash
# Sync ONLY the upper-layer code (PC -> Jetson). Prefer scripts/update_upper.sh.
#
#   bash scripts/sync_upper.sh              # default IP .100
#   bash scripts/sync_upper.sh 192.168.1.123
set -e
unset LD_LIBRARY_PATH

HERE="$(cd "$(dirname "$0")/.." && pwd)"
exec bash "${HERE}/scripts/update_upper.sh" "${1:-192.168.1.100}"
