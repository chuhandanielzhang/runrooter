#!/bin/bash
# Save the LAST upper session (CSV logs + camera recording) from the Jetson
# to this PC, stamped with Beijing time:
#
#   bash scripts/save_last.sh                    # -> logs/sessions/run_<BJ time>/
#   bash scripts/save_last.sh exp_outdoor        # -> logs/sessions/exp_outdoor_<BJ time>/
#   bash scripts/save_last.sh exp_outdoor 192.168.1.123
#
# "Last session" = everything created since the most recent start of
# hopper-upper (CSV) / hopper-perception (camera), whether or not the
# services are still running.  Files on the Jetson are NOT deleted.
set -e
unset LD_LIBRARY_PATH   # drop conda OpenSSL so system ssh works

LABEL="${1:-run}"
JETSON_IP="${2:-192.168.1.100}"
J="nvidia@${JETSON_IP}"
SSH="ssh -o ConnectTimeout=8 ${J}"
RLOGS="/home/nvidia/hopper_upper/hopper_controller/logs"
HERE="$(cd "$(dirname "$0")/.." && pwd)"

BJ_STAMP="$(TZ=Asia/Shanghai date +%Y%m%d_%H%M%S)"
DEST="${HERE}/upper_controller_pc/hopper_controller/logs/sessions/${LABEL}_${BJ_STAMP}BJ"
mkdir -p "${DEST}"

# --- last service start times (systemd keeps them after stop) ---
UP_START="$($SSH 'systemctl show hopper-upper -p ActiveEnterTimestamp --value' || true)"
PC_START="$($SSH 'systemctl show hopper-perception -p ActiveEnterTimestamp --value' || true)"
echo "last hopper-upper start:      ${UP_START:-unknown}"
echo "last hopper-perception start: ${PC_START:-unknown}"

# --- CSV logs since the last upper start ---
# Current recorder is hopper-lcm-csv -> logs/lcm_csv/lcm_*.csv.
# Legacy in-loop modee_*.csv is no longer written.
if [ -n "${UP_START}" ] && [ "${UP_START}" != "n/a" ]; then
  CSVS="$($SSH "find ${RLOGS}/lcm_csv -maxdepth 1 -name 'lcm_2*.csv' -newermt '${UP_START}'; find ${RLOGS} -maxdepth 1 -name 'modee_2*.csv' -newermt '${UP_START}'" || true)"
else
  CSVS="$($SSH "ls -t ${RLOGS}/lcm_csv/lcm_2*.csv ${RLOGS}/modee_2*.csv 2>/dev/null | head -1" || true)"
fi
if [ -n "${CSVS}" ]; then
  echo "${CSVS}" | while read -r f; do
    [ -n "$f" ] && rsync -a --info=name "${J}:${f}" "${DEST}/"
  done
else
  echo "WARNING: no session CSVs found on the Jetson"
fi

# --- camera runs since the last perception start ---
if [ -n "${PC_START}" ] && [ "${PC_START}" != "n/a" ]; then
  CAMS="$($SSH "find ${RLOGS}/cam -mindepth 1 -maxdepth 1 -type d -newermt '${PC_START}' 2>/dev/null" || true)"
else
  CAMS="$($SSH "ls -td ${RLOGS}/cam/*/ 2>/dev/null | head -1" || true)"
fi
if [ -n "${CAMS}" ]; then
  mkdir -p "${DEST}/cam"
  echo "${CAMS}" | while read -r d; do
    [ -n "$d" ] && rsync -a --info=name "${J}:${d%/}" "${DEST}/cam/"
  done
else
  echo "WARNING: no camera recording found on the Jetson"
fi

echo
echo "saved -> ${DEST}"
du -sh "${DEST}" 2>/dev/null
echo "local disk:"
df -h / | tail -1

# --- per-CSV summary: log id + hop count ---
echo
python3 - "${DEST}" <<'PY'
import glob, os, sys
import numpy as np
import pandas as pd

dest = sys.argv[1]
csvs = sorted(
    glob.glob(os.path.join(dest, "modee_2*.csv"))
    + glob.glob(os.path.join(dest, "lcm_2*.csv"))
)
if not csvs:
    print("no CSVs to summarize")
for p in csvs:
    try:
        df = pd.read_csv(p, low_memory=False, nrows=1)
        cols = set(df.columns)
        if {"t_s", "liftoff", "touchdown", "gait_mode"} <= cols:
            df = pd.read_csv(p, low_memory=False,
                             usecols=["t_s", "liftoff", "touchdown", "gait_mode"])
        elif {"wall_time_s", "wheel_en"} <= cols:
            df = pd.read_csv(
                p, low_memory=False,
                usecols=["wall_time_s", "wheel_en",
                         "wheel_w0", "wheel_w1", "wheel_w2"],
            )
            t = df.wall_time_s.to_numpy(float)
            dur = float(t[-1] - t[0]) if len(t) else 0.0
            en = (df.wheel_en.to_numpy(float) > 0.5)
            w = np.hypot(
                df.wheel_w0.to_numpy(float),
                df.wheel_w1.to_numpy(float),
                df.wheel_w2.to_numpy(float),
            )
            n_cmd = int(np.sum(en & (w > 0.05)))
            print(f"{os.path.basename(p)}: {dur:.0f}s, "
                  f"wheel_en {int(en.sum())} rows, "
                  f"|w|>0.05 {n_cmd} rows, "
                  f"w_max {float(np.nanmax(w)) if len(w) else 0:.2f} rad/s")
            continue
        else:
            print(f"{os.path.basename(p)}: {len(df)} header-only / unknown schema")
            continue
    except Exception as e:
        print(f"{os.path.basename(p)}: unreadable ({e})")
        continue
    t = df.t_s.to_numpy(float)
    lo = np.flatnonzero(df.liftoff.to_numpy(float) > 0.5)
    td = np.flatnonzero(df.touchdown.to_numpy(float) > 0.5)
    hops = real = 0
    for i in lo:
        hops += 1
        nxt = td[td > i]
        # a "real" hop clears the 0.12 s chatter window
        if len(nxt) and (t[int(nxt[0])] - t[i]) >= 0.12:
            real += 1
    dur = t[-1] - t[0] if len(t) else 0.0
    hop_s = df[df.gait_mode.astype(str) == "hopping"].t_s
    hop_dur = float(hop_s.max() - hop_s.min()) if len(hop_s) else 0.0
    print(f"{os.path.basename(p)}: {dur:.0f}s total, "
          f"hopping {hop_dur:.0f}s, liftoffs {hops} "
          f"({real} real hops >=0.12s flight)")
PY
