#!/usr/bin/env python3
"""Watch gamepad SELECT (remote Back): 1x start hopper-upper, 2x stop it.

Runs independently of hopper-upper so a single press can bring the controller
up when it is offline. Double-click window defaults to 450 ms.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time

import lcm

# Jetson path after rsync of upper_controller_pc/
_LCM_TYPES = "/home/nvidia/hopper_upper/hopper_lcm_types/lcm_types"
if _LCM_TYPES not in sys.path and os.path.isdir(_LCM_TYPES):
    sys.path.insert(0, _LCM_TYPES)

from python.gamepad_lcmt import gamepad_lcmt  # type: ignore

LCM_URL = os.environ.get("HOPPER_LCM_URL", "udpm://239.255.76.67:7667?ttl=255")
UPPER_UNIT = os.environ.get("HOPPER_UPPER_UNIT", "hopper-upper.service")
DOUBLE_S = float(os.environ.get("HOPPER_SELECT_DOUBLE_S", "0.45"))
DEBOUNCE_S = float(os.environ.get("HOPPER_SELECT_DEBOUNCE_S", "0.05"))


def _systemctl(*args: str) -> int:
    # Service runs as nvidia; systemctl mutate needs sudo (NOPASSWD on Jetson).
    cmd = ["sudo", "-n", "systemctl", *args]
    try:
        r = subprocess.run(cmd, check=False, capture_output=True, text=True)
    except OSError as exc:
        print(f"[select] systemctl failed: {exc}", flush=True)
        return 1
    if r.returncode != 0 and r.stderr:
        print(f"[select] {' '.join(cmd)} -> {r.stderr.strip()}", flush=True)
    return int(r.returncode)


def _is_active() -> bool:
    return _systemctl("is-active", "--quiet", UPPER_UNIT) == 0


def _start_upper() -> None:
    if _is_active():
        print(f"[select] 1x SELECT: {UPPER_UNIT} already active", flush=True)
        return
    print(f"[select] 1x SELECT: starting {UPPER_UNIT}", flush=True)
    _systemctl("start", UPPER_UNIT)


def _stop_upper() -> None:
    if not _is_active():
        print(f"[select] 2x SELECT: {UPPER_UNIT} already stopped", flush=True)
        return
    print(f"[select] 2x SELECT: stopping {UPPER_UNIT}", flush=True)
    _systemctl("stop", UPPER_UNIT)


def _ensure_local_multicast() -> None:
    """Pin LCM to lo once so unplugging eth cannot starve hopper_cmd."""
    script = "/usr/local/bin/lcm_multicast_route.sh"
    if not os.path.isfile(script):
        return
    try:
        subprocess.run(
            ["sudo", "-n", script],
            check=False,
            capture_output=True,
            text=True,
        )
    except OSError:
        pass


def main() -> int:
    _ensure_local_multicast()
    lc = lcm.LCM(LCM_URL)
    last_sel = False
    last_edge_t = 0.0
    pending_single = False
    pending_t0 = 0.0

    def on_gamepad(_channel: str, data: bytes) -> None:
        nonlocal last_sel, last_edge_t, pending_single, pending_t0
        msg = gamepad_lcmt.decode(data)
        sel = bool(getattr(msg, "select", 0))
        now = time.monotonic()
        if sel and not last_sel:
            if (now - last_edge_t) < DEBOUNCE_S:
                last_sel = sel
                return
            last_edge_t = now
            if pending_single and (now - pending_t0) <= DOUBLE_S:
                pending_single = False
                _stop_upper()
            else:
                pending_single = True
                pending_t0 = now
        last_sel = sel

    lc.subscribe("gamepad_lcmt", on_gamepad)
    print(
        f"[select] watching gamepad SELECT: 1x start / 2x stop {UPPER_UNIT} "
        f"(double<{DOUBLE_S:.2f}s) url={LCM_URL} (LCM via lo, no cable)",
        flush=True,
    )
    while True:
        lc.handle_timeout(50)
        if pending_single and (time.monotonic() - pending_t0) > DOUBLE_S:
            pending_single = False
            _start_upper()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        raise SystemExit(0)
