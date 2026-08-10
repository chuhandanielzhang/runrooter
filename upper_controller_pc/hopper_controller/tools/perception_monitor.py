#!/usr/bin/env python3
"""PC perception monitor -- display ONLY, no vision algorithms.

Companion viewer for tools/jetson_perception.py (which now does all
AprilTag / box / plane detection on the Jetson). The PC just renders
three UDP streams:

  :5560  annotated JPEG preview from the Jetson perception node
  :5559  perception metrics JSON (tag/box detection state)
  :5557  controller dashboard JSON from hopper-upper (gait, leg, stages)

The right-hand panel switches with the controller gait mode:
  MOBILE        tag search state, camera distance, manip reachability,
                box face state, auto-approach status
  MANIPULATION  button stage, foot cmd/actual, tracking error
  PUSH          box source/age, contact offset e, push speed, leg error
  other         raw gait/leg summary

Keys: q/ESC quit, s save snapshot to logs/figs/.

Usage:
  pc$ python3 tools/perception_monitor.py
  pc$ python3 tools/perception_monitor.py --headless --frames 50
"""
from __future__ import annotations

import argparse
import json
import socket
import time
from pathlib import Path

import cv2
import numpy as np

PANEL_W = 460
GREEN = (80, 220, 80)
YELLOW = (60, 210, 240)
RED = (70, 70, 235)
GRAY = (160, 160, 160)
WHITE = (235, 235, 235)


def _bind_udp(port: int) -> socket.socket | None:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        s.bind(("", int(port)))
        s.setblocking(False)
        return s
    except OSError as e:
        print(f"[monitor] UDP :{port} unavailable ({e})")
        return None


def _drain_json(sock: socket.socket | None) -> dict | None:
    """Return the newest decodable JSON datagram (or None)."""
    if sock is None:
        return None
    latest = None
    while True:
        try:
            raw, _peer = sock.recvfrom(65535)
        except (BlockingIOError, OSError):
            break
        try:
            latest = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            pass
    return latest


def _drain_jpeg(sock: socket.socket | None) -> np.ndarray | None:
    if sock is None:
        return None
    latest = None
    while True:
        try:
            raw, _peer = sock.recvfrom(65535)
        except (BlockingIOError, OSError):
            break
        latest = raw
    if latest is None:
        return None
    img = cv2.imdecode(np.frombuffer(latest, np.uint8), cv2.IMREAD_COLOR)
    return img


def _rot_cw(img: np.ndarray, deg: int) -> np.ndarray:
    k = (int(deg) // 90) % 4
    return img if k == 0 else np.rot90(img, k=-k)


class Panel:
    """Fixed-width text panel rendered next to the camera preview."""

    def __init__(self, height: int):
        self.img = np.full((height, PANEL_W, 3), 24, np.uint8)
        self.y = 34

    def line(self, text: str, color=WHITE, scale: float = 0.62):
        cv2.putText(
            self.img, text, (14, self.y), cv2.FONT_HERSHEY_SIMPLEX,
            scale, color, 1, cv2.LINE_AA,
        )
        self.y += int(30 * scale + 10)

    def header(self, text: str, color=WHITE):
        self.line(text, color=color, scale=0.85)
        cv2.line(
            self.img, (14, self.y - 12), (PANEL_W - 14, self.y - 12),
            (80, 80, 80), 1,
        )
        self.y += 4

    def gap(self, px: int = 10):
        self.y += px


def _fmt_vec(v, unit: str = "m") -> str:
    if v is None:
        return "n/a"
    try:
        x, y, z = (float(a) for a in v)
        return f"{x:+.3f}/{y:+.3f}/{z:+.3f} {unit}"
    except Exception:
        return "n/a"


def render_panel(
    height: int,
    percep: dict | None,
    percep_age: float,
    ctrl: dict | None,
    ctrl_age: float,
) -> np.ndarray:
    p = Panel(height)
    gait = str((ctrl or {}).get("gait_mode", "?")).upper()

    # --- link health -----------------------------------------------------
    if percep is None or percep_age > 2.0:
        p.header("PERCEPTION: OFFLINE", RED)
        p.line("start hopper-perception on the Jetson", GRAY)
    else:
        hz = percep.get("hz", "?")
        p.header(f"PERCEPTION: {hz} Hz ({percep.get('backend', '?')})",
                 GREEN)
    if ctrl is None or ctrl_age > 1.0:
        p.line("controller: OFFLINE (hopper-upper?)", RED)
    else:
        p.line(f"controller: {gait} (age {ctrl_age:.1f}s)", GREEN)
    p.gap()

    btn = (percep or {}).get("button") or {}
    box = (percep or {}).get("box")

    # --- mode-specific view ------------------------------------------------
    if gait == "MOBILE":
        approach = bool((ctrl or {}).get("approach_active", False))
        p.header("MOBILE " + ("- AUTO APPROACH" if approach else ""),
                 YELLOW if approach else WHITE)
        tag_state = str((ctrl or {}).get("mobile_tag_state", "?")).upper()
        col = {"READY": GREEN, "SEARCHING": GRAY}.get(tag_state, YELLOW)
        p.line(f"tag: {tag_state}", col)
        r_min = float((ctrl or {}).get("mobile_ready_r_min_m", 0.50))
        r_max = float((ctrl or {}).get("mobile_ready_r_max_m", 0.53))
        d = (ctrl or {}).get("mobile_tag_cam_z_m")
        if d is None:
            d = btn.get("tag_cam_z_m")
        # Goal vs now vs 差值 (radial band + workspace clip).
        p.line(f"goal: |face|_L in [{r_min:.2f}, {r_max:.2f}] m", WHITE)
        if d is not None:
            d = float(d)
            r_err = (ctrl or {}).get("mobile_tag_r_err_m")
            if r_err is None:
                if d > r_max:
                    r_err = d - r_max
                elif d < r_min:
                    r_err = d - r_min
                else:
                    r_err = 0.0
            r_err = float(r_err)
            if abs(r_err) < 1e-4:
                p.line(f"now:  {d:.2f} m   Δr = 0.00 m (in band)", GREEN)
            elif r_err > 0:
                p.line(f"now:  {d:.2f} m   Δr = +{r_err*100:.1f} cm (too far)",
                       YELLOW)
            else:
                p.line(f"now:  {d:.2f} m   Δr = {r_err*100:.1f} cm (too close)",
                       YELLOW)
        else:
            p.line("now:  n/a   Δr = n/a", GRAY)
        reach = (ctrl or {}).get("mobile_tag_reach_error_m")
        if reach is not None:
            rc = float(reach) * 100.0
            p.line(f"clip: {rc:.1f} cm  (workspace 差值, info)",
                   YELLOW if rc > 2.0 else GREEN)
        else:
            p.line("clip: n/a", GRAY)
        if btn.get("valid") and btn.get("foot_face_L") is not None:
            p.line(f"BTN L {_fmt_vec(btn.get('foot_face_L'))}", WHITE)
            p.line("preview: magenta BTN / cyan PRE", GRAY, 0.5)
        if tag_state == "READY":
            p.line("manip: OK -> LT / auto-approach", GREEN)
        elif d is not None:
            p.line(f"manip: FAR -> drive until Δr=0 "
                   f"([{r_min:.2f},{r_max:.2f}])", YELLOW)
        else:
            p.line("manip: no target", GRAY)
        if approach:
            waiting = bool((ctrl or {}).get("approach_waiting", False))
            p.line("approach: " + ("WAIT (tag lost)" if waiting
                                   else "SERVO -> tag"),
                   RED if waiting else GREEN)
        p.gap()
        if box:
            p.line(f"box: {str(box.get('source', '?')).upper()} "
                   f"w={box.get('width_m', 0):.2f} m", GREEN)
            p.line(f"  center {_fmt_vec(box.get('center_L'))}", GRAY)
            # Working-pose errors from the controller (parallelism check).
            d_err = (ctrl or {}).get("box_dist_err_m")
            y_err = (ctrl or {}).get("box_yaw_err_deg")
            if d_err is not None or y_err is not None:
                d_s = (f"{float(d_err) * 100:+.1f}cm"
                       if d_err is not None else "n/a")
                y_s = (f"{float(y_err):+.1f}deg"
                       if y_err is not None else "n/a")
                p.line(f"  dist err {d_s}  yaw err {y_s}", WHITE)
            wd = (ctrl or {}).get("push_work_dist_m")
            if wd is not None:
                p.line(f"  work dist {float(wd):.2f} m (latched)", GRAY)
            if bool((ctrl or {}).get("box_ready", False)):
                p.line("  BOX READY -> LT = PUSH", GREEN)
        else:
            p.line("box: none", GRAY)
    elif gait == "MANIPULATION":
        p.header("MANIPULATION", YELLOW)
        p.line(f"stage: {(ctrl or {}).get('button_stage', '?')}", GREEN)
        err = (ctrl or {}).get("manip_err_m")
        p.line(f"err: {float(err)*1000:.1f} mm" if err is not None
               else "err: n/a", WHITE)
        p.line(f"cmd    {_fmt_vec((ctrl or {}).get('foot_cmd_L'))}", WHITE)
        p.line(f"actual {_fmt_vec((ctrl or {}).get('foot_actual_L'))}", GRAY)
        tau = (ctrl or {}).get("tau_cmd")
        p.line(f"tau    {_fmt_vec(tau, 'Nm')}", GRAY)
    elif gait == "PUSH":
        p.header("PUSH (semi-auto box)", YELLOW)
        cbox = (ctrl or {}).get("box")
        if cbox:
            p.line(f"box: {str(cbox.get('source', '?')).upper()} "
                   f"age={cbox.get('age_s', '?')}s", GREEN)
            p.line(f"  center {_fmt_vec(cbox.get('center_L'))}", GRAY)
            p.line(f"  normal {_fmt_vec(cbox.get('normal_in_L'), '')}", GRAY)
        else:
            p.line("box: LOST (wheels stopped)", RED)
        state = str((ctrl or {}).get("push_state") or "TRACK")
        st_col = {"TRACK": GREEN, "WAIT": YELLOW}.get(state, RED)
        p.line(f"state: {state}", st_col)
        f_n = (ctrl or {}).get("push_f_meas_n")
        p.line(f"push force: {float(f_n):+.1f} N" if f_n is not None
               else "push force: n/a", WHITE)
        e = (ctrl or {}).get("push_e_m")
        v = (ctrl or {}).get("push_v_mps")
        p.line(f"contact e: {float(e)*100:+.1f} cm" if e is not None
               else "contact e: n/a", WHITE)
        p.line(f"push v: {float(v):+.2f} m/s" if v is not None
               else "push v: n/a", WHITE)
        err = (ctrl or {}).get("manip_err_m")
        p.line(f"leg err: {float(err)*1000:.1f} mm" if err is not None
               else "leg err: n/a", WHITE)
        wd = (ctrl or {}).get("push_work_dist_m")
        if wd is not None:
            p.line(f"work dist: {float(wd):.2f} m (latched)", GRAY)
        p.line("LEFT stick: up=push fwd, x=steer box", GRAY, 0.5)
        p.line("LT exits to MOBILE", GRAY, 0.5)
    else:
        p.header(gait if gait != "?" else "WAITING", WHITE)
        p.line(f"tag ids seen: {(percep or {}).get('tag_ids_seen', [])}",
               GRAY)
        p.line(f"button valid: {btn.get('valid', False)}", GRAY)
        p.line(f"box: {'yes' if box else 'no'}", GRAY)
    return p.img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics-port", type=int, default=5559)
    ap.add_argument("--preview-port", type=int, default=5560)
    ap.add_argument("--status-port", type=int, default=5557)
    ap.add_argument("--rotate", type=int, default=None,
                    choices=(0, 90, 180, 270),
                    help="override the rotation hint from the Jetson")
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--frames", type=int, default=0, help="stop after N")
    args = ap.parse_args()

    s_metrics = _bind_udp(args.metrics_port)
    s_preview = _bind_udp(args.preview_port)
    s_status = _bind_udp(args.status_port)
    print(
        f"[monitor] listening: metrics :{args.metrics_port}, "
        f"preview :{args.preview_port}, controller :{args.status_port}"
    )

    percep = None
    percep_rx = float("-inf")
    ctrl = None
    ctrl_rx = float("-inf")
    frame = None
    n = 0
    win = "Hopper Perception Monitor"
    if not args.headless:
        cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)

    try:
        while True:
            m = _drain_json(s_metrics)
            if m is not None:
                percep = m
                percep_rx = time.monotonic()
            c = _drain_json(s_status)
            if c is not None:
                ctrl = c
                ctrl_rx = time.monotonic()
            img = _drain_jpeg(s_preview)
            if img is not None:
                frame = img

            rot = (args.rotate if args.rotate is not None
                   else int((percep or {}).get("rotate_cw", 90)))
            if frame is not None:
                view = np.ascontiguousarray(_rot_cw(frame, rot))
            else:
                view = np.full((480, 300, 3), 16, np.uint8)
                cv2.putText(
                    view, "no preview", (40, 240),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, GRAY, 2, cv2.LINE_AA,
                )
            panel = render_panel(
                view.shape[0],
                percep, time.monotonic() - percep_rx,
                ctrl, time.monotonic() - ctrl_rx,
            )
            canvas = np.hstack([view, panel])

            n += 1
            if args.headless:
                if args.frames and n >= args.frames:
                    break
                time.sleep(0.05)
                continue
            cv2.imshow(win, canvas)
            key = cv2.waitKey(50) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                out = (Path(__file__).resolve().parent.parent
                       / "logs" / "figs")
                out.mkdir(parents=True, exist_ok=True)
                path = out / (
                    "perception_monitor_"
                    + time.strftime("%Y%m%d_%H%M%S") + ".png"
                )
                cv2.imwrite(str(path), canvas)
                print(f"[monitor] snapshot -> {path}")
            if args.frames and n >= args.frames:
                break
    except KeyboardInterrupt:
        pass
    finally:
        for s in (s_metrics, s_preview, s_status):
            if s is not None:
                s.close()
        if not args.headless:
            cv2.destroyAllWindows()
    print(f"[monitor] {n} frames displayed")


if __name__ == "__main__":
    main()
