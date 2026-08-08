#!/usr/bin/env python3
"""Live wall-AprilTag -> button target for MANIPULATION.

Connects to Jetson d435_net_server, detects tag36h11, draws the button
offset (right 16.5 cm, down 2.6 cm, protrude 5 cm) and press point
(1 cm into wall), and writes a setpoint JSON that ModeE MANIPULATION
can follow.

Usage:
  # ensure Jetson camera server is up, then:
  python3 tools/button_apriltag_live.py --net 192.168.1.100
  python3 tools/button_apriltag_live.py --net 192.168.1.100 --tag-id 1 --tag-size 0.09

Keys:
  q / Esc  quit
  s        freeze current setpoint (stop updating JSON)
  c        continue updating setpoint
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from button_apriltag_geometry import (  # noqa: E402
    DEFAULT_DOWN_M,
    DEFAULT_PRESS_M,
    DEFAULT_PROTRUDE_M,
    DEFAULT_RIGHT_M,
    DEFAULT_WALL_TAG_ID,
    DEFAULT_WALL_TAG_SIZE_M,
    button_targets_from_detection,
    load_T_L_C,
)
from terrain_gate_live import NetSource  # noqa: E402

SETPOINT_DEFAULT = ROOT / "logs" / "button_setpoint.json"


def _detect(gray, detector, tag_size_m: float):
    return detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=None,  # filled by caller via camera_params kw below
        tag_size=tag_size_m,
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--net", default="192.168.1.100")
    ap.add_argument("--port", type=int, default=5556)
    ap.add_argument("--tag-id", type=int, default=DEFAULT_WALL_TAG_ID)
    ap.add_argument("--tag-size", type=float, default=DEFAULT_WALL_TAG_SIZE_M,
                    help="black detection square side (m)")
    ap.add_argument("--right-m", type=float, default=DEFAULT_RIGHT_M)
    ap.add_argument("--down-m", type=float, default=DEFAULT_DOWN_M,
                    help="button center below tag center (tag +Y), m")
    ap.add_argument("--protrude-m", type=float, default=DEFAULT_PROTRUDE_M)
    ap.add_argument("--press-m", type=float, default=DEFAULT_PRESS_M)
    ap.add_argument("--calib", type=Path, default=None)
    ap.add_argument("--setpoint", type=Path, default=SETPOINT_DEFAULT)
    ap.add_argument("--rotate", type=int, default=0,
                    help="CW display rotation degrees (0/90/180/270)")
    args = ap.parse_args()

    from pupil_apriltags import Detector

    T_L_C = load_T_L_C(args.calib)
    src = NetSource(args.net, port=int(args.port))
    cam_params = (src.fx, src.fy, src.cx, src.cy)
    det = Detector(families="tag36h11", nthreads=2, quad_decimate=1.0)

    args.setpoint.parent.mkdir(parents=True, exist_ok=True)
    freeze = False
    last_write = 0.0
    print(
        f"[button] wall tag id={args.tag_id} size={args.tag_size:.3f} m | "
        f"right={args.right_m*100:.1f} cm down={args.down_m*100:.1f} cm "
        f"protrude={args.protrude_m*100:.1f} cm press={args.press_m*100:.1f} cm"
    )
    print(f"[button] setpoint -> {args.setpoint}")

    while True:
        depth, rgb = src.frame()
        if rgb is None:
            continue
        bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        tags = det.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=cam_params,
            tag_size=float(args.tag_size),
        )
        chosen = None
        for t in tags:
            if int(t.tag_id) == int(args.tag_id):
                chosen = t
                break

        vis = bgr.copy()
        info_lines = []
        if chosen is not None:
            R = np.asarray(chosen.pose_R, dtype=float).reshape(3, 3)
            t = np.asarray(chosen.pose_t, dtype=float).reshape(3)
            tg = button_targets_from_detection(
                pose_R=R,
                pose_t=t,
                T_L_C=T_L_C,
                right_m=float(args.right_m),
                down_m=float(args.down_m),
                protrude_m=float(args.protrude_m),
                press_m=float(args.press_m),
            )
            # Draw tag box
            pts = chosen.corners.astype(int)
            cv2.polylines(vis, [pts], True, (0, 255, 255), 2)
            cxy = chosen.center.astype(int)
            cv2.circle(vis, tuple(cxy), 4, (0, 255, 255), -1)
            cv2.putText(
                vis, f"id={chosen.tag_id}",
                (cxy[0] + 6, cxy[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
            )

            # Project face/press into image for overlay
            def project(p_c):
                z = float(p_c[2])
                if z < 1e-4:
                    return None
                u = int(src.fx * float(p_c[0]) / z + src.cx)
                v = int(src.fy * float(p_c[1]) / z + src.cy)
                return u, v

            p_face_i = project(tg["camera"]["face"])
            p_press_i = project(tg["camera"]["press"])
            p_pre_i = project(tg["camera"]["pre"])
            if p_face_i and p_press_i:
                cv2.circle(vis, p_face_i, 7, (0, 165, 255), -1)
                cv2.circle(vis, p_press_i, 6, (0, 0, 255), -1)
                cv2.line(vis, p_face_i, p_press_i, (0, 0, 255), 2)
                cv2.putText(
                    vis, "FACE", (p_face_i[0] + 8, p_face_i[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 165, 255), 1,
                )
                cv2.putText(
                    vis, "PRESS-1cm", (p_press_i[0] + 8, p_press_i[1]),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1,
                )
            if p_pre_i:
                cv2.circle(vis, p_pre_i, 5, (255, 255, 0), -1)

            face_L = tg["leg"]["face"]
            press_L = tg["leg"]["press"]
            pre_L = tg["leg"]["pre"]
            n_in = tg["wall_normal_in_L"]
            n_out = tg["wall_normal_out_L"]
            info_lines = [
                f"tag id={chosen.tag_id} size={args.tag_size*1000:.0f}mm  t_cam={t[2]:.3f} m",
                f"face_L  [{face_L[0]:+.3f},{face_L[1]:+.3f},{face_L[2]:+.3f}] m",
                f"press_L [{press_L[0]:+.3f},{press_L[1]:+.3f},{press_L[2]:+.3f}] m",
                f"n_in_L  [{n_in[0]:+.2f},{n_in[1]:+.2f},{n_in[2]:+.2f}] (press dir)",
                "freeze" if freeze else "LIVE setpoint",
            ]
            now = time.time()
            if (not freeze) and (now - last_write) > 0.05:
                payload = {
                    "t_wall": now,
                    "tag_id": int(chosen.tag_id),
                    "tag_size_m": float(args.tag_size),
                    "right_m": float(args.right_m),
                    "down_m": float(args.down_m),
                    "protrude_m": float(args.protrude_m),
                    "press_m": float(args.press_m),
                    "foot_pre_L": [float(x) for x in pre_L],
                    "foot_face_L": [float(x) for x in face_L],
                    "foot_press_L": [float(x) for x in press_L],
                    "wall_normal_in_L": [float(x) for x in n_in],
                    "wall_normal_out_L": [float(x) for x in n_out],
                    "press_along_wall_normal": True,
                    "valid": True,
                }
                tmp = args.setpoint.with_suffix(".json.tmp")
                tmp.write_text(json.dumps(payload, indent=2))
                tmp.replace(args.setpoint)
                last_write = now
        else:
            info_lines = ["NO TAG", f"looking for id={args.tag_id}"]
            # Mark invalid but keep last file contents.

        y0 = 24
        for line in info_lines:
            cv2.putText(
                vis, line, (10, y0), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (255, 255, 255), 2,
            )
            y0 += 22

        if int(args.rotate) in (90, 180, 270):
            rot = {
                90: cv2.ROTATE_90_CLOCKWISE,
                180: cv2.ROTATE_180,
                270: cv2.ROTATE_90_COUNTERCLOCKWISE,
            }[int(args.rotate)]
            vis = cv2.rotate(vis, rot)

        cv2.imshow("button_apriltag_live", vis)
        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q")):
            break
        if key == ord("s"):
            freeze = True
            print("[button] setpoint FROZEN")
        if key == ord("c"):
            freeze = False
            print("[button] setpoint LIVE")

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
