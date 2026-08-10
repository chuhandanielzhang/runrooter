#!/usr/bin/env python3
"""Replay / export RGB-D recordings made by jetson_perception.py.

Recording layout (see CamRecorder in jetson_perception.py):
  <run>/segNNN/meta.json   intrinsics, fps, t0
  <run>/segNNN/color.avi   MJPG BGR
  <run>/segNNN/depth.bin   per frame [f64 t_wall][u32 len][zlib z16 mm]
  <run>/segNNN/ts.csv      frame_idx,t_wall

Usage:
  # interactive viewer (color | depth colormap side by side; q quits,
  # space pauses, arrows step while paused)
  python3 tools/replay_cam_record.py logs/cam/20260809_154210/seg000

  # export side-by-side mp4
  python3 tools/replay_cam_record.py <segdir> --export-mp4 out.mp4

  # dump per-frame PNGs (color_XXXXXX.png + depth_XXXXXX.png, 16-bit mm)
  python3 tools/replay_cam_record.py <segdir> --export-png outdir
"""
from __future__ import annotations

import argparse
import json
import struct
import sys
import zlib
from pathlib import Path

import cv2
import numpy as np


def read_depth_frames(seg: Path, w: int, h: int):
    """Yield (t_wall, depth_mm_uint16[h,w]) for every stored frame."""
    with open(seg / "depth.bin", "rb") as fp:
        while True:
            hdr = fp.read(12)
            if len(hdr) < 12:
                return
            t_wall, blob_len = struct.unpack("<dI", hdr)
            blob = fp.read(blob_len)
            if len(blob) < blob_len:
                return  # truncated tail (power cut mid-frame)
            d = np.frombuffer(
                zlib.decompress(blob), dtype=np.uint16
            ).reshape(h, w)
            yield float(t_wall), d


def depth_vis(d_mm: np.ndarray, max_m: float = 4.0) -> np.ndarray:
    d = d_mm.astype(np.float32) / 1000.0
    d = np.clip(d / max_m, 0.0, 1.0)
    vis = cv2.applyColorMap((d * 255).astype(np.uint8), cv2.COLORMAP_JET)
    vis[d_mm == 0] = 0
    return vis


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("segdir", type=Path)
    ap.add_argument("--export-mp4", type=Path, default=None)
    ap.add_argument("--export-png", type=Path, default=None)
    ap.add_argument("--depth-max-m", type=float, default=4.0)
    args = ap.parse_args()

    seg = args.segdir
    meta = json.loads((seg / "meta.json").read_text())
    w, h, fps = int(meta["w"]), int(meta["h"]), int(meta.get("fps", 15))
    print(f"[replay] {seg}  {w}x{h} @ {fps} fps  t0={meta.get('t0_wall')}")

    cap = cv2.VideoCapture(str(seg / "color.avi"))
    depth_iter = read_depth_frames(seg, w, h)

    vw = None
    if args.export_mp4 is not None:
        vw = cv2.VideoWriter(
            str(args.export_mp4), cv2.VideoWriter_fourcc(*"mp4v"),
            float(fps), (w * 2, h),
        )
    if args.export_png is not None:
        args.export_png.mkdir(parents=True, exist_ok=True)

    interactive = vw is None and args.export_png is None
    paused = False
    idx = 0
    while True:
        if not paused or not interactive:
            ok, bgr = cap.read()
            nxt = next(depth_iter, None)
            if not ok or nxt is None:
                break
            t_wall, d_mm = nxt
            side = np.hstack([bgr, depth_vis(d_mm, args.depth_max_m)])
            cv2.putText(
                side, f"#{idx}  t={t_wall:.3f}", (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2,
            )
            if vw is not None:
                vw.write(side)
            if args.export_png is not None:
                cv2.imwrite(
                    str(args.export_png / f"color_{idx:06d}.png"), bgr
                )
                cv2.imwrite(
                    str(args.export_png / f"depth_{idx:06d}.png"), d_mm
                )
            idx += 1
        if interactive:
            cv2.imshow("replay (q quit, space pause)", side)
            k = cv2.waitKey(0 if paused else max(1, 1000 // fps)) & 0xFF
            if k == ord("q"):
                break
            if k == ord(" "):
                paused = not paused

    cap.release()
    if vw is not None:
        vw.release()
        print(f"[replay] wrote {args.export_mp4} ({idx} frames)")
    if args.export_png is not None:
        print(f"[replay] wrote {idx} frame pairs -> {args.export_png}")
    if interactive:
        cv2.destroyAllWindows()
    print(f"[replay] done: {idx} frames")
    return 0


if __name__ == "__main__":
    sys.exit(main())
