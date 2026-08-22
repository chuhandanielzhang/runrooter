#!/usr/bin/env python3
"""Jetson-local perception node: D435 + AprilTag button/box detection.

Runs ON the Jetson (camera host) next to hopper-upper. Replaces the old
PC-side detection round-trip (d435_net_server -> PC terrain_gate_live ->
UDP back), which added network latency and caused DETECTED/LOST flapping
in the controller. Everything below happens on-device; the PC only
receives a monitoring stream (tools/perception_monitor.py).

Outputs
  UDP :5558 (loopback, JSON)  button setpoint consumed unchanged by
                              ModeELCMController._read_button_setpoint,
                              plus a new "box" sub-dict (two-tag box face
                              or vertical-plane fallback).
  UDP :5559 (broadcast, JSON) monitor metrics for the PC viewer.
  UDP :5560 (broadcast, JPEG) annotated preview frame (single datagram).
  logs/button_setpoint.json   low-rate file fallback (same schema).

Detection
  - Button tag (id=1, 90 mm): identical geometry chain as the retired PC
    dashboard (pupil_apriltags pose -> button_apriltag_geometry -> T_L_C).
  - Box face: two tags glued symmetric about the box-face center
    (ids 2/3, 60 mm, 120 mm center-to-center;
    print box_apriltags_id2_id3_60mm_a4.pdf). Yaw comes from the
    tag-center BASELINE, not from single-tag orientation (avoids the
    planar pose ambiguity flip). One visible tag still yields the
    center through the known offset along its own X axis.
  - Plane fallback: when no box tag is visible, a low-rate RANSAC looks
    for a large vertical plane (ground plane is fitted and removed
    first).

Deps on the Jetson: pyrealsense2, numpy, cv2, pupil_apriltags
(pip install pupil-apriltags). matplotlib is NOT needed.

Usage:
  jetson$ python3 tools/jetson_perception.py
  jetson$ python3 tools/jetson_perception.py --box-tag-ids 2 3 \
              --box-tag-size 0.060 --box-tag-spacing 0.120
"""
from __future__ import annotations

import argparse
import json
import math
import socket
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modee.terrain_gate import (  # noqa: E402
    depth_to_ground_points, fit_dominant_plane,
)
from button_apriltag_geometry import (  # noqa: E402
    DEFAULT_DOWN_M, DEFAULT_RIGHT_M,
    button_targets_from_detection, load_T_L_C,
)

# Same fallback ladder as d435_net_server.py (USB2 cannot do 848x480@15).
PROFILES = [(848, 480, 15), (640, 480, 15), (640, 480, 6)]


class D435Local:
    """Local RealSense capture, color aligned to depth (depth intrinsics)."""

    def __init__(self):
        import pyrealsense2 as rs
        self.rs = rs
        self.pipe = rs.pipeline()
        prof = None
        for W, H, FPS in PROFILES:
            cfg = rs.config()
            cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, FPS)
            cfg.enable_stream(rs.stream.color, W, H, rs.format.rgb8, FPS)
            try:
                prof = self.pipe.start(cfg)
                self.fps = int(FPS)
                print(f"[percep] D435 {W}x{H} @ {FPS} fps", flush=True)
                break
            except RuntimeError as e:
                print(f"[percep] {W}x{H}@{FPS} unavailable ({e})", flush=True)
        if prof is None:
            raise SystemExit("no workable depth+color profile")
        self.align = rs.align(rs.stream.depth)
        # Motion blur is the enemy (2026-08-13 甩尾太糊丢视野): AprilTag
        # decode dies at ~1 tag cell (~10 px) of smear and PUSH tail swings
        # sweep ~600 px/s, yet THIS unit's firmware does NOT support
        # auto_exposure_limit -- AE was free to run up to 66 ms indoors.
        #   1) fw has the AE limit  -> AE on, integration capped at 4 ms;
        #   2) fw does not (ours)   -> MANUAL exposure fixed ~4 ms + a
        #      software gain servo per frame (auto_gain()). Tags tolerate
        #      dark/noisy frames fine; they do not tolerate smear.
        self._gain_servo = False
        try:
            cs = prof.get_device().first_color_sensor()
            self._cs = cs
            if cs.supports(rs.option.auto_exposure_priority):
                # priority=0: AE may not lower fps to gain exposure.
                cs.set_option(rs.option.auto_exposure_priority, 0.0)
            capped = False
            try:
                # Newer firmwares gate the limit behind a toggle option.
                tog = getattr(rs.option, "auto_exposure_limit_toggle", None)
                if tog is not None and cs.supports(tog):
                    cs.set_option(tog, 1.0)
                rng = cs.get_option_range(rs.option.auto_exposure_limit)
                cap = float(min(max(4000.0, rng.min), rng.max))  # us
                cs.set_option(rs.option.enable_auto_exposure, 1.0)
                cs.set_option(rs.option.auto_exposure_limit, cap)
                capped = True
                print(f"[percep] color AE on, blur cap {cap:.0f} us "
                      f"(fw range {rng.min:.0f}..{rng.max:.0f})", flush=True)
            except Exception as e:
                print(f"[percep] no AE limit ({e}) -> manual exposure + "
                      "software gain servo", flush=True)
            if not capped:
                erng = cs.get_option_range(rs.option.exposure)
                # Unit heuristic: some fw report color exposure in us
                # (range up to ~166000), others in 100 us steps (~10000).
                exp = 4000.0 if float(erng.max) >= 20000.0 else 40.0
                self._exp_base = float(np.clip(exp, erng.min, erng.max))
                self._exp_cur = self._exp_base
                self._exp_rng = (float(erng.min), float(erng.max))
                grng = cs.get_option_range(rs.option.gain)
                self._gain_rng = (float(grng.min), float(grng.max))
                self._gain = float(np.clip(
                    float(grng.default), *self._gain_rng
                ))
                cs.set_option(rs.option.enable_auto_exposure, 0.0)
                cs.set_option(rs.option.exposure, self._exp_cur)
                cs.set_option(rs.option.gain, self._gain)
                self._gain_servo = True
                print(f"[percep] manual exposure {self._exp_cur:.0f} "
                      f"(fw range {erng.min:.0f}..{erng.max:.0f}), gain "
                      f"servo start {self._gain:.0f} in "
                      f"[{self._gain_rng[0]:.0f},{self._gain_rng[1]:.0f}]",
                      flush=True)
        except Exception as e:
            print(f"[percep] color exposure setup failed ({e})", flush=True)

        ds = prof.get_device().first_depth_sensor()
        self.scale = float(ds.get_depth_scale())
        intr = (prof.get_stream(rs.stream.depth)
                .as_video_stream_profile().get_intrinsics())
        self.fx, self.fy = float(intr.fx), float(intr.fy)
        self.cx, self.cy = float(intr.ppx), float(intr.ppy)
        self.w, self.h = int(intr.width), int(intr.height)

    def auto_gain(self, rgb) -> None:
        """Software AE at (near-)fixed shutter: gain first, shutter last.

        Keeps mean brightness ~115/255 by servoing gain; only when gain
        saturates does the shutter move, bounded to 2x the 4 ms base
        (dark) or the fw minimum (glare), so blur stays bounded.
        """
        if not self._gain_servo:
            return
        self._ag_n = getattr(self, "_ag_n", 0) + 1
        if self._ag_n % 2:
            return
        try:
            mean = float(np.asarray(rgb)[::8, ::8].mean())
        except Exception:
            return
        err = 115.0 - mean
        if abs(err) < 12.0:
            return
        g_lo, g_hi = self._gain_rng
        step = float(np.clip(0.25 * err, -8.0, 8.0))
        g_new = float(np.clip(self._gain + step, g_lo, g_hi))
        try:
            if abs(g_new - self._gain) >= 0.5:
                self._gain = g_new
                self._cs.set_option(self.rs.option.gain, g_new)
                return
            e_lo, e_hi = self._exp_rng
            e_cap = float(min(2.0 * self._exp_base, e_hi))
            e_new = float(np.clip(
                self._exp_cur * (1.15 if err > 0.0 else 0.85), e_lo, e_cap
            ))
            if abs(e_new - self._exp_cur) >= 1.0:
                self._exp_cur = e_new
                self._cs.set_option(self.rs.option.exposure, e_new)
        except Exception:
            pass

    def frame(self):
        frames = self.align.process(
            self.pipe.wait_for_frames(timeout_ms=2000))
        d = np.asanyarray(frames.get_depth_frame().get_data()).astype(float)
        d *= self.scale
        d[d <= 0.0] = np.nan
        c = frames.get_color_frame()
        rgb = (np.asanyarray(c.get_data()) if c
               else np.zeros((self.h, self.w, 3), np.uint8))
        return d, rgb


class CamRecorder:
    """Replayable RGB-D recording, one directory per run, segment-rotated.

    Every frame the perception loop already grabbed is written -- no extra
    camera work.  Layout (under --record-dir, default logs/cam):

      <stamp>/segNNN/meta.json   intrinsics, depth scale (mm), fps, t0
      <stamp>/segNNN/color.avi   MJPG (BGR), native fps
      <stamp>/segNNN/depth.bin   per frame: [f64 t_wall][u32 len][zlib z16 mm]
      <stamp>/segNNN/ts.csv      frame_idx,t_wall (color/depth share order)

    Replay/export: tools/replay_cam_record.py <segdir>.
    Segments rotate every --record-seg-min minutes so a power cut only
    loses the tail of the current segment.
    """

    def __init__(self, root: Path, src: "D435Local", seg_min: float = 5.0):
        import zlib  # stdlib; bound as attr so frame() needs no re-import
        self._zlib = zlib
        self.src = src
        self.seg_len_s = float(max(0.5, seg_min * 60.0))
        # Always Asia/Shanghai (Beijing), independent of process TZ
        # (2026-08-12 user: "log要记录着北京时间").
        stamp = datetime.now(timezone(timedelta(hours=8))).strftime(
            "%Y%m%d_%H%M%S"
        )
        self.run_dir = Path(root) / stamp
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.seg_idx = -1
        self.vw = None
        self.depth_fp = None
        self.ts_fp = None
        self.frame_idx = 0
        self._open_segment()
        print(f"[percep] RECORDING -> {self.run_dir}", flush=True)

    def _open_segment(self):
        self._close_segment()
        self.seg_idx += 1
        self.seg_t0 = time.time()
        seg = self.run_dir / f"seg{self.seg_idx:03d}"
        seg.mkdir(parents=True, exist_ok=True)
        meta = {
            "w": int(self.src.w), "h": int(self.src.h),
            "fps": int(getattr(self.src, "fps", 15)),
            "depth_unit": "mm_uint16_zlib",
            "fx": self.src.fx, "fy": self.src.fy,
            "cx": self.src.cx, "cy": self.src.cy,
            "t0_wall": self.seg_t0,
        }
        (seg / "meta.json").write_text(json.dumps(meta, indent=1))
        fourcc = cv2.VideoWriter_fourcc(*"MJPG")
        self.vw = cv2.VideoWriter(
            str(seg / "color.avi"), fourcc,
            float(meta["fps"]), (self.src.w, self.src.h),
        )
        self.depth_fp = open(seg / "depth.bin", "wb")
        self.ts_fp = open(seg / "ts.csv", "w")
        self.ts_fp.write("frame_idx,t_wall\n")

    def _close_segment(self):
        if self.vw is not None:
            self.vw.release()
            self.vw = None
        for attr in ("depth_fp", "ts_fp"):
            fp = getattr(self, attr, None)
            if fp is not None:
                try:
                    fp.close()
                except Exception:
                    pass
                setattr(self, attr, None)

    def write(self, d_m: np.ndarray, rgb: np.ndarray, t_wall: float):
        import struct
        if self.vw is None:
            return
        if (t_wall - self.seg_t0) >= self.seg_len_s:
            self._open_segment()
        self.vw.write(cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        d_mm = np.nan_to_num(
            d_m * 1000.0, nan=0.0, posinf=0.0, neginf=0.0
        )
        d_u16 = np.clip(d_mm, 0.0, 65535.0).astype(np.uint16)
        blob = self._zlib.compress(d_u16.tobytes(), 1)
        self.depth_fp.write(struct.pack("<dI", float(t_wall), len(blob)))
        self.depth_fp.write(blob)
        self.ts_fp.write(f"{self.frame_idx},{t_wall:.6f}\n")
        self.frame_idx += 1
        if self.frame_idx % 150 == 0:   # ~10 s at 15 fps
            self.depth_fp.flush()
            self.ts_fp.flush()

    def close(self):
        self._close_segment()


class TagDetector:
    """pupil_apriltags primary; OpenCV ArUco APRILTAG_36h11 fallback.

    Both paths return dicts with tag_id / corners / pose_R / pose_t in the
    repo convention (p_cam = R @ p_tag + t; tag X right, Y down on the
    face, Z out of the face toward the camera).
    """

    def __init__(self, base_tag_size_m: float):
        self.base_size = float(base_tag_size_m)
        self.kind = None
        try:
            from pupil_apriltags import Detector
            self._det = Detector(
                families="tag36h11", nthreads=2, quad_decimate=1.0
            )
            self.kind = "pupil"
        except Exception as e:
            print(f"[percep] pupil_apriltags unavailable ({e}); "
                  "falling back to cv2.aruco", flush=True)
            self._aruco_dict = cv2.aruco.getPredefinedDictionary(
                cv2.aruco.DICT_APRILTAG_36h11
            )
            try:
                self._aruco = cv2.aruco.ArucoDetector(
                    self._aruco_dict, cv2.aruco.DetectorParameters()
                )
            except AttributeError:      # OpenCV < 4.7 legacy API
                self._aruco = None
            self.kind = "aruco"

    def detect(self, gray, fx, fy, cx, cy, size_by_id: dict) -> list[dict]:
        if self.kind == "pupil":
            dets = self._det.detect(
                gray,
                estimate_tag_pose=True,
                camera_params=(fx, fy, cx, cy),
                tag_size=self.base_size,
            )
            out = []
            for det in dets:
                tid = int(det.tag_id)
                pose_t = np.asarray(det.pose_t, dtype=float).reshape(3)
                # Planar-tag pose translation scales linearly with the
                # assumed size: one detect pass at base_size, rescale here.
                size = float(size_by_id.get(tid, self.base_size))
                if abs(size - self.base_size) > 1e-6:
                    pose_t = pose_t * (size / self.base_size)
                out.append({
                    "tag_id": tid,
                    "corners": np.asarray(det.corners, dtype=float),
                    "pose_R": np.asarray(
                        det.pose_R, dtype=float
                    ).reshape(3, 3),
                    "pose_t": pose_t,
                })
            return out
        # --- ArUco fallback: solvePnP per tag with its own size --------
        if self._aruco is not None:
            corners, ids, _rej = self._aruco.detectMarkers(gray)
        else:
            corners, ids, _rej = cv2.aruco.detectMarkers(
                gray, self._aruco_dict
            )
        out = []
        if ids is None:
            return out
        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        # aruco -> pupil frame conversion, verified against pupil_apriltags
        # on synthetic renders (2026-08-08): R_pupil = R_aruco @ diag(-1,1,-1)
        # (180 deg about the tag Y axis; artwork-rotation invariant), t equal.
        D_ARUCO_TO_PUPIL = np.diag([-1.0, 1.0, -1.0])
        for quad, tid in zip(corners, ids.reshape(-1)):
            tid = int(tid)
            s = float(size_by_id.get(tid, self.base_size))
            h = s / 2.0
            # SOLVEPNP_IPPE_SQUARE REQUIRES the canonical object order
            # (-h,+h) (+h,+h) (+h,-h) (-h,-h) <-> image TL,TR,BR,BL.
            # The previous y-down ordering silently produced mirrored /
            # near-zero-depth poses (robot drove the WRONG WAY, 2026-08-08).
            obj = np.array([
                [-h, h, 0.0], [h, h, 0.0], [h, -h, 0.0], [-h, -h, 0.0],
            ], dtype=float)
            img = np.asarray(quad, dtype=float).reshape(4, 2)
            ok, rvec, tvec = cv2.solvePnP(
                obj, img, K, None, flags=cv2.SOLVEPNP_IPPE_SQUARE
            )
            if not ok:
                continue
            tvec = np.asarray(tvec, dtype=float).reshape(3)
            if not (0.02 < float(tvec[2]) < 8.0):
                continue                      # degenerate/behind-camera PnP
            R, _ = cv2.Rodrigues(rvec)
            R = np.asarray(R, dtype=float).reshape(3, 3) @ D_ARUCO_TO_PUPIL
            out.append({
                "tag_id": tid,
                "corners": img,
                "pose_R": R,
                "pose_t": tvec,
            })
        return out


def _unit(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=float).reshape(-1)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else v


def sample_depth_m(
    depth_m: np.ndarray,
    u: float,
    v: float,
    *,
    radius_px: int = 4,
) -> float:
    """Median depth (m) in a small window around image (u,v). NaN if empty."""
    h, w = depth_m.shape[:2]
    ui, vi = int(round(u)), int(round(v))
    r = int(max(0, radius_px))
    u0, u1 = max(0, ui - r), min(w, ui + r + 1)
    v0, v1 = max(0, vi - r), min(h, vi + r + 1)
    if u0 >= u1 or v0 >= v1:
        return float("nan")
    patch = depth_m[v0:v1, u0:u1]
    vals = patch[np.isfinite(patch) & (patch > 0.05) & (patch < 8.0)]
    if vals.size < 3:
        return float("nan")
    return float(np.median(vals))


def wall_normal_from_tag_depth(
    depth_m: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    u: float, v: float,
    T_L_C: np.ndarray,
    *,
    half_px: int = 48,
    stride: int = 2,
    max_nz: float = 0.40,
) -> np.ndarray | None:
    """Fit a local vertical wall plane around the tag and return n_in in L.

    Tag PnP yaw flips (planar pose ambiguity). Depth around the marker
    is a large, stable face — use that for "camera parallel to wall".
    n_in points INTO the wall (same convention as the tag -Z).
    """
    h, w = depth_m.shape[:2]
    ui, vi = int(round(u)), int(round(v))
    r = int(max(8, half_px))
    u0, u1 = max(0, ui - r), min(w, ui + r + 1)
    v0, v1 = max(0, vi - r), min(h, vi + r + 1)
    pts = []
    for vv in range(v0, v1, stride):
        row = depth_m[vv]
        for uu in range(u0, u1, stride):
            z = float(row[uu])
            if not (0.15 < z < 2.5):
                continue
            pts.append((
                (uu - cx) / fx * z,
                (vv - cy) / fy * z,
                z,
            ))
    if len(pts) < 80:
        return None
    try:
        n_cam, _d = fit_dominant_plane(np.asarray(pts, dtype=float), iters=28)
    except Exception:
        return None
    R_LC = np.asarray(T_L_C, dtype=float).reshape(4, 4)[:3, :3]
    n_out_L = R_LC @ np.asarray(n_cam, dtype=float).reshape(3)
    nn = float(np.linalg.norm(n_out_L))
    if nn < 1e-9:
        return None
    n_out_L = n_out_L / nn
    # Wall, not floor: normal must be mostly horizontal in L.
    if abs(float(n_out_L[2])) > float(max_nz):
        return None
    n_in = -n_out_L
    nh = math.hypot(float(n_in[0]), float(n_in[1]))
    if nh < 1e-6:
        return None
    return n_in.astype(float)


def project_L_to_uv(
    p_L,
    T_L_C: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> tuple[float, float] | None:
    """Leg-frame point -> depth/color image (u,v). None if behind camera."""
    T = np.asarray(T_L_C, dtype=float).reshape(4, 4)
    R, t = T[:3, :3], T[:3, 3]
    p_C = R.T @ (np.asarray(p_L, dtype=float).reshape(3) - t)
    z = float(p_C[2])
    if not (0.05 < z < 8.0):
        return None
    return float(fx * p_C[0] / z + cx), float(fy * p_C[1] / z + cy)


def draw_button_marker(
    img: np.ndarray,
    uv,
    *,
    label: str,
    color: tuple,
    radius: int = 10,
) -> None:
    """Draw a crosshair+circle for a virtual button / hover point."""
    if uv is None:
        return
    u, v = int(round(uv[0])), int(round(uv[1]))
    h, w = img.shape[:2]
    if not (0 <= u < w and 0 <= v < h):
        return
    cv2.circle(img, (u, v), radius, color, 2, cv2.LINE_AA)
    cv2.drawMarker(
        img, (u, v), color, markerType=cv2.MARKER_CROSS,
        markerSize=radius * 2, thickness=2, line_type=cv2.LINE_AA,
    )
    cv2.putText(
        img, label, (u + radius + 4, v - 4),
        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2, cv2.LINE_AA,
    )


def _uv_rot90_cw(uv, w: int, h: int):
    """Map a camera-image (u,v) into the 90°-CW rotated preview."""
    if uv is None:
        return None
    try:
        u, v = float(uv[0]), float(uv[1])
    except (TypeError, ValueError, IndexError):
        return None
    return (float(h) - 1.0 - v, u)


def _pts_rot90_cw(pts, w: int, h: int) -> np.ndarray:
    pts = np.asarray(pts, dtype=float).reshape(-1, 2)
    out = np.empty_like(pts)
    out[:, 0] = float(h) - 1.0 - pts[:, 1]
    out[:, 1] = pts[:, 0]
    return out


def _project_cam(p_C, fx: float, fy: float, cx: float, cy: float):
    p = np.asarray(p_C, dtype=float).reshape(3)
    z = float(p[2])
    if not (0.05 < z < 8.0):
        return None
    return (float(fx * p[0] / z + cx), float(fy * p[1] / z + cy))


def scale_pose_t_to_depth(
    pose_t: np.ndarray,
    depth_z_m: float,
) -> tuple[np.ndarray, float]:
    """Scale PnP translation so its Z matches the depth reading.

    Planar-tag PnP range scales with the assumed tag size; a few-mm size
    error becomes tens of cm at ~0.5 m. Depth is metric. Returns
    (scaled_t, scale). If depth is unusable, returns the original t and 1.
    """
    t = np.asarray(pose_t, dtype=float).reshape(3).copy()
    pnp_z = float(t[2])
    if not (0.05 < pnp_z < 8.0 and 0.05 < float(depth_z_m) < 8.0):
        return t, 1.0
    scale = float(depth_z_m) / pnp_z
    # Reject absurd corrections (wrong pixel / flying pixel).
    if not (0.5 < scale < 2.0):
        return t, 1.0
    return t * scale, scale


def box_from_tags(
    dets_by_id: dict,
    *,
    left_id: int,
    right_id: int,
    spacing_m: float,
    T_L_C: np.ndarray,
) -> dict | None:
    """Box-face estimate from the symmetric tag pair (or one of them).

    Yaw/right-axis from the two tag-center baseline when both are seen
    (robust: position-only, no planar pose ambiguity); single-tag
    fallback offsets along that tag's own X axis by half the spacing.
    """
    R_LC = np.asarray(T_L_C, dtype=float).reshape(4, 4)[:3, :3]
    t_LC = np.asarray(T_L_C, dtype=float).reshape(4, 4)[:3, 3]

    def to_L(p_cam):
        return R_LC @ np.asarray(p_cam, dtype=float).reshape(3) + t_LC

    left = dets_by_id.get(int(left_id))
    right = dets_by_id.get(int(right_id))
    if left is None and right is None:
        return None

    seen = [d for d in (left, right) if d is not None]
    # Face normal (out of the face toward camera) = tag +Z, averaged.
    n_out_L = _unit(np.mean(
        [R_LC @ d["pose_R"][:, 2] for d in seen], axis=0
    ))
    cam_z = float(np.mean([float(d["pose_t"][2]) for d in seen]))

    if left is not None and right is not None:
        pL = to_L(left["pose_t"])
        pR = to_L(right["pose_t"])
        center_L = 0.5 * (pL + pR)
        baseline = pR - pL
        width_m = float(np.linalg.norm(baseline))
        face_right_L = _unit(baseline)
        source = "tag2"
    else:
        d = left if left is not None else right
        # Tag X axis = "right on the tag face" in the repo convention.
        face_right_L = _unit(R_LC @ d["pose_R"][:, 0])
        sign = +1.0 if d is left else -1.0
        center_L = to_L(d["pose_t"]) + sign * (spacing_m / 2.0) * face_right_L
        width_m = float(spacing_m)
        source = "tag1of2"
    return {
        "valid": True,
        "source": source,
        "center_L": [float(x) for x in center_L],
        "normal_out_L": [float(x) for x in n_out_L],
        "normal_in_L": [float(x) for x in -n_out_L],
        "face_right_L": [float(x) for x in face_right_L],
        "width_m": width_m,
        "tag_ids_seen": [int(d["tag_id"]) for d in seen],
        "cam_z_m": cam_z,
    }


def box_from_plane(
    depth_m: np.ndarray,
    fx: float, fy: float, cx: float, cy: float,
    *,
    T_L_C: np.ndarray,
    min_width_m: float,
    max_normal_z: float = 0.35,
) -> dict | None:
    """Vertical-plane fallback (no tags): find a large box-like face.

    Fits the dominant plane; if that is the ground (normal near L +/-Z,
    i.e. gravity), removes its inliers and fits once more. Only a plane
    whose normal is near-horizontal in L and whose in-plane extent is
    wide enough is reported.
    """
    R_LC = np.asarray(T_L_C, dtype=float).reshape(4, 4)[:3, :3]
    t_LC = np.asarray(T_L_C, dtype=float).reshape(4, 4)[:3, 3]
    pts = depth_to_ground_points(
        depth_m, fx, fy, cx, cy, np.eye(3), np.zeros(3), stride=8
    )
    pts = pts[np.isfinite(pts).all(axis=1)]
    pts = pts[pts[:, 2] < 3.0]           # ignore far background
    if len(pts) > 4000:
        pts = pts[:: len(pts) // 4000]
    for _attempt in range(2):
        if len(pts) < 300:
            return None
        try:
            n_cam, d = fit_dominant_plane(pts, iters=40)
        except ValueError:
            return None
        inl = np.abs(pts @ n_cam + d) < 0.02
        n_L = _unit(R_LC @ n_cam)
        if abs(float(n_L[2])) <= float(max_normal_z):
            q = pts[inl]
            if len(q) < 200:
                return None
            center_cam = np.median(q, axis=0)
            # In-plane axes: right = horizontal in L, up = the rest.
            up_L = np.array([0.0, 0.0, -1.0])   # L is FRD: -Z is up
            right_L = _unit(np.cross(up_L, n_L))
            q_L = q @ R_LC.T + t_LC.reshape(1, 3)
            c_L = R_LC @ center_cam + t_LC
            rel = q_L - c_L.reshape(1, 3)
            r_span = rel @ right_L
            u_span = rel @ _unit(np.cross(n_L, right_L))
            width = float(np.percentile(r_span, 95)
                          - np.percentile(r_span, 5))
            height = float(np.percentile(u_span, 95)
                           - np.percentile(u_span, 5))
            if width < float(min_width_m):
                return None
            n_out = n_L if float(n_L @ (t_LC - c_L)) > 0.0 else -n_L
            return {
                "valid": True,
                "source": "plane",
                "center_L": [float(x) for x in c_L],
                "normal_out_L": [float(x) for x in n_out],
                "normal_in_L": [float(x) for x in -n_out],
                "face_right_L": [float(x) for x in right_L],
                "width_m": width,
                "height_m": height,
                "tag_ids_seen": [],
                "cam_z_m": float(center_cam[2]),
            }
        # Dominant plane is the ground -- drop it and retry once.
        pts = pts[~inl]
    return None


def encode_preview(rgb_annotated: np.ndarray, max_bytes: int) -> bytes | None:
    """JPEG that fits one UDP datagram; shrink quality/scale as needed."""
    img = cv2.cvtColor(rgb_annotated, cv2.COLOR_RGB2BGR)
    if img.shape[1] > 480:
        s = 480.0 / img.shape[1]
        img = cv2.resize(img, (0, 0), fx=s, fy=s)
    for q in (70, 50, 35, 20):
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
        if ok and len(buf) <= max_bytes:
            return buf.tobytes()
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag-id", type=int, default=1,
                    help="wall-button AprilTag id")
    ap.add_argument("--tag-size", type=float, default=0.09)
    ap.add_argument("--button-right-m", type=float, default=DEFAULT_RIGHT_M)
    ap.add_argument("--button-down-m", type=float, default=DEFAULT_DOWN_M)
    ap.add_argument("--button-protrude-m", type=float, default=0.05)
    ap.add_argument("--button-press-m", type=float, default=0.02)
    ap.add_argument("--box-tag-ids", type=int, nargs=2, default=(2, 3),
                    metavar=("LEFT", "RIGHT"),
                    help="box-face tag pair, symmetric about the center "
                         "(LEFT/RIGHT as seen facing the box)")
    # 2026-08-13: box_apriltags_id2_id3_60mm_a4.pdf — 60 mm tags, 60 mm
    # outer gap -> center-to-center 120 mm.
    ap.add_argument("--box-tag-size", type=float, default=0.060)
    ap.add_argument("--box-tag-spacing", type=float, default=0.120,
                    help="center-to-center tag distance on the box face (m)")
    ap.add_argument("--box-min-width-m", type=float, default=0.25,
                    help="plane fallback: min face width to call it a box")
    ap.add_argument("--no-plane", action="store_true",
                    help="disable the vertical-plane box fallback")
    ap.add_argument("--plane-period", type=float, default=0.5,
                    help="seconds between plane-fallback fits")
    ap.add_argument("--setpoint-host", default="127.0.0.1",
                    help="controller host (loopback: same Jetson)")
    ap.add_argument("--setpoint-port", type=int, default=5558)
    # Subnet-directed broadcast, NOT 255.255.255.255: the limited broadcast
    # egresses only via the DEFAULT-route interface, which on this Jetson is
    # WiFi (10.161.x) -- the PC on the wired 192.168.1.0/24 link never saw a
    # packet (2026-08-08). 192.168.1.255 routes out the wired NIC.
    ap.add_argument("--monitor-host", default="192.168.1.255")
    ap.add_argument("--monitor-port", type=int, default=5559)
    ap.add_argument("--preview-port", type=int, default=5560)
    ap.add_argument("--preview-hz", type=float, default=5.0)
    ap.add_argument(
        "--setpoint-json",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "logs" / "button_setpoint.json",
        help="low-rate local file fallback (controller reads it if UDP "
             "is down)",
    )
    # RGB-D recording (2026-08-09 user: every camera frame the upper
    # session sees must be stored and replayable -- depth AND color).
    ap.add_argument("--no-record", action="store_true",
                    help="disable RGB-D recording")
    ap.add_argument(
        "--record-dir",
        type=Path,
        default=Path(__file__).resolve().parent.parent / "logs" / "cam",
        help="recording root (one <stamp>/ run dir per start)",
    )
    ap.add_argument("--record-seg-min", type=float, default=5.0,
                    help="segment rotation period (minutes)")
    args = ap.parse_args()

    src = D435Local()
    rec = None
    if not args.no_record:
        try:
            rec = CamRecorder(
                args.record_dir, src, seg_min=float(args.record_seg_min)
            )
            # systemd stop sends SIGTERM: exit through atexit so the AVI
            # index and the depth/ts buffers are finalized, not truncated.
            import atexit
            import signal
            atexit.register(rec.close)
            signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
        except Exception as e:
            print(f"[percep] recording DISABLED ({e})", flush=True)
            rec = None
    T_L_C = load_T_L_C()
    size_by_id = {
        int(args.tag_id): float(args.tag_size),
        int(args.box_tag_ids[0]): float(args.box_tag_size),
        int(args.box_tag_ids[1]): float(args.box_tag_size),
    }
    det = TagDetector(float(args.tag_size))
    print(
        f"[percep] backend={det.kind} button_id={args.tag_id} "
        f"box_ids={tuple(args.box_tag_ids)} "
        f"spacing={args.box_tag_spacing*100:.0f}cm "
        f"-> {args.setpoint_host}:{args.setpoint_port}",
        flush=True,
    )

    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    mon_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    mon_sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
    args.setpoint_json.parent.mkdir(parents=True, exist_ok=True)

    box_last: dict | None = None
    box_last_t = 0.0
    plane_next_t = 0.0
    file_next_t = 0.0
    preview_next_t = 0.0
    depth_log_t = 0.0
    n = 0
    hz_t0 = time.time()
    hz = 0.0

    while True:
        try:
            d, rgb = src.frame()
        except RuntimeError as e:
            print(f"[percep] frame timeout ({e})", flush=True)
            continue
        if hasattr(src, "auto_gain"):
            src.auto_gain(rgb)
        now = time.time()
        if rec is not None:
            try:
                rec.write(d, rgb, now)
            except Exception as e:
                print(f"[percep] record write failed ({e}); "
                      "recording OFF", flush=True)
                try:
                    rec.close()
                except Exception:
                    pass
                rec = None
        n += 1
        if now - hz_t0 >= 2.0:
            hz = n / (now - hz_t0)
            n = 0
            hz_t0 = now

        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        try:
            dets = det.detect(
                gray, src.fx, src.fy, src.cx, src.cy, size_by_id
            )
        except Exception as e:
            print(f"[percep] detector error: {e}", flush=True)
            dets = []
        dets_by_id = {d_["tag_id"]: d_ for d_ in dets}

        # ---- button setpoint (schema unchanged vs terrain_gate_live) ----
        payload = {
            "t_wall": now,
            "valid": False,
            "tag_id": int(args.tag_id),
            "tag_size_m": float(args.tag_size),
            "right_m": float(args.button_right_m),
            "down_m": float(args.button_down_m),
            "protrude_m": float(args.button_protrude_m),
            "press_m": float(args.button_press_m),
            "press_along_wall_normal": True,
        }
        btn = dets_by_id.get(int(args.tag_id))
        if btn is not None:
            # Metric range from aligned depth at the tag center (and a
            # light average of the four corners). PnP Z alone drifts with
            # tag-size / IPPE ambiguity -- that was reporting ~0.5 m when
            # the robot was still farther (user 2026-08-08).
            c_uv = np.mean(btn["corners"], axis=0)
            z_depth = sample_depth_m(d, c_uv[0], c_uv[1], radius_px=5)
            corner_zs = [
                sample_depth_m(d, uv[0], uv[1], radius_px=3)
                for uv in btn["corners"]
            ]
            corner_zs = [z for z in corner_zs if np.isfinite(z)]
            if corner_zs:
                z_c = float(np.median(corner_zs))
                if np.isfinite(z_depth):
                    z_depth = 0.5 * (z_depth + z_c)
                else:
                    z_depth = z_c
            pnp_z = float(btn["pose_t"][2])
            pose_t, z_scale = scale_pose_t_to_depth(btn["pose_t"], z_depth)
            targets = button_targets_from_detection(
                pose_R=btn["pose_R"],
                pose_t=pose_t,
                T_L_C=T_L_C,
                right_m=float(args.button_right_m),
                down_m=float(args.button_down_m),
                protrude_m=float(args.button_protrude_m),
                press_m=float(args.button_press_m),
            )
            n_in = np.asarray(targets["wall_normal_in_L"], dtype=float)
            n_out = np.asarray(targets["wall_normal_out_L"], dtype=float)
            n_depth = wall_normal_from_tag_depth(
                d, src.fx, src.fy, src.cx, src.cy,
                float(c_uv[0]), float(c_uv[1]), T_L_C,
            )
            wall_src = "tag"
            if n_depth is not None:
                # Depth plane yaw is stable; keep the tag hemisphere.
                if float(np.dot(n_depth[:2], n_in[:2])) < 0.0:
                    n_depth = -n_depth
                n_in = n_depth
                n_out = -n_in
                wall_src = "depth"
            # Prefer depth for the published range; fall back to scaled PnP.
            z_pub = float(z_depth) if np.isfinite(z_depth) else float(pose_t[2])
            T_C_tag = np.asarray(targets["T_C_tag"], dtype=float).reshape(4, 4)

            def _tag_uv(p_tag):
                p_C = (
                    T_C_tag[:3, :3]
                    @ np.asarray(p_tag, dtype=float).reshape(3)
                    + T_C_tag[:3, 3]
                )
                return _project_cam(p_C, src.fx, src.fy, src.cx, src.cy)

            o_uv = _tag_uv((0.0, 0.0, 0.0))
            # Overlay arrows follow wall-right / wall-down (the offset
            # signs), not raw tag +X/+Y — those point left/up on this
            # camera (screenshot 02:17).
            ax = 0.08 if float(args.button_right_m) >= 0.0 else -0.08
            ay = 0.08 if float(args.button_down_m) >= 0.0 else -0.08
            x_uv = _tag_uv((ax, 0.0, 0.0))
            y_uv = _tag_uv((0.0, ay, 0.0))
            payload.update({
                "valid": True,
                "foot_pre_L": [float(x) for x in targets["leg"]["pre"]],
                "foot_face_L": [float(x) for x in targets["leg"]["face"]],
                "foot_press_L": [float(x) for x in targets["leg"]["press"]],
                "tag_origin_uv": ([float(o_uv[0]), float(o_uv[1])]
                                  if o_uv is not None else None),
                "tag_x_uv": ([float(x_uv[0]), float(x_uv[1])]
                             if x_uv is not None else None),
                "tag_y_uv": ([float(y_uv[0]), float(y_uv[1])]
                             if y_uv is not None else None),
                # Tag center in L (FOV keep target). Button is ~19.5 cm
                # right of the tag; approach uses this to center the
                # marker in the image before the final press offset.
                "tag_center_L": [
                    float(x) for x in targets["leg"]["tag_center"]
                ],
                "wall_normal_in_L": [float(x) for x in n_in],
                "wall_normal_out_L": [float(x) for x in n_out],
                "wall_normal_source": wall_src,
                "tag_cam_z_m": z_pub,
                "tag_pnp_z_m": pnp_z,
                "tag_depth_z_m": (
                    float(z_depth) if np.isfinite(z_depth) else None
                ),
                "tag_z_scale": float(z_scale),
            })
            if now >= depth_log_t:
                depth_log_t = now + 1.0
                yaw_l = math.degrees(math.atan2(float(n_in[1]), float(n_in[0])))
                print(
                    f"[percep] tag id={args.tag_id} "
                    f"depth_z={z_depth if np.isfinite(z_depth) else float('nan'):.3f} m "
                    f"pnp_z={pnp_z:.3f} m scale={z_scale:.3f} "
                    f"wall={wall_src} n_yaw={yaw_l:+.1f}deg "
                    f"pre_L=({targets['leg']['pre'][0]:+.2f},"
                    f"{targets['leg']['pre'][1]:+.2f},"
                    f"{targets['leg']['pre'][2]:+.2f})",
                    flush=True,
                )

        # ---- box face: tag pair first, plane fallback at low rate ------
        box = box_from_tags(
            dets_by_id,
            left_id=int(args.box_tag_ids[0]),
            right_id=int(args.box_tag_ids[1]),
            spacing_m=float(args.box_tag_spacing),
            T_L_C=T_L_C,
        )
        if box is None and not args.no_plane and now >= plane_next_t:
            plane_next_t = now + float(max(0.1, args.plane_period))
            try:
                box = box_from_plane(
                    d, src.fx, src.fy, src.cx, src.cy,
                    T_L_C=T_L_C,
                    min_width_m=float(args.box_min_width_m),
                )
            except Exception as e:
                print(f"[percep] plane fit error: {e}", flush=True)
                box = None
        if box is not None:
            box_last, box_last_t = box, now
        elif box_last is not None and (now - box_last_t) > 1.0:
            box_last = None
        if box_last is not None:
            payload["box"] = dict(box_last)
            payload["box"]["age_s"] = round(now - box_last_t, 3)

        # ---- publish: loopback UDP + low-rate file fallback -------------
        encoded = json.dumps(payload)
        try:
            sock.sendto(
                encoded.encode("utf-8"),
                (str(args.setpoint_host), int(args.setpoint_port)),
            )
        except OSError:
            pass
        if now >= file_next_t:
            file_next_t = now + 0.2
            try:
                tmp = args.setpoint_json.with_suffix(".json.tmp")
                tmp.write_text(encoded)
                tmp.replace(args.setpoint_json)
            except OSError:
                pass

        # ---- PC monitor stream ------------------------------------------
        # Camera is mounted on its side. Rotate the preview 90° CW FIRST,
        # then draw labels so the text stays upright (previously putText
        # ran on the raw frame and the monitor rotation made it sideways).
        if now >= preview_next_t:
            preview_next_t = now + 1.0 / float(max(0.5, args.preview_hz))
            h0, w0 = rgb.shape[:2]
            ann = np.ascontiguousarray(np.rot90(rgb, k=-1))
            for d_ in dets:
                tid = d_["tag_id"]
                col = ((255, 230, 0) if tid == int(args.tag_id)
                       else (0, 255, 100))
                quad = _pts_rot90_cw(d_["corners"], w0, h0)
                cv2.polylines(
                    ann, [np.rint(quad).astype(np.int32)],
                    True, col, 2, cv2.LINE_AA,
                )
                c_uv = np.rint(np.mean(quad, axis=0)).astype(int)
                label = f"id{tid}"
                if tid == int(args.tag_id) and payload.get("valid"):
                    zd = payload.get("tag_depth_z_m")
                    zp = payload.get("tag_pnp_z_m")
                    if zd is not None and zp is not None:
                        label = f"id{tid} D{zd:.2f}/P{zp:.2f}m"
                    elif zd is not None:
                        label = f"id{tid} D{zd:.2f}m"
                cv2.putText(
                    ann, label, (int(c_uv[0]) + 8, int(c_uv[1]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA,
                )
            btn_uv = pre_uv = press_uv = None
            if payload.get("valid"):
                btn_uv = _uv_rot90_cw(project_L_to_uv(
                    payload["foot_face_L"], T_L_C,
                    src.fx, src.fy, src.cx, src.cy,
                ), w0, h0)
                pre_uv = _uv_rot90_cw(project_L_to_uv(
                    payload["foot_pre_L"], T_L_C,
                    src.fx, src.fy, src.cx, src.cy,
                ), w0, h0)
                press_uv = _uv_rot90_cw(project_L_to_uv(
                    payload["foot_press_L"], T_L_C,
                    src.fx, src.fy, src.cx, src.cy,
                ), w0, h0)
                ox = _uv_rot90_cw(payload.get("tag_origin_uv"), w0, h0)
                xx = _uv_rot90_cw(payload.get("tag_x_uv"), w0, h0)
                yy = _uv_rot90_cw(payload.get("tag_y_uv"), w0, h0)
                if ox is not None and xx is not None:
                    cv2.arrowedLine(
                        ann,
                        (int(round(ox[0])), int(round(ox[1]))),
                        (int(round(xx[0])), int(round(xx[1]))),
                        (0, 180, 255), 2, cv2.LINE_AA, tipLength=0.25,
                    )
                    cv2.putText(
                        ann, "right/BTN",
                        (int(round(xx[0])) + 4, int(round(xx[1]))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 180, 255), 1,
                        cv2.LINE_AA,
                    )
                if ox is not None and yy is not None:
                    cv2.arrowedLine(
                        ann,
                        (int(round(ox[0])), int(round(ox[1]))),
                        (int(round(yy[0])), int(round(yy[1]))),
                        (80, 255, 80), 2, cv2.LINE_AA, tipLength=0.25,
                    )
                    cv2.putText(
                        ann, "down",
                        (int(round(yy[0])) + 4, int(round(yy[1]))),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (80, 255, 80), 1,
                        cv2.LINE_AA,
                    )
                if ox is not None and btn_uv is not None:
                    cv2.line(
                        ann,
                        (int(round(ox[0])), int(round(ox[1]))),
                        (int(round(btn_uv[0])), int(round(btn_uv[1]))),
                        (255, 80, 255), 1, cv2.LINE_AA,
                    )
                draw_button_marker(
                    ann, pre_uv, label="PRE", color=(80, 200, 255), radius=8
                )
                draw_button_marker(
                    ann, btn_uv, label="BTN", color=(255, 60, 255), radius=12
                )
                draw_button_marker(
                    ann, press_uv, label="PRESS", color=(60, 255, 120),
                    radius=7,
                )
                cv2.putText(
                    ann,
                    (
                        f"BTN = RIGHT {abs(args.button_right_m)*100:.1f}cm  "
                        f"DOWN {abs(args.button_down_m)*100:.1f}cm (on wall)"
                    ),
                    (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 60, 255),
                    1, cv2.LINE_AA,
                )
            metrics = {
                "t_wall": now,
                "hz": round(hz, 1),
                "backend": det.kind,
                "button": {
                    "valid": bool(payload["valid"]),
                    "tag_cam_z_m": payload.get("tag_cam_z_m"),
                    "tag_depth_z_m": payload.get("tag_depth_z_m"),
                    "tag_pnp_z_m": payload.get("tag_pnp_z_m"),
                    "tag_z_scale": payload.get("tag_z_scale"),
                    "foot_face_L": payload.get("foot_face_L"),
                    "foot_pre_L": payload.get("foot_pre_L"),
                    "btn_uv": ([float(btn_uv[0]), float(btn_uv[1])]
                               if btn_uv is not None else None),
                    "pre_uv": ([float(pre_uv[0]), float(pre_uv[1])]
                               if pre_uv is not None else None),
                    "offset": (
                        f"right {abs(args.button_right_m)*100:.1f}cm, "
                        f"down {abs(args.button_down_m)*100:.1f}cm (tag -X -Y)"
                    ),
                },
                "box": payload.get("box"),
                "tag_ids_seen": sorted(dets_by_id.keys()),
                "rotate_cw": 0,
            }
            try:
                mon_sock.sendto(
                    json.dumps(metrics).encode("utf-8"),
                    (str(args.monitor_host), int(args.monitor_port)),
                )
                jpg = encode_preview(ann, max_bytes=60000)
                if jpg is not None:
                    mon_sock.sendto(
                        jpg,
                        (str(args.monitor_host), int(args.preview_port)),
                    )
            except OSError:
                pass


if __name__ == "__main__":
    main()
