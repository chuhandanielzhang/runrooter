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
    (default ids 2/3, known spacing). Yaw comes from the tag-center
    BASELINE, not from single-tag orientation (avoids the planar pose
    ambiguity flip). One visible tag still yields the center through the
    known offset along its own X axis.
  - Plane fallback: when no box tag is visible, a low-rate RANSAC looks
    for a large vertical plane (ground plane is fitted and removed
    first).

Deps on the Jetson: pyrealsense2, numpy, cv2, pupil_apriltags
(pip install pupil-apriltags). matplotlib is NOT needed.

Usage:
  jetson$ python3 tools/jetson_perception.py
  jetson$ python3 tools/jetson_perception.py --box-tag-ids 2 3 \
              --box-tag-spacing 0.30
"""
from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modee.terrain_gate import (  # noqa: E402
    depth_to_ground_points, fit_dominant_plane,
)
from button_apriltag_geometry import (  # noqa: E402
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
                print(f"[percep] D435 {W}x{H} @ {FPS} fps", flush=True)
                break
            except RuntimeError as e:
                print(f"[percep] {W}x{H}@{FPS} unavailable ({e})", flush=True)
        if prof is None:
            raise SystemExit("no workable depth+color profile")
        self.align = rs.align(rs.stream.depth)
        # Motion-blur guard (2026-08-08): the D435 COLOR sensor is rolling
        # shutter with auto-exposure; dim indoor light pushes exposure toward
        # the full frame period (~66 ms @15 fps) and the tag smears whenever
        # the base moves. Blur scales with exposure time, so keep AE but
        # cap it at 10 ms and hold the frame rate (priority=0).
        try:
            cs = prof.get_device().first_color_sensor()
            if cs.supports(rs.option.auto_exposure_priority):
                cs.set_option(rs.option.auto_exposure_priority, 0.0)
            # NOTE: supports() misreports auto_exposure_limit as False on
            # some librealsense builds -- just try to set it.
            try:
                cs.set_option(rs.option.auto_exposure_limit, 10000.0)  # us
                print("[percep] color AE capped at 10 ms (blur guard)",
                      flush=True)
            except Exception:
                # Firmware without AE-limit: fixed exposure. 8 ms / gain 90
                # (earlier blur guard) was too dark indoors -- tag dropped
                # at ~0.6 m on approach (log 2026-08-08 21:21) while still
                # centered in frame. 15 ms @ 0.12 m/s -> ~1.8 mm smear,
                # fine for a 9 cm tag; gain 128 lifts SNR without AE lag.
                cs.set_option(rs.option.enable_auto_exposure, 0.0)
                cs.set_option(rs.option.exposure, 150.0)   # 15 ms
                cs.set_option(rs.option.gain, 128.0)
                print("[percep] color fixed exposure 15 ms, gain 128 "
                      "(indoor tag, no AE-limit fw)", flush=True)
        except Exception as e:
            print(f"[percep] exposure cap unavailable ({e})", flush=True)
        ds = prof.get_device().first_depth_sensor()
        self.scale = float(ds.get_depth_scale())
        intr = (prof.get_stream(rs.stream.depth)
                .as_video_stream_profile().get_intrinsics())
        self.fx, self.fy = float(intr.fx), float(intr.fy)
        self.cx, self.cy = float(intr.ppx), float(intr.ppy)
        self.w, self.h = int(intr.width), int(intr.height)

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
    ap.add_argument("--button-right-m", type=float, default=0.165)
    ap.add_argument("--button-down-m", type=float, default=0.026)
    ap.add_argument("--button-protrude-m", type=float, default=0.05)
    ap.add_argument("--button-press-m", type=float, default=0.02)
    ap.add_argument("--box-tag-ids", type=int, nargs=2, default=(2, 3),
                    metavar=("LEFT", "RIGHT"),
                    help="box-face tag pair, symmetric about the center "
                         "(LEFT/RIGHT as seen facing the box)")
    ap.add_argument("--box-tag-size", type=float, default=0.09)
    ap.add_argument("--box-tag-spacing", type=float, default=0.30,
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
    args = ap.parse_args()

    src = D435Local()
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
        now = time.time()
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
            # Prefer depth for the published range; fall back to scaled PnP.
            z_pub = float(z_depth) if np.isfinite(z_depth) else float(pose_t[2])
            payload.update({
                "valid": True,
                "foot_pre_L": [float(x) for x in targets["leg"]["pre"]],
                "foot_face_L": [float(x) for x in targets["leg"]["face"]],
                "foot_press_L": [float(x) for x in targets["leg"]["press"]],
                # Tag center in L (FOV keep target). Button is ~16.5 cm
                # right of the tag; approach uses this to center the
                # marker in the image before the final press offset.
                "tag_center_L": [
                    float(x) for x in targets["leg"]["tag_center"]
                ],
                "wall_normal_in_L": [
                    float(x) for x in targets["wall_normal_in_L"]
                ],
                "wall_normal_out_L": [
                    float(x) for x in targets["wall_normal_out_L"]
                ],
                "tag_cam_z_m": z_pub,
                "tag_pnp_z_m": pnp_z,
                "tag_depth_z_m": (
                    float(z_depth) if np.isfinite(z_depth) else None
                ),
                "tag_z_scale": float(z_scale),
            })
            if now >= depth_log_t:
                depth_log_t = now + 1.0
                print(
                    f"[percep] tag id={args.tag_id} "
                    f"depth_z={z_depth if np.isfinite(z_depth) else float('nan'):.3f} m "
                    f"pnp_z={pnp_z:.3f} m scale={z_scale:.3f} "
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
        if now >= preview_next_t:
            preview_next_t = now + 1.0 / float(max(0.5, args.preview_hz))
            ann = np.ascontiguousarray(rgb)
            for d_ in dets:
                tid = d_["tag_id"]
                col = ((255, 230, 0) if tid == int(args.tag_id)
                       else (0, 255, 100))
                cv2.polylines(
                    ann, [np.rint(d_["corners"]).astype(np.int32)],
                    True, col, 2, cv2.LINE_AA,
                )
                c_uv = np.rint(np.mean(d_["corners"], axis=0)).astype(int)
                label = f"id{tid}"
                if tid == int(args.tag_id) and payload.get("valid"):
                    zd = payload.get("tag_depth_z_m")
                    zp = payload.get("tag_pnp_z_m")
                    if zd is not None and zp is not None:
                        label = f"id{tid} D{zd:.2f}/P{zp:.2f}m"
                    elif zd is not None:
                        label = f"id{tid} D{zd:.2f}m"
                cv2.putText(
                    ann, label, tuple(c_uv),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA,
                )
            # Virtual button geometry (tag + offsets) projected into the
            # camera image so the PC monitor shows where the robot thinks
            # the button / hover / press points are.
            btn_uv = pre_uv = press_uv = None
            if payload.get("valid"):
                btn_uv = project_L_to_uv(
                    payload["foot_face_L"], T_L_C,
                    src.fx, src.fy, src.cx, src.cy,
                )
                pre_uv = project_L_to_uv(
                    payload["foot_pre_L"], T_L_C,
                    src.fx, src.fy, src.cx, src.cy,
                )
                press_uv = project_L_to_uv(
                    payload["foot_press_L"], T_L_C,
                    src.fx, src.fy, src.cx, src.cy,
                )
                # Line tag-center -> button makes the lateral offset obvious.
                if btn is not None and btn_uv is not None:
                    tag_uv = np.mean(btn["corners"], axis=0)
                    cv2.line(
                        ann,
                        (int(round(tag_uv[0])), int(round(tag_uv[1]))),
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
                },
                "box": payload.get("box"),
                "tag_ids_seen": sorted(dets_by_id.keys()),
                "rotate_cw": 90,
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
