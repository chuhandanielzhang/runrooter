#!/usr/bin/env python3
"""Live flat-ground checker: RGB scene + depth + corridor profile.

DEFAULT (verdict mode): FLAT / SLOPE / STEP -> HOP / BLOCKED / NO
GROUND.  Every frame the dominant plane is RANSAC-fitted from the point
cloud and the cloud is leveled to it, so the camera can be HANDHELD at
any angle -- no mount calibration needed.  Caveat: handheld leveling
makes a GLOBAL slope unobservable (camera tilt and ground tilt are the
same math without an IMU), so SLOPE is only reported with a trusted
pose (--fixed-pose / --full, i.e. mounted on the robot).

With a RealSense D435 plugged in, streams real depth + color; otherwise
falls back to a synthetic demo (approach a cable / step / wall in a
loop).

Panels:
  left   RGB scene, ANNOTATED: corridor footprint and the detected step
  middle depth (turbo)
  right  corridor height profile vs the flat threshold
Banner: FLAT (green) / NOT FLAT (red) / NO GROUND (gray) + metrics.

Usage:
  python3 tools/terrain_gate_live.py                 # local camera or demo
  python3 tools/terrain_gate_live.py --net 192.168.1.100
                             # camera on the Jetson via d435_net_server.py
  python3 tools/terrain_gate_live.py --net 192.168.1.100 --rotate 90
                             # clockwise display rotation (camera mount)
  python3 tools/terrain_gate_live.py --demo          # force synthetic
  python3 tools/terrain_gate_live.py --full          # old 3-mode gate, fixed pose
  python3 tools/terrain_gate_live.py --fixed-pose --pitch 30 --cam-height 0.367
  python3 tools/terrain_gate_live.py --headless --frames 40   # smoke test
"""
import argparse
import json
import socket
import sys
import time
from collections import deque
from pathlib import Path

import cv2
import numpy as np


def _rot_img(img: np.ndarray, rot_cw: int) -> np.ndarray:
    """Rotate image clockwise by rot_cw degrees (0/90/180/270)."""
    k = (int(rot_cw) // 90) % 4
    if k == 0:
        return img
    return np.rot90(img, k=-k)  # np.rot90 is CCW; negative -> CW


def _rot_uv(uv: np.ndarray, w: int, h: int, rot_cw: int) -> np.ndarray:
    """Map original pixel (u,v) into display coords after CW rotation."""
    k = (int(rot_cw) // 90) % 4
    if k == 0 or uv.size == 0:
        return uv
    u, v = uv[:, 0], uv[:, 1]
    if k == 1:      # CW 90: (u,v) -> (h-1-v, u)
        return np.stack([h - 1.0 - v, u], axis=1)
    if k == 2:      # 180
        return np.stack([w - 1.0 - u, h - 1.0 - v], axis=1)
    # CW 270
    return np.stack([v, w - 1.0 - u], axis=1)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from modee.terrain_gate import (  # noqa: E402
    TerrainGate, TerrainGateConfig, depth_to_ground_points,
    fit_dominant_plane, plane_frame,
)

W, H = 848, 480
MODE_COLOR = {"WHEEL": "#2ca02c", "HOP": "#ff7f0e", "STOP": "#d62728"}
FLAT_COLOR = {"FLAT": "#2ca02c", "NOT FLAT": "#d62728", "NO GROUND": "#7f7f7f"}
VERDICT_COLOR = {
    "FLAT": "#2ca02c", "SLOPE": "#1f77b4", "STEP -> HOP": "#ff7f0e",
    "BLOCKED": "#d62728", "NO GROUND": "#7f7f7f",
}


def describe_vote(vote, cfg, slope_valid: bool) -> str:
    """4-class terrain verdict from the geometric metrics.

    slope_valid: only trust the slope metric when the camera pose is
    known (fixed mount).  With handheld auto-ground leveling a global
    slope is indistinguishable from camera tilt and reads ~0.
    """
    if vote.n_valid_bins < 3 or not np.isfinite(vote.step_m):
        return "NO GROUND"
    if (vote.step_m <= cfg.step_wheel_max_m
            and vote.rough_m <= cfg.rough_wheel_max_m):
        if slope_valid and abs(vote.slope_deg) > cfg.slope_wheel_max_deg:
            return "SLOPE"
        return "FLAT"
    if vote.step_m <= cfg.step_hop_max_m:
        return "STEP -> HOP"
    return "BLOCKED"


class Debounce:
    """Same hysteresis as TerrainGate, applied to the verdict string."""

    def __init__(self, n: int, initial: str = "NO GROUND"):
        self.n, self.latched = n, initial
        self._pend, self._cnt = None, 0

    def update(self, v: str) -> str:
        if v == self.latched:
            self._pend, self._cnt = None, 0
        elif v == self._pend:
            self._cnt += 1
            if self._cnt >= self.n:
                self.latched, self._pend, self._cnt = v, None, 0
        else:
            self._pend, self._cnt = v, 1
        return self.latched


def cam_pose(pitch_deg: float, height_m: float):
    p = np.deg2rad(pitch_deg)
    R = np.array([
        [0.0, -np.sin(p), np.cos(p)],
        [-1.0, 0.0, 0.0],
        [0.0, -np.cos(p), -np.sin(p)],
    ])
    t = np.array([0.026, 0.0, height_m])
    return R, t


class Projector:
    """Ground-frame points -> pixel coords of the depth camera."""

    def __init__(self, R, t, fx, fy, cx, cy):
        self.R, self.t = R, t
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy

    def uv(self, pts_g: np.ndarray) -> np.ndarray:
        p = (np.atleast_2d(pts_g) - self.t) @ self.R
        z = np.where(np.abs(p[:, 2]) < 1e-6, np.nan, p[:, 2])
        return np.stack([self.fx * p[:, 0] / z + self.cx,
                         self.fy * p[:, 1] / z + self.cy], axis=1)


class RealsenseSource:
    def __init__(self):
        import pyrealsense2 as rs
        self.rs = rs
        self.pipe = rs.pipeline()
        cfg = rs.config()
        cfg.enable_stream(rs.stream.depth, W, H, rs.format.z16, 15)
        cfg.enable_stream(rs.stream.color, 848, 480, rs.format.rgb8, 15)
        self.profile = self.pipe.start(cfg)
        self.align = rs.align(rs.stream.depth)   # color -> depth frame
        ds = self.profile.get_device().first_depth_sensor()
        self.scale = float(ds.get_depth_scale())
        intr = (self.profile.get_stream(rs.stream.depth)
                .as_video_stream_profile().get_intrinsics())
        self.fx, self.fy, self.cx, self.cy = intr.fx, intr.fy, intr.ppx, intr.ppy
        self.w, self.h = int(intr.width), int(intr.height)
        self.label = "D435 live"

    def frame(self):
        frames = self.align.process(self.pipe.wait_for_frames(timeout_ms=2000))
        d = np.asanyarray(frames.get_depth_frame().get_data()).astype(float)
        d *= self.scale
        d[d <= 0.0] = np.nan
        c = frames.get_color_frame()
        rgb = np.asanyarray(c.get_data()) if c else None
        return d, rgb


class NetSource:
    """Frames from d435_net_server.py running on the camera host
    (Jetson).  See that file for the wire protocol."""

    def __init__(self, host: str, port: int = 5556):
        import json
        import socket
        import struct
        self._struct = struct
        self.sock = socket.create_connection((host, port), timeout=5)
        self.sock.settimeout(5)
        hdr = json.loads(self._recv(
            struct.unpack("<I", self._recv(4))[0]))
        self.w, self.h = int(hdr["w"]), int(hdr["h"])
        self.fx, self.fy = float(hdr["fx"]), float(hdr["fy"])
        self.cx, self.cy = float(hdr["cx"]), float(hdr["cy"])
        self.scale = float(hdr["scale"])
        self.label = f"D435 net @ {host}"

    def _recv(self, n: int) -> bytes:
        buf = b""
        while len(buf) < n:
            chunk = self.sock.recv(n - len(buf))
            if not chunk:
                raise ConnectionError("d435 net server closed")
            buf += chunk
        return buf

    def frame(self):
        import zlib
        import cv2
        nd, nc = self._struct.unpack("<II", self._recv(8))
        d = np.frombuffer(zlib.decompress(self._recv(nd)), np.uint16)
        d = d.reshape(self.h, self.w).astype(float) * self.scale
        d[d <= 0.0] = np.nan
        rgb = None
        if nc:
            bgr = cv2.imdecode(
                np.frombuffer(self._recv(nc), np.uint8), cv2.IMREAD_COLOR)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        return d, rgb


class SyntheticSource:
    """Loop: approach a 2 cm cable, a 12 cm step, a 40 cm wall, then a
    15 deg ramp.

    Renders depth AND a pseudo-RGB (floor gray, obstacle orange, sky
    white) from the same ray cast so the overlay can be exercised.
    """
    SCENES = [
        (0.02, "cable 2cm", "step"),
        (0.12, "step 12cm", "step"),
        (0.40, "wall 40cm", "step"),
        (15.0, "slope 15deg", "slope"),
    ]

    def __init__(self, R, t):
        self.R, self.t = R, t
        self.fx = self.fy = 425.0
        self.cx, self.cy = W / 2.0, H / 2.0
        self.w, self.h = W, H
        self.k = 0
        self.x = 2.0
        self.label = "SYNTHETIC demo"

    def frame(self):
        h, name, kind = self.SCENES[self.k]
        self.label = f"SYNTHETIC: {name} @ {self.x:.2f} m"
        us, vs = np.meshgrid(np.arange(W, dtype=float),
                             np.arange(H, dtype=float))
        v_cam = np.stack([(us - self.cx) / self.fx,
                          (vs - self.cy) / self.fy,
                          np.ones_like(us)], -1)
        v_g = v_cam @ self.R.T
        depth = np.full((H, W), np.nan)
        surf = np.zeros((H, W), dtype=int)          # 0 sky, 1 floor, 2 obst
        dz = v_g[..., 2]
        vx = v_g[..., 0]
        # flat floor in front of the feature
        with np.errstate(divide="ignore", invalid="ignore"):
            zc = (0.0 - self.t[2]) / dz
        xh = self.t[0] + zc * v_g[..., 0]
        take = (zc > 0.05) & np.isfinite(zc) & (xh < self.x)
        depth = np.where(take, zc, depth)
        surf = np.where(take, 1, surf)
        if kind == "slope":
            # ramp z = tan(h deg) * (x - x0) for x >= x0
            tn = np.tan(np.deg2rad(h))
            with np.errstate(divide="ignore", invalid="ignore"):
                zc = ((tn * (self.t[0] - self.x) - self.t[2])
                      / (dz - tn * vx))
            xh = self.t[0] + zc * v_g[..., 0]
            take = ((zc > 0.05) & np.isfinite(zc) & (xh >= self.x)
                    & (~np.isfinite(depth) | (zc < depth)))
            depth = np.where(take, zc, depth)
            surf = np.where(take, 2, surf)
        else:
            # raised plateau at height h behind x0, plus its riser face
            with np.errstate(divide="ignore", invalid="ignore"):
                zc = (h - self.t[2]) / dz
            xh = self.t[0] + zc * v_g[..., 0]
            take = ((zc > 0.05) & np.isfinite(zc) & (xh >= self.x)
                    & (~np.isfinite(depth) | (zc < depth)))
            depth = np.where(take, zc, depth)
            surf = np.where(take, 2, surf)
            with np.errstate(divide="ignore", invalid="ignore"):
                zc = (self.x - self.t[0]) / vx
            zh = self.t[2] + zc * v_g[..., 2]
            take = ((zc > 0.05) & np.isfinite(zc) & (zh >= 0) & (zh <= h)
                    & (~np.isfinite(depth) | (zc < depth)))
            depth = np.where(take, zc, depth)
            surf = np.where(take, 2, surf)
        # pseudo-RGB with a hint of distance shading
        shade = np.clip(1.0 - 0.18 * np.nan_to_num(depth, nan=0.0), 0.35, 1.0)
        rgb = np.empty((H, W, 3))
        rgb[surf == 0] = (0.97, 0.97, 1.0)
        rgb[surf == 1] = (0.72, 0.72, 0.70)
        rgb[surf == 2] = (0.90, 0.55, 0.20)
        rgb *= shade[..., None]
        rgb = (rgb * 255).astype(np.uint8)
        self.x -= 0.04
        if self.x < 0.45:
            self.x = 2.0
            self.k = (self.k + 1) % len(self.SCENES)
        return depth, rgb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--demo", action="store_true", help="force synthetic")
    ap.add_argument("--net", metavar="HOST", default=None,
                    help="stream frames from d435_net_server.py on HOST "
                         "(e.g. the Jetson: --net 192.168.1.100)")
    ap.add_argument("--full", action="store_true",
                    help="3-mode WHEEL/HOP/STOP gate with fixed mount pose")
    ap.add_argument("--fixed-pose", action="store_true",
                    help="flat check but trust --pitch/--cam-height instead "
                         "of auto plane fit")
    ap.add_argument("--pitch", type=float, default=30.0)
    ap.add_argument("--cam-height", type=float, default=0.367)
    ap.add_argument("--headless", action="store_true")
    ap.add_argument("--frames", type=int, default=0, help="stop after N (0=run)")
    ap.add_argument("--rotate", type=int, default=90, choices=(0, 90, 180, 270),
                    help="clockwise display rotation in degrees (default 90)")
    ap.add_argument("--no-button-tag", action="store_true",
                    help="disable AprilTag button overlay/setpoint publisher")
    ap.add_argument("--tag-id", type=int, default=1)
    ap.add_argument("--tag-size", type=float, default=0.09)
    ap.add_argument("--button-right-m", type=float, default=0.165)
    ap.add_argument("--button-down-m", type=float, default=0.026)
    ap.add_argument("--button-protrude-m", type=float, default=0.05)
    ap.add_argument("--button-press-m", type=float, default=0.01)
    ap.add_argument(
        "--button-setpoint",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "logs" / "button_setpoint.json",
    )
    ap.add_argument(
        "--button-setpoint-port", type=int, default=5558,
        help="UDP port used to send AprilTag targets to the Jetson controller",
    )
    ap.add_argument(
        "--dashboard-status",
        type=Path,
        default=Path(__file__).resolve().parent.parent
        / "logs" / "dashboard_status.json",
        help="controller gait/leg status JSON",
    )
    ap.add_argument(
        "--dashboard-status-port", type=int, default=5557,
        help="UDP port for live controller status from Jetson",
    )
    ap.add_argument("--no-record", action="store_true",
                    help="disable automatic MANIPULATION dashboard video")
    ap.add_argument("--record-fps", type=float, default=10.0)
    args = ap.parse_args()
    flat_mode = not args.full
    auto_ground = flat_mode and not args.fixed_pose
    rot_cw = int(args.rotate)

    R, t = cam_pose(args.pitch, args.cam_height)
    src = None
    if args.net:
        src = NetSource(args.net)
        print(f"[live] using D435 stream from {args.net}")
    elif not args.demo:
        try:
            src = RealsenseSource()
            print("[live] using RealSense D435")
        except Exception as e:
            print(f"[live] no camera ({e}); falling back to synthetic demo")
    if src is None:
        src = SyntheticSource(R, t)

    import matplotlib
    if args.headless:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    gate = TerrainGate(TerrainGateConfig())
    cfg = gate.cfg
    proj = Projector(R, t, src.fx, src.fy, src.cx, src.cy)

    # Optional button detector shares the same RGB frame and publishes the
    # exact JSON consumed by MANIPULATION. The terrain corridor and button
    # approach path therefore appear in one dashboard/window.
    button_detector = None
    button_T_L_C = None
    button_geometry = None
    button_last_write = 0.0
    button_targets_last = None
    button_udp_sock = None
    if not args.no_button_tag:
        try:
            from pupil_apriltags import Detector
            from button_apriltag_geometry import (
                button_targets_from_detection,
                load_T_L_C,
            )
            button_detector = Detector(
                families="tag36h11", nthreads=2, quad_decimate=1.0
            )
            button_T_L_C = load_T_L_C()
            button_geometry = button_targets_from_detection
            button_udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            args.button_setpoint.parent.mkdir(parents=True, exist_ok=True)
            print(
                "[button] dashboard enabled: "
                f"id={args.tag_id}, size={args.tag_size*1000:.0f} mm, "
                f"right={args.button_right_m*100:.1f} cm, "
                f"down={args.button_down_m*100:.1f} cm"
            )
        except Exception as e:
            print(f"[button] disabled ({e})")

    plt.ion()
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(17, 5.2))
    if not args.headless:
        fig.canvas.manager.set_window_title("Terrain Gate LIVE")

    sw, sh = src.w, src.h
    # Display size after CW rotation (depth/RANSAC stay in camera frame).
    if rot_cw % 180 == 90:
        dw, dh = sh, sw
    else:
        dw, dh = sw, sh
    im_rgb = ax0.imshow(np.zeros((dh, dw, 3), dtype=np.uint8))
    ax0.set_title(f"RGB scene (annotated, rot {rot_cw} CW)")
    ax0.axis("off")
    # corridor footprint on the ground (auto-ground: re-projected per frame)
    corridor_lns = [ax0.plot([], [], color="#2ca02c", lw=1.6, alpha=0.9)[0]
                    for _ in range(4)]

    def draw_corridor():
        xs = np.linspace(cfg.range_min_m, cfg.range_max_m, 24)
        for i, y in enumerate((-cfg.corridor_half_width_m,
                               cfg.corridor_half_width_m)):
            uv = _rot_uv(proj.uv(np.stack([xs, np.full_like(xs, y),
                                           np.zeros_like(xs)], 1)),
                         sw, sh, rot_cw)
            corridor_lns[i].set_data(uv[:, 0], uv[:, 1])
        for i, x in enumerate((cfg.range_min_m, cfg.range_max_m)):
            uv = _rot_uv(proj.uv(np.array([[x, -cfg.corridor_half_width_m, 0.0],
                                           [x, cfg.corridor_half_width_m, 0.0]])),
                         sw, sh, rot_cw)
            corridor_lns[2 + i].set_data(uv[:, 0], uv[:, 1])

    draw_corridor()
    (step_ln,) = ax0.plot([], [], lw=3.0, color="crimson")
    step_txt = ax0.text(0, 0, "", color="white", fontsize=10,
                        fontweight="bold",
                        bbox=dict(boxstyle="round", fc="crimson", ec="none",
                                  alpha=0.85))
    step_txt.set_visible(False)
    ax0.set_xlim(0, dw)
    ax0.set_ylim(dh, 0)

    im_d = ax1.imshow(np.zeros((dh, dw)), vmin=0.2, vmax=2.5, cmap="turbo")
    ax1.set_title(f"depth (m, rot {rot_cw} CW)")
    ax1.axis("off")

    (ln,) = ax2.plot([], [], "o-", ms=3, lw=1.2, color="tab:blue")
    thr_name = "flat" if flat_mode else "wheel"
    ax2.axhline(cfg.step_wheel_max_m * 100, color="gray", ls=":",
                label=f"{thr_name} step max {cfg.step_wheel_max_m*100:.0f} cm")
    ax2.set_xlim(cfg.range_min_m, cfg.range_max_m)
    ax2.set_ylim(-8, 45)
    ax2.set_xlabel("distance ahead (m)")
    ax2.set_ylabel("height (cm)")
    ax2.grid(True, alpha=0.3)
    ax2.legend(loc="upper left", fontsize=8)
    banner = fig.suptitle("...", fontsize=16, fontweight="bold")
    button_txt = ax0.text(
        0.02, 0.98, "BUTTON TAG: waiting",
        transform=ax0.transAxes, va="top", ha="left",
        color="white", fontsize=9, family="monospace",
        bbox=dict(boxstyle="round", fc="black", ec="white", alpha=0.72),
    )

    # Alternative right-hand panels shown only while the controller reports
    # MANIPULATION. They occupy the same slots as depth/profile.
    ax_leg_cmd = fig.add_axes(ax1.get_position(), visible=False)
    ax_leg_path = fig.add_axes(ax2.get_position(), visible=False)
    cmd_lines = [
        ax_leg_cmd.plot([], [], lw=1.5, label=f"cmd {axis}")[0]
        for axis in ("X", "Y", "Z")
    ]
    actual_lines = [
        ax_leg_cmd.plot(
            [], [], lw=1.0, ls="--", alpha=0.7, label=f"actual {axis}"
        )[0]
        for axis in ("X", "Y", "Z")
    ]
    ax_leg_cmd.set_title("Leg Cartesian cmd / actual")
    ax_leg_cmd.set_xlabel("time (s)")
    ax_leg_cmd.set_ylabel("foot coordinate in L frame (m)")
    ax_leg_cmd.grid(True, alpha=0.3)
    ax_leg_cmd.legend(fontsize=7, ncol=2, loc="best")

    (planned_path_ln,) = ax_leg_path.plot(
        [], [], "o-", lw=2.0, label="PRE → FACE → PRESS"
    )
    (cmd_point_ln,) = ax_leg_path.plot(
        [], [], "o", ms=9, label="current cmd"
    )
    (actual_point_ln,) = ax_leg_path.plot(
        [], [], "x", ms=9, mew=2, label="actual foot"
    )
    ax_leg_path.set_title("Leg command path (X-Z)")
    ax_leg_path.set_xlabel("foot X_L (m)")
    ax_leg_path.set_ylabel("foot Z_L (m)")
    ax_leg_path.grid(True, alpha=0.3)
    ax_leg_path.legend(fontsize=7, loc="best")
    leg_info_txt = ax_leg_path.text(
        0.02, 0.98, "",
        transform=ax_leg_path.transAxes, va="top", ha="left",
        fontsize=8, family="monospace",
        bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.75),
    )
    leg_t_hist = deque(maxlen=300)
    leg_cmd_hist = [deque(maxlen=300) for _ in range(3)]
    leg_actual_hist = [deque(maxlen=300) for _ in range(3)]
    leg_status_last_stamp = -1.0
    manip_active_prev = False
    video_writer = None
    video_path = None
    video_t0 = 0.0
    video_frames = 0
    controller_status_net = None
    controller_status_rx_t = float("-inf")
    status_sock = None
    try:
        status_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        status_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        status_sock.bind(("", int(args.dashboard_status_port)))
        status_sock.setblocking(False)
        print(
            f"[status] listening on UDP :{args.dashboard_status_port} "
            "for Jetson controller"
        )
    except OSError as exc:
        print(f"[status] UDP unavailable ({exc}); using local JSON fallback")
        if status_sock is not None:
            status_sock.close()
            status_sock = None

    n = 0
    mode = "STOP"
    verdict = "NO GROUND"
    deb = Debounce(cfg.hysteresis_n)
    # Slope stays on even with auto-ground: RANSAC levels the DOMINANT
    # plane, so a ramp entering the corridor still tilts the profile.
    # Only a slope filling the whole view reads as flat (needs IMU/pose).
    slope_valid = True
    t_last = time.time()
    try:
        while True:
            d, rgb = src.frame()
            if auto_ground:
                p_cam = depth_to_ground_points(
                    d, src.fx, src.fy, src.cx, src.cy,
                    np.eye(3), np.zeros(3), stride=6)
                try:
                    sub = p_cam[::max(1, len(p_cam) // 4000)]
                    nrm, dd = fit_dominant_plane(sub)
                    Rg = plane_frame(nrm)
                    pts = p_cam @ Rg.T
                    pts[:, 2] += dd
                    proj.R, proj.t = Rg, np.array([0.0, 0.0, dd])
                    draw_corridor()
                except ValueError:
                    pts = np.empty((0, 3))
            else:
                pts = depth_to_ground_points(d, src.fx, src.fy, src.cx,
                                             src.cy, R, t, stride=6)
            mode, vote = gate.update(pts)
            raw = describe_vote(vote, cfg, slope_valid)
            verdict = deb.update(raw)

            if rgb is not None:
                rgb_annotated = np.ascontiguousarray(rgb.copy())
                button_status = "BUTTON TAG: waiting"
                if button_detector is not None and button_geometry is not None:
                    try:
                        gray = cv2.cvtColor(rgb_annotated, cv2.COLOR_RGB2GRAY)
                        detections = button_detector.detect(
                            gray,
                            estimate_tag_pose=True,
                            camera_params=(src.fx, src.fy, src.cx, src.cy),
                            tag_size=float(args.tag_size),
                        )
                        chosen = next(
                            (
                                det for det in detections
                                if int(det.tag_id) == int(args.tag_id)
                            ),
                            None,
                        )
                        if chosen is not None:
                            pose_R = np.asarray(
                                chosen.pose_R, dtype=float
                            ).reshape(3, 3)
                            pose_t = np.asarray(
                                chosen.pose_t, dtype=float
                            ).reshape(3)
                            targets = button_geometry(
                                pose_R=pose_R,
                                pose_t=pose_t,
                                T_L_C=button_T_L_C,
                                right_m=float(args.button_right_m),
                                down_m=float(args.button_down_m),
                                protrude_m=float(args.button_protrude_m),
                                press_m=float(args.button_press_m),
                            )
                            button_targets_last = targets

                            # RGB overlay: tag outline plus PRE -> FACE ->
                            # PRESS path. Colors are RGB tuples (the array is
                            # displayed directly by matplotlib).
                            corners = np.rint(chosen.corners).astype(np.int32)
                            cv2.polylines(
                                rgb_annotated, [corners], True,
                                (255, 230, 0), 2, cv2.LINE_AA,
                            )

                            def project_button_point(p_cam):
                                p_cam = np.asarray(
                                    p_cam, dtype=float
                                ).reshape(3)
                                if float(p_cam[2]) <= 1e-4:
                                    return None
                                return (
                                    int(round(
                                        src.fx * float(p_cam[0])
                                        / float(p_cam[2]) + src.cx
                                    )),
                                    int(round(
                                        src.fy * float(p_cam[1])
                                        / float(p_cam[2]) + src.cy
                                    )),
                                )

                            p_pre = project_button_point(
                                targets["camera"]["pre"]
                            )
                            p_face = project_button_point(
                                targets["camera"]["face"]
                            )
                            p_press = project_button_point(
                                targets["camera"]["press"]
                            )
                            path_px = [
                                p for p in (p_pre, p_face, p_press)
                                if p is not None
                            ]
                            if len(path_px) >= 2:
                                cv2.polylines(
                                    rgb_annotated,
                                    [np.asarray(path_px, dtype=np.int32)],
                                    False, (0, 255, 255), 3, cv2.LINE_AA,
                                )
                            if p_pre is not None:
                                cv2.circle(
                                    rgb_annotated, p_pre, 14,
                                    (0, 255, 255), 2, cv2.LINE_AA,
                                )
                            if p_face is not None:
                                cv2.circle(
                                    rgb_annotated, p_face, 10,
                                    (255, 165, 0), 2, cv2.LINE_AA,
                                )
                            if p_press is not None:
                                cv2.circle(
                                    rgb_annotated, p_press, 6,
                                    (255, 0, 0), -1, cv2.LINE_AA,
                                )

                            face_L = targets["leg"]["face"]
                            press_L = targets["leg"]["press"]
                            pre_L = targets["leg"]["pre"]
                            n_in = targets["wall_normal_in_L"]
                            button_status = (
                                f"TAG {int(chosen.tag_id)} LOCKED  "
                                f"z={float(pose_t[2]):.3f}m\n"
                                f"PRE   {pre_L[0]:+.3f} "
                                f"{pre_L[1]:+.3f} {pre_L[2]:+.3f}\n"
                                f"FACE  {face_L[0]:+.3f} "
                                f"{face_L[1]:+.3f} {face_L[2]:+.3f}\n"
                                f"PRESS {press_L[0]:+.3f} "
                                f"{press_L[1]:+.3f} {press_L[2]:+.3f}\n"
                                f"N_IN  {n_in[0]:+.2f} "
                                f"{n_in[1]:+.2f} {n_in[2]:+.2f}"
                            )

                            now_button = time.time()
                            if now_button - button_last_write > 0.05:
                                payload = {
                                    "t_wall": now_button,
                                    "tag_id": int(chosen.tag_id),
                                    "tag_size_m": float(args.tag_size),
                                    "right_m": float(args.button_right_m),
                                    "down_m": float(args.button_down_m),
                                    "protrude_m": float(
                                        args.button_protrude_m
                                    ),
                                    "press_m": float(args.button_press_m),
                                    "foot_pre_L": [
                                        float(x) for x in pre_L
                                    ],
                                    "foot_face_L": [
                                        float(x) for x in face_L
                                    ],
                                    "foot_press_L": [
                                        float(x) for x in press_L
                                    ],
                                    "wall_normal_in_L": [
                                        float(x) for x in n_in
                                    ],
                                    "wall_normal_out_L": [
                                        float(x) for x
                                        in targets["wall_normal_out_L"]
                                    ],
                                    "press_along_wall_normal": True,
                                    "valid": True,
                                }
                                tmp = args.button_setpoint.with_suffix(
                                    ".json.tmp"
                                )
                                encoded_payload = json.dumps(payload)
                                tmp.write_text(
                                    json.dumps(payload, indent=2)
                                )
                                tmp.replace(args.button_setpoint)
                                button_udp_sock.sendto(
                                    encoded_payload.encode("utf-8"),
                                    (
                                        args.net or "127.0.0.1",
                                        int(args.button_setpoint_port),
                                    ),
                                )
                                button_last_write = now_button
                        else:
                            button_status = (
                                f"BUTTON TAG {args.tag_id}: NOT FOUND\n"
                                "cyan=PRE orange=FACE red=PRESS"
                            )
                    except Exception as e:
                        button_status = f"BUTTON DETECTOR ERROR\n{e}"

                button_txt.set_text(button_status)
                im_rgb.set_data(_rot_img(rgb_annotated, rot_cw))
            im_d.set_data(_rot_img(np.where(np.isfinite(d), d, 0.0), rot_cw))
            if vote.profile_x is not None and vote.profile_z is not None:
                ln.set_data(vote.profile_x, vote.profile_z * 100.0)

            # annotate the detected step on the RGB scene
            tag = raw if flat_mode else vote.mode
            col = (VERDICT_COLOR.get(raw, "black") if flat_mode
                   else MODE_COLOR.get(vote.mode, "black"))
            if (np.isfinite(vote.step_dist_m)
                    and vote.step_m > cfg.step_wheel_max_m):
                yy = np.linspace(-cfg.corridor_half_width_m,
                                 cfg.corridor_half_width_m, 12)
                uv = _rot_uv(proj.uv(np.stack(
                    [np.full_like(yy, vote.step_dist_m),
                     yy, np.zeros_like(yy)], 1)), sw, sh, rot_cw)
                step_ln.set_data(uv[:, 0], uv[:, 1])
                step_ln.set_color(col)
                u0, v0 = np.nanmean(uv[:, 0]), np.nanmin(uv[:, 1])
                step_txt.set_position((u0, max(18.0, v0 - 14.0)))
                step_txt.set_text(
                    f"{tag}: step {vote.step_m*100:.0f} cm "
                    f"@ {vote.step_dist_m:.2f} m"
                )
                step_txt.get_bbox_patch().set_facecolor(col)
                step_txt.set_visible(True)
            else:
                step_ln.set_data([], [])
                step_txt.set_visible(False)

            hz = 1.0 / max(1e-3, time.time() - t_last)
            t_last = time.time()
            if flat_mode:
                slope_s = (f" | slope {vote.slope_deg:.1f} deg"
                           if slope_valid else "")
                banner.set_text(
                    f"{verdict}   (step {vote.step_m*100:.1f} cm"
                    f"{slope_s}"
                    f" | rough {vote.rough_m*1000:.0f} mm"
                    f" | bins {vote.n_valid_bins}"
                    f" | {hz:.1f} Hz)   [{src.label}]"
                )
                banner.set_color(VERDICT_COLOR.get(verdict, "black"))
            else:
                banner.set_text(
                    f"{mode}   (vote {vote.mode}"
                    f" | step {vote.step_m*100:.1f} cm"
                    f" | slope {vote.slope_deg:.1f} deg"
                    f" | rough {vote.rough_m*1000:.0f} mm"
                    f" | steps {vote.n_steps} | {hz:.1f} Hz)   [{src.label}]"
                )
                banner.set_color(MODE_COLOR.get(mode, "black"))

            # Controller-aware panel switch. HOPPING/MOBILE retain depth and
            # terrain profile; MANIPULATION replaces those two panels with
            # leg command history and the Cartesian approach path.
            controller_status = None
            controller_status_age = float("inf")
            if status_sock is not None:
                while True:
                    try:
                        raw_status, _peer = status_sock.recvfrom(65535)
                    except BlockingIOError:
                        break
                    except OSError:
                        break
                    try:
                        controller_status_net = json.loads(
                            raw_status.decode("utf-8")
                        )
                        controller_status_rx_t = time.monotonic()
                    except (UnicodeDecodeError, json.JSONDecodeError):
                        pass
            net_age = time.monotonic() - controller_status_rx_t
            if controller_status_net is not None and net_age <= 0.5:
                controller_status = controller_status_net
                controller_status_age = net_age
            try:
                if controller_status is None:
                    controller_status = json.loads(
                        args.dashboard_status.read_text()
                    )
                    controller_status_age = (
                        time.time()
                        - float(controller_status.get("t_wall", 0.0))
                    )
                    if controller_status_age > 0.5:
                        controller_status = None
            except Exception:
                controller_status = None
            if controller_status is None:
                age_text = (
                    "missing"
                    if not np.isfinite(controller_status_age)
                    else f"stale {controller_status_age:.1f}s"
                )
                button_txt.set_text(
                    button_txt.get_text()
                    + f"\nCTRL: OFFLINE/{age_text} — restart run_modee"
                )
            manip_active = bool(
                controller_status is not None
                and controller_status.get("gait_mode") == "manipulation"
            )
            if (
                controller_status is not None
                and controller_status.get("gait_mode") == "mobile"
            ):
                mobile_tag_state = str(
                    controller_status.get("mobile_tag_state", "searching")
                ).upper()
                reach_err = controller_status.get(
                    "mobile_tag_reach_error_m"
                )
                reach_text = (
                    ""
                    if reach_err is None
                    else f" clip={float(reach_err)*100:.1f}cm"
                )
                button_txt.set_text(
                    button_txt.get_text()
                    + f"\nCTRL MOBILE: {mobile_tag_state}{reach_text}"
                )
            ax1.set_visible(not manip_active)
            ax2.set_visible(not manip_active)
            ax_leg_cmd.set_visible(manip_active)
            ax_leg_path.set_visible(manip_active)

            if manip_active and controller_status is not None:
                if not manip_active_prev:
                    leg_t_hist.clear()
                    for hist in leg_cmd_hist + leg_actual_hist:
                        hist.clear()
                    leg_status_last_stamp = -1.0
                stamp = float(controller_status.get("t_wall", time.time()))
                cmd = np.asarray(
                    controller_status.get(
                        "foot_cmd_L", [np.nan, np.nan, np.nan]
                    ),
                    dtype=float,
                ).reshape(3)
                actual_raw = controller_status.get(
                    "foot_actual_L", [None, None, None]
                )
                actual = np.asarray(
                    [
                        np.nan if value is None else float(value)
                        for value in actual_raw
                    ],
                    dtype=float,
                ).reshape(3)
                if stamp > leg_status_last_stamp + 1e-6:
                    leg_status_last_stamp = stamp
                    leg_t_hist.append(stamp)
                    for axis in range(3):
                        leg_cmd_hist[axis].append(float(cmd[axis]))
                        leg_actual_hist[axis].append(float(actual[axis]))

                if leg_t_hist:
                    t_rel = np.asarray(leg_t_hist) - float(leg_t_hist[0])
                    for axis in range(3):
                        cmd_lines[axis].set_data(
                            t_rel, np.asarray(leg_cmd_hist[axis])
                        )
                        actual_lines[axis].set_data(
                            t_rel, np.asarray(leg_actual_hist[axis])
                        )
                    ax_leg_cmd.relim()
                    ax_leg_cmd.autoscale_view()

                if button_targets_last is not None:
                    leg_pts = button_targets_last["leg"]
                    route = np.asarray(
                        [
                            leg_pts["pre"],
                            leg_pts["face"],
                            leg_pts["press"],
                        ],
                        dtype=float,
                    ).reshape(3, 3)
                    planned_path_ln.set_data(route[:, 0], route[:, 2])
                cmd_point_ln.set_data([cmd[0]], [cmd[2]])
                if np.all(np.isfinite(actual)):
                    actual_point_ln.set_data([actual[0]], [actual[2]])
                else:
                    actual_point_ln.set_data([], [])
                ax_leg_path.relim()
                ax_leg_path.autoscale_view()
                # Keep a useful minimum view even before the route arrives.
                if np.isfinite(cmd[0]) and np.isfinite(cmd[2]):
                    ax_leg_path.set_xlim(cmd[0] - 0.08, cmd[0] + 0.08)
                    ax_leg_path.set_ylim(cmd[2] - 0.08, cmd[2] + 0.08)

                tau = np.asarray(
                    controller_status.get("tau_cmd", [0.0, 0.0, 0.0]),
                    dtype=float,
                ).reshape(3)
                stage = str(controller_status.get("button_stage", "?"))
                err = controller_status.get("manip_err_m")
                err_text = (
                    "nan" if err is None else f"{float(err)*1000:.1f}mm"
                )
                leg_info_txt.set_text(
                    f"stage={stage}\n"
                    f"cmd={cmd[0]:+.3f},{cmd[1]:+.3f},{cmd[2]:+.3f}\n"
                    f"err={err_text}\n"
                    f"tau={tau[0]:+.2f},{tau[1]:+.2f},{tau[2]:+.2f} Nm"
                )
                banner.set_text(
                    f"MANIPULATION  stage={stage}  err={err_text}  "
                    f"|tau|max={np.max(np.abs(tau)):.2f} Nm  "
                    f"[{src.label}]"
                )
                banner.set_color("tab:purple")
            manip_active_prev = manip_active

            need_canvas_frame = bool(
                manip_active and not args.no_record
            )
            if not args.headless or need_canvas_frame:
                fig.canvas.draw()
            if not args.headless:
                fig.canvas.flush_events()
                plt.pause(0.001)

            # Automatically record the complete switched dashboard while
            # MANIPULATION is active. Frame duplication keeps real-time video
            # duration correct even if terrain processing has variable Hz.
            if need_canvas_frame:
                if video_writer is None:
                    videos_dir = (
                        Path(__file__).resolve().parent.parent
                        / "logs" / "videos"
                    )
                    videos_dir.mkdir(parents=True, exist_ok=True)
                    video_path = videos_dir / (
                        "manip_dashboard_"
                        + time.strftime("%Y%m%d_%H%M%S")
                        + ".mp4"
                    )
                    width, height = fig.canvas.get_width_height()
                    fps_record = float(max(1.0, args.record_fps))
                    video_writer = cv2.VideoWriter(
                        str(video_path),
                        cv2.VideoWriter_fourcc(*"mp4v"),
                        fps_record,
                        (int(width), int(height)),
                    )
                    if not video_writer.isOpened():
                        print(f"[video] failed to open {video_path}")
                        video_writer.release()
                        video_writer = None
                        video_path = None
                    else:
                        video_t0 = time.time()
                        video_frames = 0
                        print(f"[video] MANIPULATION -> {video_path}")
                if video_writer is not None:
                    rgba = np.asarray(fig.canvas.buffer_rgba())
                    frame_bgr = cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
                    fps_record = float(max(1.0, args.record_fps))
                    due = max(
                        1, int((time.time() - video_t0) * fps_record) + 1
                    )
                    while video_frames < due:
                        video_writer.write(frame_bgr)
                        video_frames += 1
            elif video_writer is not None:
                video_writer.release()
                print(
                    f"[video] saved {video_frames} frames -> {video_path}"
                )
                video_writer = None
                video_path = None
                video_frames = 0

            n += 1
            if args.frames and n >= args.frames:
                break
    except KeyboardInterrupt:
        pass
    if video_writer is not None:
        video_writer.release()
        print(f"[video] saved {video_frames} frames -> {video_path}")
    if status_sock is not None:
        status_sock.close()
    if button_udp_sock is not None:
        button_udp_sock.close()
    if args.headless and n:
        out = Path(__file__).resolve().parent.parent / "logs" / "figs" \
            / "terrain_gate_live_snapshot.png"
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=110)
        print(f"[live] snapshot -> {out}")
    if flat_mode:
        print(f"[live] {n} frames, last verdict = {verdict} (gate {mode})")
    else:
        print(f"[live] {n} frames, last mode = {mode}")


if __name__ == "__main__":
    main()
