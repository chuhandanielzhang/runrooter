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
import sys
import time
from pathlib import Path

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
                im_rgb.set_data(_rot_img(rgb, rot_cw))
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
            if not args.headless:
                fig.canvas.draw_idle()
                fig.canvas.flush_events()
                plt.pause(0.001)
            n += 1
            if args.frames and n >= args.frames:
                break
    except KeyboardInterrupt:
        pass
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
