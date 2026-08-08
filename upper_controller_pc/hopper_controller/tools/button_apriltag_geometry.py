#!/usr/bin/env python3
"""Wall-button pose from a wall AprilTag (tag36h11).

Tag frame (pupil_apriltags / OpenCV): X right on the tag face, Y down,
Z out of the wall toward the camera.

Geometry from wall photo (2026-08-08): 90 mm tag id=1, measured in-image
against tag side length:
  - button center ≈ 16.5 cm RIGHT of tag center
  - button center ≈ 2.6 cm DOWN of tag center
  - button face protrudes 5 cm from the wall (user)
  - press stroke is 1 cm into the wall (along -Z_tag)

So in tag coordinates:
  p_face  = (+0.165, +0.026, +0.05)
  p_press = (+0.165, +0.026, +0.04)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

# Measured from camera photo vs 90 mm tag (2026-08-08).
DEFAULT_RIGHT_M = 0.165
DEFAULT_DOWN_M = 0.026
DEFAULT_PROTRUDE_M = 0.05
DEFAULT_PRESS_M = 0.01

# Printed wall tag (user 2026-08-08): 90 mm black detection square, id=1.
DEFAULT_WALL_TAG_SIZE_M = 0.09
DEFAULT_WALL_TAG_ID = 1

CALIB_DEFAULT = (
    Path(__file__).resolve().parent / "apriltags_print" / "calib" / "T_L_C.json"
)


def load_T_L_C(path: Path | str | None = None) -> np.ndarray:
    p = Path(path) if path is not None else CALIB_DEFAULT
    data = json.loads(p.read_text())
    if "T_L_C" in data:
        T = np.asarray(data["T_L_C"], dtype=float).reshape(4, 4)
    else:
        R = np.asarray(data["R"], dtype=float).reshape(3, 3)
        t = np.asarray(data["t_m"], dtype=float).reshape(3)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
    return T


def button_points_in_tag(
    *,
    right_m: float = DEFAULT_RIGHT_M,
    down_m: float = DEFAULT_DOWN_M,
    protrude_m: float = DEFAULT_PROTRUDE_M,
    press_m: float = DEFAULT_PRESS_M,
) -> dict[str, np.ndarray]:
    """Return face / press / pre-approach points in the tag frame."""
    right_m = float(right_m)
    down_m = float(down_m)
    protrude_m = float(protrude_m)
    press_m = float(press_m)
    face = np.array([right_m, down_m, protrude_m], dtype=float)
    press = np.array([right_m, down_m, protrude_m - press_m], dtype=float)
    # Hover 3 cm in front of the button face before committing to press.
    pre = np.array([right_m, down_m, protrude_m + 0.03], dtype=float)
    return {"face": face, "press": press, "pre": pre, "tag_center": np.zeros(3)}


def transform_points(T_dst_src: np.ndarray, pts: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    T = np.asarray(T_dst_src, dtype=float).reshape(4, 4)
    out = {}
    for k, p in pts.items():
        ph = np.array([p[0], p[1], p[2], 1.0], dtype=float)
        out[k] = (T @ ph)[:3].astype(float)
    return out


def tag_pose_to_T_cam_tag(pose_R: np.ndarray, pose_t: np.ndarray) -> np.ndarray:
    """Build T_cam_tag from detector pose (p_cam = R @ p_tag + t)."""
    T = np.eye(4)
    T[:3, :3] = np.asarray(pose_R, dtype=float).reshape(3, 3)
    T[:3, 3] = np.asarray(pose_t, dtype=float).reshape(3)
    return T


def button_targets_from_detection(
    *,
    pose_R: np.ndarray,
    pose_t: np.ndarray,
    T_L_C: np.ndarray,
    right_m: float = DEFAULT_RIGHT_M,
    down_m: float = DEFAULT_DOWN_M,
    protrude_m: float = DEFAULT_PROTRUDE_M,
    press_m: float = DEFAULT_PRESS_M,
) -> dict:
    """Camera + leg-frame button targets from one tag pose."""
    pts_tag = button_points_in_tag(
        right_m=right_m,
        down_m=down_m,
        protrude_m=protrude_m,
        press_m=press_m,
    )
    T_C_tag = tag_pose_to_T_cam_tag(pose_R, pose_t)
    pts_C = transform_points(T_C_tag, pts_tag)
    pts_L = transform_points(T_L_C, pts_C)
    T_L_tag = (np.asarray(T_L_C, dtype=float).reshape(4, 4) @ T_C_tag)
    # Tag +Z points out of the wall (toward camera). Press into wall = -Z.
    n_out_L = T_L_tag[:3, 2].astype(float)
    n_out_L = n_out_L / max(1e-9, float(np.linalg.norm(n_out_L)))
    n_in_L = (-n_out_L).astype(float)
    return {
        "tag": pts_tag,
        "camera": pts_C,
        "leg": pts_L,
        "T_C_tag": T_C_tag,
        "T_L_tag": T_L_tag,
        "wall_normal_out_L": n_out_L,
        "wall_normal_in_L": n_in_L,
    }
