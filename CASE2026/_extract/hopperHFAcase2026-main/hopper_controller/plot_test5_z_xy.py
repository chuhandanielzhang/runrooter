#!/usr/bin/env python3
"""Plot z vs time and XY. Height: stance = kinematic z (R_wb@foot_b), flight = p_hat_w2."""
import csv, os
import matplotlib.pyplot as plt
import numpy as np

_LOGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
LOG = os.path.join(_LOGDIR, "test2_1.csv")
OUT = os.path.join(_LOGDIR, "test2_1_z_xy.png")

def quat_to_R(w, x, y, z):
    """Body-to-world rotation from quat (w,x,y,z)."""
    w, x, y, z = float(w), float(x), float(y), float(z)
    n = w * w + x * x + y * y + z * z
    if n < 1e-12:
        return np.eye(3)
    s = 2.0 / n
    return np.array([
        [1 - s * (y * y + z * z), s * (x * y - w * z), s * (x * z + w * y)],
        [s * (x * y + w * z), 1 - s * (x * x + z * z), s * (y * z - w * x)],
        [s * (x * z - w * y), s * (y * z + w * x), 1 - s * (x * x + y * y)],
    ], dtype=float)

def main():
    with open(LOG) as f:
        r = csv.reader(f)
        header = next(r)
        rows = list(r)

    idx = {h: i for i, h in enumerate(header)}
    i_t = idx["t_s"]
    i_z = idx["p_hat_w2"]
    i_x = idx["p_hat_w0"]
    i_y = idx["p_hat_w1"]
    i_stance = idx.get("stance", None)
    i_fb0 = idx.get("foot_b0", None)
    i_fb1 = idx.get("foot_b1", None)
    i_fb2 = idx.get("foot_b2", None)
    i_qw = idx.get("imu_quat_w", None)
    i_qx = idx.get("imu_quat_x", None)
    i_qy = idx.get("imu_quat_y", None)
    i_qz = idx.get("imu_quat_z", None)
    use_kinematic = all(k is not None for k in (i_stance, i_fb0, i_fb1, i_fb2, i_qw, i_qx, i_qy, i_qz))
    need = max(i_t, i_z, i_x, i_y) + 1
    if use_kinematic:
        need = max(need, i_stance, i_fb0, i_fb1, i_fb2, i_qw, i_qx, i_qy, i_qz) + 1

    t, z, x, y = [], [], [], []
    for row in rows:
        if len(row) < need:
            continue
        try:
            ti = float(row[i_t])
            xi = float(row[i_x])
            yi = float(row[i_y])
            zi_est = float(row[i_z])
        except (ValueError, TypeError):
            continue
        if use_kinematic:
            try:
                stance = int(float(row[i_stance]))
                foot_b = np.array([float(row[i_fb0]), float(row[i_fb1]), float(row[i_fb2])])
                qw, qx, qy, qz = float(row[i_qw]), float(row[i_qx]), float(row[i_qy]), float(row[i_qz])
            except (ValueError, TypeError, IndexError):
                zi = zi_est
            else:
                R = quat_to_R(qw, qx, qy, qz)
                # z = body height above foot in world; when stance, foot on ground => body height
                z_kinematic = -float((R @ foot_b)[2])
                zi = z_kinematic if stance == 1 else zi_est
        else:
            zi = zi_est
        t.append(ti)
        z.append(zi)
        x.append(xi)
        y.append(yi)
    t = np.array(t)
    z = np.array(z)
    x = np.array(x)
    y = np.array(y)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(t, z, color="C0", linewidth=0.8)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("z (m)")
    ax1.set_title("Height vs time (stance=kinematic, flight=est)" if use_kinematic else "Height vs time (test2.1)")
    ax1.grid(True, alpha=0.3)

    ax2.plot(x, y, color="C1", linewidth=0.8)
    ax2.set_xlabel("x (m)")
    ax2.set_ylabel("y (m)")
    ax2.set_title("XY trajectory (test2.1)")
    ax2.set_aspect("equal")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUT, dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved:", OUT)

if __name__ == "__main__":
    main()
