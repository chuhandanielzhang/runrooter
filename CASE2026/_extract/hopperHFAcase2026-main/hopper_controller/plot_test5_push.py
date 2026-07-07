#!/usr/bin/env python3
"""Generate push-recovery figures from test5 CSV."""
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LOG = "/home/abc/hopper_logs/modee_csv/test5_20260224_0350.csv"
OUT_DIR = "/home/abc/hopper_logs/modee_csv/paper_figs"
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "legend.fontsize": 7,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 300,
    "lines.linewidth": 0.70,
    "axes.grid": True,
    "grid.alpha": 0.20,
    "grid.linewidth": 0.4,
})

C_ROLL  = "#1f77b4"
C_PITCH = "#e67e22"
C_VX    = "#1f77b4"
C_VY    = "#e67e22"
C_PUSH  = "#d62728"

def load_csv(path):
    with open(path) as f:
        r = csv.reader(f)
        header = next(r)
        rows = list(r)
    idx = {h: i for i, h in enumerate(header)}
    data = {}
    for col in header:
        ci = idx[col]
        vals = []
        for row in rows:
            try: vals.append(float(row[ci]))
            except: vals.append(float("nan"))
        data[col] = np.array(vals)
    return data

def find_td_indices(td_arr):
    events = []
    prev = 0
    for i in range(len(td_arr)):
        v = int(td_arr[i]) if np.isfinite(td_arr[i]) else 0
        if v == 1 and prev == 0: events.append(i)
        prev = v
    return events

def add_phase(ax, t, stance, alpha=0.12):
    in_s = False; t0 = 0.0
    for i in range(len(t)):
        s = int(stance[i]) if np.isfinite(stance[i]) else 0
        if s == 1 and not in_s: t0 = t[i]; in_s = True
        elif s == 0 and in_s:
            ax.axvspan(t0, t[i], color=C_ROLL, alpha=alpha, lw=0)
            in_s = False
    if in_s: ax.axvspan(t0, t[-1], color=C_ROLL, alpha=alpha, lw=0)

def mark_pushes(ax, push_times):
    for tp in push_times:
        ax.axvline(tp, color=C_PUSH, lw=1.2, ls="--", alpha=0.85, zorder=5)

# ===== LOAD =====
data = load_csv(LOG)
td_events = find_td_indices(data["touchdown"])
t0_abs = data["t_s"][0]
t = data["t_s"] - t0_abs
stance = data["stance"]

# Trim to hopping region (skip first few seconds of idle)
# First touchdown is at hop 1
i_start = td_events[0] - 50 if td_events[0] > 50 else 0
i_end = len(t)
for k in data:
    data[k] = data[k][i_start:i_end]
t = data["t_s"] - data["t_s"][0]
stance = data["stance"]
td_events_trimmed = find_td_indices(data["touchdown"])

# Push times: user says hops 5, 8, 18
push_hops = [5, 8, 18]
push_times = []
for h in push_hops:
    if h - 1 < len(td_events_trimmed):
        push_times.append(t[td_events_trimmed[h - 1]])
print(f"Push times: {push_times}")
print(f"Total hops: {len(td_events_trimmed)}, duration: {t[-1]:.1f}s")

# ============================================================
# Fig (a): Attitude + Angular velocity with push markers
# ============================================================
fig1, ax = plt.subplots(2, 1, figsize=(5.8, 2.8), sharex=True)

a = ax[0]
add_phase(a, t, stance)
mark_pushes(a, push_times)
a.plot(t, np.degrees(data["rpy_hat_roll"]),  color=C_ROLL, lw=0.8, label=r"Roll $\phi$")
a.plot(t, np.degrees(data["rpy_hat_pitch"]), color=C_PITCH, lw=0.8, label=r"Pitch $\theta$")
a.axhline(0, color="k", lw=0.3, alpha=0.4)
a.set_ylabel("Attitude (deg)")
a.legend(loc="lower left", framealpha=0.7, edgecolor="none")
# label pushes
for i, tp in enumerate(push_times):
    a.annotate(f"Push {i+1}", xy=(tp, a.get_ylim()[1]*0.85),
               fontsize=6.5, color=C_PUSH, ha="center", weight="bold")

a = ax[1]
add_phase(a, t, stance)
mark_pushes(a, push_times)
a.plot(t, np.degrees(data["imu_gyro_x"]), color=C_ROLL, lw=0.65, label=r"$\omega_x$")
a.plot(t, np.degrees(data["imu_gyro_y"]), color=C_PITCH, lw=0.65, label=r"$\omega_y$")
a.axhline(0, color="k", lw=0.3, alpha=0.4)
a.set_ylabel("Angular vel (deg/s)")
a.set_xlabel("Time (s)")
a.legend(loc="lower left", framealpha=0.7, edgecolor="none")

fig1.align_ylabels(ax)
fig1.tight_layout(h_pad=0.20)
fig1.savefig(os.path.join(OUT_DIR, "fig_push_attitude.png"), bbox_inches="tight")
plt.close(fig1)
print("Saved fig_push_attitude.png")

# ============================================================
# Fig (b): Velocity tracking with push markers
# ============================================================
fig2, ax = plt.subplots(2, 1, figsize=(5.8, 2.5), sharex=True)

a = ax[0]
add_phase(a, t, stance)
mark_pushes(a, push_times)
a.plot(t, data["v_hat_w0"],    color=C_VX, lw=0.8, label=r"$\hat{v}_x$")
a.plot(t, data["desired_vx_w"], color="k",  lw=0.8, ls="--", label=r"$v_x^{des}$")
a.set_ylabel("$v_x$ (m/s)")
a.legend(loc="upper right", ncol=2, framealpha=0.7, edgecolor="none")

a = ax[1]
add_phase(a, t, stance)
mark_pushes(a, push_times)
a.plot(t, data["v_hat_w1"],    color=C_VY, lw=0.8, label=r"$\hat{v}_y$")
a.plot(t, data["desired_vy_w"], color="k",  lw=0.8, ls="--", label=r"$v_y^{des}$")
a.set_ylabel("$v_y$ (m/s)")
a.set_xlabel("Time (s)")
a.legend(loc="upper right", ncol=2, framealpha=0.7, edgecolor="none")

fig2.align_ylabels(ax)
fig2.tight_layout(h_pad=0.20)
fig2.savefig(os.path.join(OUT_DIR, "fig_push_velocity.png"), bbox_inches="tight")
plt.close(fig2)
print("Saved fig_push_velocity.png")

print(f"\nAll push-recovery figures saved to: {OUT_DIR}/")
