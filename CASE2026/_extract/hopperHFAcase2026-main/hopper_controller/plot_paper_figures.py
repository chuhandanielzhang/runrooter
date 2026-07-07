#!/usr/bin/env python3
"""Generate paper figures from test2_1 CSV — middle 20 hops."""
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

_LOGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
LOG = os.path.join(_LOGDIR, "test2_1.csv")
OUT_DIR = os.path.join(_LOGDIR, "paper_figs")
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

C_LEG   = "#1f77b4"
C_PROP  = "#d62728"
C_DES   = "k"
C_TOTAL = "#2ca02c"
C_ROLL  = "#1f77b4"
C_PITCH = "#e67e22"
C_VX    = "#1f77b4"
C_VY    = "#e67e22"
C_FZ    = "#1f77b4"
C_FX    = "#9467bd"
C_FY    = "#8c564b"
C_T1    = "#1f77b4"
C_T2    = "#e67e22"
C_T3    = "#d62728"

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
    return data, len(rows), idx

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
            ax.axvspan(t0, t[i], color=C_LEG, alpha=alpha, lw=0)
            in_s = False
    if in_s: ax.axvspan(t0, t[-1], color=C_LEG, alpha=alpha, lw=0)

def mark_td_lo(ax, t, td, lo):
    prev_td = prev_lo = 0
    for i in range(len(t)):
        td_n = int(td[i]) if np.isfinite(td[i]) else 0
        lo_n = int(lo[i]) if np.isfinite(lo[i]) else 0
        if td_n == 1 and prev_td == 0: ax.axvline(t[i], color="#2ca02c", lw=0.35, alpha=0.45)
        if lo_n == 1 and prev_lo == 0: ax.axvline(t[i], color="#ff7f0e", lw=0.35, alpha=0.45)
        prev_td, prev_lo = td_n, lo_n

# ===== LOAD & TRIM to middle 20 hops =====
data, n, idx = load_csv(LOG)
td_events = find_td_indices(data["touchdown"])
i0 = td_events[14]
i1 = td_events[34]
for k in data:
    data[k] = data[k][i0:i1]
n = i1 - i0
t = data["t_s"] - data["t_s"][0]
stance = data["stance"]
td = data["touchdown"]
lo = data["liftoff"]

# ============================================================
# (b) Leg compression + Attitude + Angular velocity
# ============================================================
fig1, ax = plt.subplots(3, 1, figsize=(5.8, 3.7), sharex=True)

a = ax[0]
add_phase(a, t, stance)
a.plot(t, data["leg_len_m"], color=C_LEG, lw=0.9)
a.axhline(0.464, color="k", lw=0.6, ls="--", alpha=0.5)
a.set_ylabel("Leg length (m)")
a.text(t[-1]*0.98, 0.466, "$l_0$", ha="right", va="bottom", fontsize=8)

a = ax[1]
add_phase(a, t, stance)
a.plot(t, np.degrees(data["rpy_hat_roll"]),  color=C_ROLL, lw=0.8, label=r"Roll $\phi$")
a.plot(t, np.degrees(data["rpy_hat_pitch"]), color=C_PITCH, lw=0.8, label=r"Pitch $\theta$")
a.axhline(0, color="k", lw=0.3, alpha=0.4)
a.set_ylabel("Attitude (deg)")
a.legend(loc="lower left", framealpha=0.7, edgecolor="none")

a = ax[2]
add_phase(a, t, stance)
a.plot(t, np.degrees(data["imu_gyro_x"]), color=C_ROLL, lw=0.65, label=r"$\omega_x$")
a.plot(t, np.degrees(data["imu_gyro_y"]), color=C_PITCH, lw=0.65, label=r"$\omega_y$")
a.axhline(0, color="k", lw=0.3, alpha=0.4)
a.set_ylabel("Angular vel (deg/s)")
a.set_xlabel("Time (s)")
a.legend(loc="lower left", framealpha=0.7, edgecolor="none")

fig1.align_ylabels(ax)
fig1.tight_layout(h_pad=0.20)
fig1.savefig(os.path.join(OUT_DIR, "fig_hopping.png"), bbox_inches="tight")
plt.close(fig1)
print("Saved fig_hopping.png")

# ============================================================
# (c) Wrench Allocation — NO phase bar
# ============================================================
fig2, ax = plt.subplots(2, 1, figsize=(5.8, 2.8), sharex=True)

a = ax[0]
add_phase(a, t, stance)
a.plot(t, data["tau_des_w0"],     color=C_DES,   lw=1.0, label=r"$\tau_{des}$")
a.plot(t, data["tau_contact_w0"], color=C_LEG,   lw=0.8, label=r"$\tau_{leg}$")
a.plot(t, data["tau_props_w0"],   color=C_PROP,  lw=0.8, label=r"$\tau_{prop}$")
a.plot(t, data["tau_total_w0"],   color=C_TOTAL, lw=0.55, ls="--", label=r"$\tau_{total}$")
a.set_ylabel("Roll torque (Nm)")
a.legend(loc="upper left", ncol=4, framealpha=0.7, edgecolor="none")

a = ax[1]
add_phase(a, t, stance)
a.plot(t, data["tau_des_w1"],     color=C_DES,   lw=1.0)
a.plot(t, data["tau_contact_w1"], color=C_LEG,   lw=0.8)
a.plot(t, data["tau_props_w1"],   color=C_PROP,  lw=0.8)
a.plot(t, data["tau_total_w1"],   color=C_TOTAL, lw=0.55, ls="--")
a.set_ylabel("Pitch torque (Nm)")
a.set_xlabel("Time (s)")

fig2.align_ylabels(ax)
fig2.tight_layout(h_pad=0.20)
fig2.savefig(os.path.join(OUT_DIR, "fig_wrench.png"), bbox_inches="tight")
plt.close(fig2)
print("Saved fig_wrench.png")

# ============================================================
# (d) GRF + Prop Thrust
# ============================================================
fig3, ax = plt.subplots(2, 1, figsize=(5.8, 2.7), sharex=True)

a = ax[0]
add_phase(a, t, stance)
mark_td_lo(a, t, td, lo)
a.plot(t, data["f_contact_w2"], color=C_FZ, lw=0.9, label=r"$f_z$")
a.plot(t, data["f_contact_w0"], color=C_FX, lw=0.55, alpha=0.7, label=r"$f_x$")
a.plot(t, data["f_contact_w1"], color=C_FY, lw=0.55, alpha=0.7, label=r"$f_y$")
a.set_ylabel("Contact force (N)")
a.legend(loc="upper right", ncol=3, framealpha=0.7, edgecolor="none")

a = ax[1]
add_phase(a, t, stance)
mark_td_lo(a, t, td, lo)
a.plot(t, data["thrust0"], color=C_T1, lw=0.75, label=r"$t_1$")
a.plot(t, data["thrust1"], color=C_T2, lw=0.75, label=r"$t_2$")
a.plot(t, data["thrust2"], color=C_T3, lw=0.75, label=r"$t_3$")
a.set_ylabel("Thrust (N)")
a.set_xlabel("Time (s)")
a.legend(loc="upper right", ncol=3, framealpha=0.7, edgecolor="none")

fig3.align_ylabels(ax)
fig3.tight_layout(h_pad=0.20)
fig3.savefig(os.path.join(OUT_DIR, "fig_grf_thrust.png"), bbox_inches="tight")
plt.close(fig3)
print("Saved fig_grf_thrust.png")

# ============================================================
# (e) Velocity Tracking only
# ============================================================
fig4, ax = plt.subplots(2, 1, figsize=(5.8, 2.5), sharex=True)

a = ax[0]
add_phase(a, t, stance)
a.plot(t, data["v_hat_w0"],    color=C_VX, lw=0.8, label=r"$\hat{v}_x$")
a.plot(t, data["desired_vx_w"], color="k",  lw=0.8, ls="--", label=r"$v_x^{des}$")
a.set_ylabel("$v_x$ (m/s)")
a.legend(loc="upper right", ncol=2, framealpha=0.7, edgecolor="none")

a = ax[1]
add_phase(a, t, stance)
a.plot(t, data["v_hat_w1"],    color=C_VY, lw=0.8, label=r"$\hat{v}_y$")
a.plot(t, data["desired_vy_w"], color="k",  lw=0.8, ls="--", label=r"$v_y^{des}$")
a.set_ylabel("$v_y$ (m/s)")
a.set_xlabel("Time (s)")
a.legend(loc="upper right", ncol=2, framealpha=0.7, edgecolor="none")

fig4.align_ylabels(ax)
fig4.tight_layout(h_pad=0.20)
fig4.savefig(os.path.join(OUT_DIR, "fig_velocity.png"), bbox_inches="tight")
plt.close(fig4)
print("Saved fig_velocity.png")

print(f"\nAll figures saved to: {OUT_DIR}/")
