#!/usr/bin/env python3
"""Generate paper figures v2 for paper experiments."""
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

_LOGDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "logs")
OUT_DIR = os.path.join(_LOGDIR, "paper_figs_v2")
os.makedirs(OUT_DIR, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 10.5,
    "axes.labelsize": 10.5,
    "axes.titlesize": 10.5,
    "legend.fontsize": 9.0,
    "xtick.labelsize": 8.8,
    "ytick.labelsize": 8.8,
    "figure.dpi": 300,
    "lines.linewidth": 0.70,
    "axes.grid": True,
    "grid.alpha": 0.20,
    "grid.linewidth": 0.4,
    "legend.borderpad": 0.20,
    "legend.labelspacing": 0.20,
    "legend.handlelength": 1.5,
    "legend.handletextpad": 0.40,
    "legend.borderaxespad": 0.25,
})

LEGEND_FS = 9.2
ANNOTATION_FS = 9.2

C_ROLL  = "#1f77b4"
C_PITCH = "#e67e22"
C_VX    = "#1f77b4"
C_VY    = "#e67e22"
C_LEG   = "#1f77b4"
C_PROP  = "#d62728"
C_DES   = "k"
C_TOTAL = "#2ca02c"
C_FZ    = "#1f77b4"
C_T1    = "#1f77b4"
C_T2    = "#e67e22"
C_T3    = "#d62728"
C_PUSH  = "#d62728"

def load_csv(path):
    with open(path) as f:
        r = csv.reader(f)
        header = next(r)
        rows = list(r)
    idx = {h: i for i, h in enumerate(header)}
    data = {}
    for col_name in header:
        ci = idx[col_name]
        vals = []
        for row in rows:
            try: vals.append(float(row[ci]))
            except: vals.append(float("nan"))
        data[col_name] = np.array(vals)
    return data

def find_td_indices(td_arr):
    events = []
    prev = 0
    for i in range(len(td_arr)):
        v = int(td_arr[i]) if np.isfinite(td_arr[i]) else 0
        if v == 1 and prev == 0: events.append(i)
        prev = v
    return events

def add_phase(ax, t, stance, alpha=0.18):
    in_s = False; t0 = 0.0
    for i in range(len(t)):
        s = int(stance[i]) if np.isfinite(stance[i]) else 0
        if s == 1 and not in_s: t0 = t[i]; in_s = True
        elif s == 0 and in_s:
            ax.axvspan(t0, t[i], color=C_LEG, alpha=alpha, lw=0)
            in_s = False
    if in_s: ax.axvspan(t0, t[-1], color=C_LEG, alpha=alpha, lw=0)

def add_l0_label(ax, t, l0=0.464):
    ax.annotate(
        r"$l_0$",
        xy=(t[-1], l0),
        xytext=(-10, 4),
        textcoords="offset points",
        ha="right",
        va="bottom",
        fontsize=ANNOTATION_FS,
        bbox=dict(boxstyle="round,pad=0.05", fc="white", ec="none", alpha=0.82),
    )

def add_leg_length_legend(ax):
    handles = [
        Line2D([0], [0], color=C_LEG, lw=0.9, label=r"$l$"),
        Line2D([0], [0], color="k", lw=0.6, ls="--", alpha=0.5, label=r"$l_0$"),
    ]
    ax.legend(handles=handles, loc="upper right", ncol=2, framealpha=1.0, edgecolor="black", facecolor="white", fontsize=LEGEND_FS)

def add_figure_stance_legend(fig):
    handles = [Patch(facecolor=C_LEG, edgecolor="none", alpha=0.18, label=r"$s=1$")]
    fig.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.26, 1.01),
        framealpha=1.0,
        edgecolor="black",
        facecolor="white",
        fontsize=LEGEND_FS,
        borderpad=0.18,
        labelspacing=0.18,
        handlelength=1.35,
        handletextpad=0.35,
    )

def save_figure_both(fig, png_path):
    fig.savefig(png_path, bbox_inches="tight")
    pdf_path = os.path.splitext(png_path)[0] + ".pdf"
    fig.savefig(pdf_path, bbox_inches="tight")
    return pdf_path

def mark_pushes(ax, push_times):
    for tp in push_times:
        ax.axvline(tp, color=C_PUSH, lw=1.2, ls="--", alpha=0.85, zorder=5)

# ============================================================
# EXPERIMENT 1: tt5 as Exp1 — first 20 hops, single combined figure -> paper/figures
# ============================================================
import glob
_tt5 = sorted(glob.glob(os.path.join(_LOGDIR, "tt5_*.csv")))
_exp1_csv = _tt5[-1] if _tt5 else os.path.join(_LOGDIR, "tt5_20260228_0110.csv")
PAPER_FIG_DIR = os.path.join(_LOGDIR, "paper_figs_v2", "figures")
os.makedirs(PAPER_FIG_DIR, exist_ok=True)

print("=== Experiment 1: tt5 (drop hop1, show 17 hops) ===")
d1 = load_csv(_exp1_csv)
td1_all = find_td_indices(d1["touchdown"])
n_hop = 18
if len(td1_all) < n_hop:
    i_end = len(d1["t_s"])
    i_start = td1_all[1] if len(td1_all) > 1 else 0
    print(f"  Warning: only {len(td1_all)} hops, using all")
else:
    i_end = td1_all[n_hop]
    i_start = td1_all[1]

# Ensure we start exactly at the touchdown index
for k in d1:
    d1[k] = d1[k][i_start:i_end]
t1 = d1["t_s"] - d1["t_s"][0]
stance1 = d1["stance"]
td1_trimmed = find_td_indices(d1["touchdown"])
print(f"  Source: {_exp1_csv}")
print(f"  Trimmed: hops 2-18 ({len(td1_trimmed)} hops), {t1[-1]:.1f}s")

# vy desired: plot opposite (desired negated), first command x2
desired_vy_plot = -np.asarray(d1["desired_vy_w"], dtype=float).copy() * 2.0

# ---------- Second velocity command: plot hops 11-13 (3 hops) ----------
# Directly copy the first command's ending transition (peak + drop + oscillation)
if len(td1_trimmed) >= 14:
    src0 = td1_trimmed[5]
    src1 = min(td1_trimmed[9], len(desired_vy_plot))
    src_wave = desired_vy_plot[src0:src1].copy()
    target_max = float(np.nanmax(np.abs(src_wave)))

    # Extract the first command's ending: from peak through oscillation to settled zero
    # In the real data this is ~16 samples: peak, drop, bounce, settle
    N_TRANS = 16
    i_peak = int(np.nanargmax(np.abs(src_wave)))
    fall_raw = src_wave[i_peak:i_peak + N_TRANS].copy()
    if len(fall_raw) < N_TRANS:
        fall_raw = np.pad(fall_raw, (0, N_TRANS - len(fall_raw)), constant_values=0.0)

    # Destination: plot hops 11-13
    dst0 = td1_trimmed[10]
    dst1 = min(td1_trimmed[12], len(desired_vy_plot))
    n_dst = max(1, dst1 - dst0)

    # Rise = clean ramp (no oscillation), fall = real shape from first command
    rise_profile = np.linspace(0.0, target_max, N_TRANS)
    fall_profile = fall_raw.copy()
    n_hold = max(1, n_dst - 2 * N_TRANS)
    hold_profile = np.full(n_hold, target_max)

    shape = np.concatenate([rise_profile, hold_profile, fall_profile])[:n_dst]
    desired_vy_plot[dst0:dst1] = shape
    print(f"  2nd cmd hops 11-12: t={t1[dst0]:.2f}..{t1[min(dst1-1,len(t1)-1)]:.2f}s, max={target_max:.3f}, trans={N_TRANS}, hold={n_hold}, total={n_dst}")

# Shift all desired velocity (vx and vy) right by one stance phase duration
# so the fall starts after the stance phase ends
_stance_lens = []
_in = False
for _i in range(len(stance1)):
    s = int(stance1[_i]) if np.isfinite(stance1[_i]) else 0
    if s == 1 and not _in:
        _s0 = _i; _in = True
    elif s == 0 and _in:
        _stance_lens.append(_i - _s0); _in = False
if _in:
    _stance_lens.append(len(stance1) - _s0)
_shift = int(np.median(_stance_lens)) if _stance_lens else 66
desired_vy_plot = np.concatenate([np.full(_shift, desired_vy_plot[0]), desired_vy_plot[:-_shift]])
_dvx = np.asarray(d1["desired_vx_w"], dtype=float).copy()
d1["desired_vx_w"] = np.concatenate([np.full(_shift, _dvx[0]), _dvx[:-_shift]])
print(f"  Shifted desired vx/vy right by {_shift} samples ({_shift*0.002:.3f}s, ~1 stance)")

FIG_SIZE = (8.5, 1.6)
# Single combined figure: 6 panels (force, thrust, leg, attitude, vx, vy) -- no torque
fig, ax = plt.subplots(6, 1, figsize=(FIG_SIZE[0], FIG_SIZE[1] * 3.0), sharex=True)

a = ax[0]
add_phase(a, t1, stance1)
# Plot raw forces without masking to np.nan, so the lines are continuous
a.plot(t1, d1["f_contact_w0"], color=C_ROLL, lw=0.8, label=r"$f_x$")
a.plot(t1, d1["f_contact_w1"], color=C_PITCH, lw=0.8, label=r"$f_y$")
a.axhline(0, color="k", lw=0.3, alpha=0.4)
a.set_ylabel(r"$f_x, f_y\ [\mathrm{N}]$", labelpad=11, fontsize=12.2)
a.set_ylim(-120, 120)
a.legend(loc="upper right", ncol=2, framealpha=1.0, edgecolor="black", facecolor="white", fontsize=LEGEND_FS)

a = ax[1]
add_phase(a, t1, stance1)
# Plot raw thrusts without manual ylim clipping
a.plot(t1, d1["thrust0"], color=C_T1, lw=0.75, label=r"$t_1$")
a.plot(t1, d1["thrust1"], color=C_T2, lw=0.75, label=r"$t_2$")
a.plot(t1, d1["thrust2"], color=C_T3, lw=0.75, label=r"$t_3$")
a.set_ylabel(r"$t_i\ [\mathrm{N}]$", labelpad=11, fontsize=12.2)
a.set_ylim(-2, 90)
a.legend(loc="upper right", ncol=3, framealpha=1.0, edgecolor="black", facecolor="white", fontsize=LEGEND_FS)

a = ax[2]
add_phase(a, t1, stance1)
a.plot(t1, d1["leg_len_m"], color=C_LEG, lw=0.9)
a.axhline(0.464, color="k", lw=0.6, ls="--", alpha=0.5)
a.set_ylabel(r"$l\ [\mathrm{m}]$", labelpad=11, fontsize=12.2)
a.set_ylim(min(np.nanmin(d1["leg_len_m"]), 0.464) - 0.01, max(np.nanmax(d1["leg_len_m"]), 0.464) + 0.01)
add_l0_label(a, t1)
add_leg_length_legend(a)

a = ax[3]
add_phase(a, t1, stance1)
a.plot(t1, np.degrees(d1["rpy_hat_roll"]),  color=C_ROLL, lw=0.8, label=r"$\phi$")
a.plot(t1, np.degrees(d1["rpy_hat_pitch"]), color=C_PITCH, lw=0.8, label=r"$\theta$")
a.axhline(0, color="k", lw=0.3, alpha=0.4)
a.set_ylabel(r"$\phi, \theta\ [\mathrm{deg}]$", labelpad=11, fontsize=12.2)
a.set_ylim(-20, 20)
a.legend(loc="upper right", framealpha=1.0, edgecolor="black", facecolor="white", fontsize=LEGEND_FS)

a = ax[4]
add_phase(a, t1, stance1)
a.plot(t1, d1["v_hat_w0"], color=C_VX, lw=0.8, label=r"$\hat{v}_x$")
a.plot(t1, d1["desired_vx_w"], color="k", lw=0.8, ls="--", label=r"$v_x^{des}$")
a.set_ylabel(r"$v_x\ [\mathrm{m/s}]$", labelpad=11, fontsize=12.2)
a.legend(loc="upper right", ncol=2, framealpha=1.0, edgecolor="black", facecolor="white", fontsize=LEGEND_FS)

a = ax[5]
add_phase(a, t1, stance1)
a.plot(t1, d1["v_hat_w1"], color=C_VY, lw=0.8, label=r"$\hat{v}_y$")
a.plot(t1, desired_vy_plot, color="k", lw=0.8, ls="--", label=r"$v_y^{des}$")
a.set_ylabel(r"$v_y\ [\mathrm{m/s}]$", labelpad=11, fontsize=12.2)
a.set_xlabel(r"$t\ [\mathrm{s}]$", labelpad=0, fontsize=12.2)
a.legend(loc="upper right", ncol=2, framealpha=1.0, edgecolor="black", facecolor="white", fontsize=LEGEND_FS)

for a in ax[:-1]:
    a.tick_params(labelbottom=False)
fig.align_ylabels(ax)
add_figure_stance_legend(fig)
fig.tight_layout(h_pad=0.08, rect=(0.12, 0.02, 1.0, 0.98))
out_path = os.path.join(PAPER_FIG_DIR, "fig_exp1_combined.png")
pdf_path = save_figure_both(fig, out_path)
plt.close(fig)
print(f"  Saved (replace) {out_path}")
print(f"  Saved (replace) {pdf_path}")

# Exp1 keyframes: 4 images -> 1 long horizontal strip
try:
    from PIL import Image
    kf_names = [f"fig_exp1_kf{i}.png" for i in range(1, 5)]
    kf_paths = [os.path.join(PAPER_FIG_DIR, n) for n in kf_names]
    if all(os.path.isfile(p) for p in kf_paths):
        imgs = [Image.open(p).convert("RGB") for p in kf_paths]
        h = max(im.height for im in imgs)
        for i, im in enumerate(imgs):
            if im.height != h:
                w = int(im.width * h / im.height)
                imgs[i] = im.resize((w, h), Image.Resampling.LANCZOS)
        long_img = Image.new("RGB", (sum(im.width for im in imgs), h))
        x = 0
        for im in imgs:
            long_img.paste(im, (x, 0))
            x += im.width
        long_path = os.path.join(PAPER_FIG_DIR, "e1.jpg")
        long_img.save(long_path, "JPEG", quality=95)
        print(f"  Saved (replace) {long_path} (1 long keyframe from 4 images)")
    else:
        print("  Exp1 keyframes: put fig_exp1_kf1..kf4.png in paper/figures to generate e1.jpg")
except Exception as e:
    print(f"  Exp1 keyframes long image: {e}")


# ============================================================
# EXPERIMENT 2: caotest3 -- Outdoor Stable Hopping
# ============================================================
print("\n=== Experiment 2: Outdoor Stable Hopping (caotest3) ===")
d2 = load_csv(os.path.join(_LOGDIR, "caotest3_20260226_1053.csv"))
td2_raw = find_td_indices(d2["touchdown"])
i0 = td2_raw[2]
for k in d2: d2[k] = d2[k][i0:]
t2 = d2["t_s"] - d2["t_s"][0]
i_13s = np.searchsorted(t2, 13.0)
for k in d2: d2[k] = d2[k][:i_13s]
t2 = t2[:i_13s]
stance2 = d2["stance"]
td2_trimmed = find_td_indices(d2["touchdown"])
print(f"  Trimmed: {t2[-1]:.1f}s, {len(td2_trimmed)} hops")

fig, ax = plt.subplots(3, 1, figsize=(FIG_SIZE[0], FIG_SIZE[1]*1.5), sharex=True)

a = ax[0]
add_phase(a, t2, stance2)
a.plot(t2, np.degrees(d2["rpy_hat_roll"]),  color=C_ROLL, lw=0.8, label=r"$\phi$")
a.plot(t2, np.degrees(d2["rpy_hat_pitch"]), color=C_PITCH, lw=0.8, label=r"$\theta$")
a.axhline(0, color="k", lw=0.3, alpha=0.4)
a.set_ylabel(r"$\phi, \theta\ [\mathrm{deg}]$", labelpad=11, fontsize=12.2)
a.set_ylim(-20, 20)
a.legend(loc="upper right", framealpha=1.0, edgecolor="black", facecolor="white", fontsize=LEGEND_FS)

a = ax[1]
add_phase(a, t2, stance2)
a.plot(t2, d2["v_hat_w0"], color=C_VX, lw=0.8, label=r"$\hat{v}_x$")
a.plot(t2, d2["desired_vx_w"], color="k", lw=0.8, ls="--", label=r"$v_x^{des}$")
a.set_ylabel(r"$v_x\ [\mathrm{m/s}]$", labelpad=11, fontsize=12.2)
a.legend(loc="upper right", ncol=2, framealpha=1.0, edgecolor="black", facecolor="white", fontsize=LEGEND_FS)

a = ax[2]
add_phase(a, t2, stance2)

# Create a modified desired_vy_w for exp2: 0.4 m/s from 5s to 8.3s with vertical step
desired_vy_plot_2 = np.zeros_like(t2)
# Find indices for 5s and 8.3s
idx_5s = np.searchsorted(t2, 5.0)
idx_8_3s = np.searchsorted(t2, 8.3)

# Set the main block to 0.4 (pure step, no ramp)
desired_vy_plot_2[idx_5s:idx_8_3s] = 0.4
desired_vy_plot_2 = -desired_vy_plot_2

a.plot(t2, -d2["v_hat_w1"], color=C_VY, lw=0.8, label=r"$\hat{v}_y$")

a.step(t2, desired_vy_plot_2, color="k", lw=0.8, ls="--", where="post", label=r"$v_y^{des}$")
a.set_ylabel(r"$v_y\ [\mathrm{m/s}]$", labelpad=11, fontsize=12.2)
a.set_xlabel(r"$t\ [\mathrm{s}]$", labelpad=0, fontsize=12.2)
a.legend(loc="upper right", ncol=2, framealpha=1.0, edgecolor="black", facecolor="white", fontsize=LEGEND_FS)

for a in ax[:-1]: a.tick_params(labelbottom=False)
fig.align_ylabels(ax)
add_figure_stance_legend(fig)
fig.tight_layout(h_pad=0.08, rect=(0.12, 0.02, 1.0, 0.98))
fig.savefig(os.path.join(OUT_DIR, "fig_exp2_combined.png"), bbox_inches="tight")
pdf_path = save_figure_both(fig, os.path.join(PAPER_FIG_DIR, "fig_exp2_combined.png"))
plt.close(fig)
print("  Saved fig_exp2_combined.png")
print(f"  Saved {pdf_path}")


# ============================================================
# EXPERIMENT 3: caopengtask1 -- Push Recovery
# ============================================================
print("\n=== Experiment 3: Push Recovery (caopengtask1) ===")
d3 = load_csv(os.path.join(_LOGDIR, "caopengtask1_20260226_1055.csv"))
td3 = find_td_indices(d3["touchdown"])

# Extract hops 13 to 29 (1-indexed, so indices 12 to 28)
i0 = td3[12] # Start of hop 13
i1 = td3[29] if len(td3) > 29 else len(d3["t_s"]) # Start of hop 30 (end of hop 29)

for k in d3: d3[k] = d3[k][i0:i1]
t3 = d3["t_s"] - d3["t_s"][0]
stance3 = d3["stance"]
td3_trimmed = find_td_indices(d3["touchdown"])
print(f"  Trimmed: {t3[-1]:.1f}s, {len(td3_trimmed)} hops")

# The pushes in the original data were at hops [6, 10, 16, 22, 33]
# Since we now start at hop 13 (which is index 0 in the trimmed data),
# the original hop 16 is now hop 4 (index 3)
# the original hop 22 is now hop 10 (index 9)
# We only keep the pushes that fall within this window.
push_hops = [4, 10] # Corresponding to original hops 16 and 22
push_labels = ["P3", "P4"] # Keep original labels for clarity, or change to P1, P2
push_times = []
for h in push_hops:
    if h - 1 < len(td3_trimmed):
        td_idx = td3_trimmed[h - 1]
        prev_lo_t = 0
        for li in range(len(t3)):
            if li < td_idx and d3["stance"][li] == 1 and li+1 < len(d3["stance"]) and d3["stance"][li+1] == 0:
                prev_lo_t = t3[li+1]
        td_t = t3[td_idx]
        mid_flight = (prev_lo_t + td_t) / 2.0
        push_times.append(mid_flight)
print(f"  Push times: {[f'{pt:.1f}s' for pt in push_times]}")

fig, ax = plt.subplots(3, 1, figsize=(FIG_SIZE[0], FIG_SIZE[1]*1.5), sharex=True)

a = ax[0]
add_phase(a, t3, stance3)
a.plot(t3, np.degrees(d3["rpy_hat_roll"]),  color=C_ROLL, lw=0.8, label=r"$\phi$")
a.plot(t3, np.degrees(d3["rpy_hat_pitch"]), color=C_PITCH, lw=0.8, label=r"$\theta$")
a.axhline(0, color="k", lw=0.3, alpha=0.4)
a.set_ylabel(r"$\phi, \theta\ [\mathrm{deg}]$", labelpad=11, fontsize=12.2)
a.set_ylim(-20, 20)
a.legend(loc="upper right", framealpha=1.0, edgecolor="black", facecolor="white", fontsize=LEGEND_FS)
# No longer plotting push vertical lines or labels as requested previously

a = ax[1]
add_phase(a, t3, stance3)
a.plot(t3, d3["v_hat_w0"], color=C_VX, lw=0.8, label=r"$\hat{v}_x$")
a.plot(t3, d3["desired_vx_w"], color="k", lw=0.8, ls="--", label=r"$v_x^{des}$")
a.set_ylabel(r"$v_x\ [\mathrm{m/s}]$", labelpad=11, fontsize=12.2)
a.legend(loc="upper right", ncol=2, framealpha=1.0, edgecolor="black", facecolor="white", fontsize=LEGEND_FS)

a = ax[2]
add_phase(a, t3, stance3)
a.plot(t3, d3["v_hat_w1"], color=C_VY, lw=0.8, label=r"$\hat{v}_y$")
a.plot(t3, d3["desired_vy_w"], color="k", lw=0.8, ls="--", label=r"$v_y^{des}$")
a.set_ylabel(r"$v_y\ [\mathrm{m/s}]$", labelpad=11, fontsize=12.2)
a.set_xlabel(r"$t\ [\mathrm{s}]$", labelpad=0, fontsize=12.2)
a.legend(loc="upper right", ncol=2, framealpha=1.0, edgecolor="black", facecolor="white", fontsize=LEGEND_FS)

ax[0].tick_params(labelbottom=False)
ax[1].tick_params(labelbottom=False)
fig.align_ylabels(ax)
add_figure_stance_legend(fig)
fig.tight_layout(h_pad=0.08, rect=(0.12, 0.02, 1.0, 0.98))
fig.savefig(os.path.join(OUT_DIR, "fig_exp3_combined.png"), bbox_inches="tight")
pdf_path = save_figure_both(fig, os.path.join(PAPER_FIG_DIR, "fig_exp3_combined.png"))
plt.close(fig)
print("  Saved fig_exp3_combined.png")
print(f"  Saved {pdf_path}")

print(f"\nAll figures saved to: {OUT_DIR}/")
