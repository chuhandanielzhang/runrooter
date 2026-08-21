# 有速度的切换log

Indoor **mobile → hopping → mobile** with commanded speed. 2026-08-13 **19:29** BJ.

This is the named 8-hop switch (`switchwin`). Source:

`logs/sessions/switch_20260813_BJ/modee_20260813_192900_switchwin.csv`

This folder is that file, renamed **`有速度的切换log.csv`**. Camera is not included (evening `cam/20260813_191541` is ~11 GB).

Did not fall (`|rpy|` max 0.25 rad). The 8 hops are the middle burst. Commanded planar speed during that burst is about **0.65 m/s** (`desired_vx_w` / `desired_vy_w`); estimated speed (`v_hat`) peaks near **2.33 m/s**.

File starts with a 0.4 s hopping stub, then:

| | Beijing start | Beijing end | what |
|---|---|---|---|
| mobile | 19:29:00.902 | 19:29:09.092 | roll / approach |
| **hopping** | **19:29:09.092** | **19:29:13.248** | **8 hops**, vdes ≈ 0.65 |
| mobile | 19:29:13.248 | 19:29:51.017 | after switch |

| # | Beijing liftoff | vdes | vhat |
|---|---|---|---|
| 1 | 19:29:09.619 | 0.00 | 0.00 |
| 2 | 19:29:10.068 | 0.00 | 0.77 |
| 3 | 19:29:10.526 | 0.00 | 1.00 |
| 4 | 19:29:10.995 | 0.66 | 1.20 |
| 5 | 19:29:11.451 | 0.63 | 1.03 |
| 6 | 19:29:11.909 | 0.66 | 1.01 |
| 7 | 19:29:12.353 | 0.66 | 1.10 |
| 8 | 19:29:12.793 | 0.65 | 1.00 |

Hop target `hop_height_m` is 1 cm. Four `apex=1` rows log physical apex ~2–3 cm.

## Files

| File | What |
|---|---|
| `有速度的切换log.csv` | ModeE log, 19:29:00–19:29:51 BJ (8 hops, m→h→m, with speed) |
