# 零速度切换

Outdoor **mobile → hopping → mobile** at commanded speed **0**. 2026-08-10 **22:22** BJ (not 22:24).

Source session: `outdoor1_20260810_222503BJ`.  
This folder is the switch window from `modee_20260810_222236.csv`, renamed **`零速度切换.csv`** (22:22:36–22:23:20). Full file is 115 MB (GitHub has the `.gz` under `logs/sessions/`).

Not `demo/有速度的切换log` (that is 8/13 19:29, vdes ≈ 0.65).

Did not fall (`|rpy|` max 0.26 rad). Hop target `hop_height_m` is 1 cm. **8 hops** in the middle burst, all `desired_v` = 0.

| | Beijing start | Beijing end | what |
|---|---|---|---|
| mobile | 22:22:36.675 | 22:22:45.886 | |
| **hopping** | **22:22:45.886** | **22:22:52.649** | **8 hops**, vdes = 0 |
| mobile | 22:22:52.649 | 22:23:20 | after switch |

| # | Beijing liftoff | vdes | vhat |
|---|---|---|---|
| 1 | 22:22:46.444 | 0.00 | 0.01 |
| 2 | 22:22:46.901 | 0.00 | 0.52 |
| 3 | 22:22:47.414 | 0.00 | 0.56 |
| 4 | 22:22:47.900 | 0.00 | 0.41 |
| 5 | 22:22:48.375 | 0.00 | 0.57 |
| 6 | 22:22:48.864 | 0.00 | 0.77 |
| 7 | 22:22:49.273 | 0.00 | 0.87 |
| 8 | 22:22:49.791 | 0.00 | 0.81 |

## Files

| File | What |
|---|---|
| `零速度切换.csv` | ModeE window, 22:22:36–22:23:20 BJ |
| `color.avi` | RGB `cam/20260810_222122/seg000` (from 22:21:22). Depth omitted (490 MB). |
