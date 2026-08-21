# 过快切换失败

Indoor mobile → **9 hops** → mobile **too soon**, then **fell**. 2026-08-13 **19:44** BJ.

This is the named fall run (`switchfall`), after `demo/高度低切换失败` (19:39, 5 hops). Source snapshot:

`logs/sessions/switch_20260813_BJ/modee_20260813_194401_switchfall.csv.gz`

Raw pull is ~152 MB (19:44:01–19:50:03) and over GitHub’s 100 MB limit. This folder is the event window renamed **`过快切换失败.csv`** (19:44:01–19:44:35). Camera is not included.

Hop command is **1 cm**. Last liftoff **19:44:20.204**, gait already **mobile at 19:44:20.762** (~0.56 s later, RM folding ~11.3). Attitude blows up at **19:44:21.05**; `|rpy|` max **2.72** at 19:44:21.515. **B at 19:44:22.612** is after the fall.

File starts with a 0.1 s hopping stub, then:

| | Beijing start | Beijing end | what |
|---|---|---|---|
| mobile | 19:44:02.014 | 19:44:15.906 | before the burst |
| **hopping** | **19:44:15.906** | **19:44:20.762** | **9 hops** |
| mobile | 19:44:20.762 | 19:44:35 | switch-back, then fall |
| fall | **19:44:21.05** | | `|rpy|` > 0.8, peak 2.72 |

| # | Beijing liftoff | h_tgt | vdes | vhat |
|---|---|---|---|---|
| 1 | 19:44:16.423 | 1 cm | 0.00 | 0.00 |
| 2 | 19:44:16.939 | 1 cm | 0.00 | 0.43 |
| 3 | 19:44:17.467 | 1 cm | 0.21 | 0.89 |
| 4 | 19:44:17.970 | 1 cm | 0.33 | 0.51 |
| 5 | 19:44:18.441 | 1 cm | 0.00 | 0.11 |
| 6 | 19:44:18.893 | 1 cm | 0.00 | 0.26 |
| 7 | 19:44:19.354 | 1 cm | 0.21 | 0.77 |
| 8 | 19:44:19.783 | 1 cm | 0.67 | 1.20 |
| 9 | 19:44:20.204 | 1 cm | 0.00 | 1.25 |

## Files

| File | What |
|---|---|
| `过快切换失败.csv` | ModeE window, 19:44:01–19:44:35 BJ (9 hops, too-fast switch, fall) |
