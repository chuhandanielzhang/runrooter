# 高度低切换失败

Indoor **low-height** mobile → hopping, then the switch back **failed**. 2026-08-13 **19:39** BJ.

This is the named 5-hop run (`switchlow`), right after `demo/有速度的切换log` (19:29, 8 hops). Source:

`logs/sessions/switch_20260813_BJ/modee_20260813_193927_switchlow.csv`

This folder is that file, renamed **`高度低切换失败.csv`**. Camera is not included.

Hop command is **1 cm** (`hop_height_m=0.01`). Five hops, then **B at 19:39:43** to fold / leave hopping. RM never reached the fold pose (~11.5). Gait stayed `hopping`. Did not fall (`|rpy|` max 0.20 rad). Commanded speed is near zero (not the velocity switch).

File starts with a 1.2 s hopping stub, then:

| | Beijing start | Beijing end | what |
|---|---|---|---|
| mobile | 19:39:28.742 | 19:39:39.460 | before the burst |
| **hopping** | **19:39:39.460** | 19:40:21.836 | **5 hops**, then fold attempt, still hopping |
| B | **19:39:43.721** | | LT fold; RM stuck ~`(7.7, 8.1, 9.7)` not 11.5 |

| # | Beijing liftoff | h_tgt | vdes |
|---|---|---|---|
| 1 | 19:39:39.925 | 1 cm | 0 |
| 2 | 19:39:40.373 | 1 cm | 0 |
| 3 | 19:39:40.815 | 1 cm | 0 |
| 4 | 19:39:41.272 | 1 cm | 0 |
| 5 | 19:39:41.826 | 1 cm | 0 |

A later 6th liftoff at 19:40:10.547 is after the failed fold, still `hopping`. Apex flags on the burst are ~1.8–2.6 cm.

## Files

| File | What |
|---|---|
| `高度低切换失败.csv` | ModeE log, 19:39:27–19:40:21 BJ (low height, 5 hops, switch-back failed) |
