# 室内全程

Indoor **full** LCM. 2026-08-21 **17:57–18:13** BJ.

Source session: `shineiyitao1_20260821_1811BJ` (same file as `cheku1_20260821_1811BJ`).  
Full log is `lcm_20260821_175745.csv` (102 MB uncompressed, over GitHub’s 100 MB limit). This folder is that file, NUL-stripped and gzipped as **`室内全程.csv.gz`**.

`demo/室内一套` is the same run’s **action window** only (18:08:00–18:13:22). Camera clips are also here.

No `gait_mode` in the LCM.

| | Beijing | what |
|---|---|---|
| record start | 17:57:45 | idle / setup |
| mobile | 18:08:24 | wheels on, RM folded ~11.5 |
| hop | 18:08:32–18:08:36 | PWM on, RM unfolded |
| **hop ~10** | **18:10:47–18:10:56** | main burst (session name 18:11) |
| 推箱 / 按钮 | **18:12:38–18:12:58** | hold + wheels, then button |
| hop | 18:13:08–18:13:12 | after the button |
| record end | 18:13:22 | |

## Files

| File | What |
|---|---|
| `室内全程.csv.gz` | full LCM, 17:57:45–18:13:22 BJ (`gunzip` to CSV) |
| `cheku1_1811_180819-181303.mp4` | full decodable RGB, 0:00 = 18:08:19 |
| `cheku1_1811_181000-181303.mp4` | RGB from 18:10:00 |
| `cheku1_1811_to_button_181220-181250.mp4` | 30 s to the button |
| `cheku1_1811_button_depth_181245-181258.mp4` | button window, depth false-color |
