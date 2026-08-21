# 室内一套

Indoor continuous run. 2026-08-21 **18:11** BJ.

Source session: `shineiyitao1_20260821_1811BJ` (same CSV as `cheku1_20260821_1811BJ`).  
Full LCM `lcm_20260821_175745.csv` is 102 MB (17:57:45–18:13:22), over GitHub’s 100 MB limit. This folder is the run window renamed **`室内一套.csv`** (18:08:00–18:13:22).

No `gait_mode` in the LCM. Story is mobile → hop bursts → 推箱 / 按钮. Camera RGB decodes to about 18:12:50 (MJPG tail is corrupt); depth to 18:12:58.

| | Beijing | what |
|---|---|---|
| mobile | 18:08:24 | wheels on, RM folded ~11.5 |
| hop | 18:08:32–18:08:36 | PWM on, RM unfolded |
| mobile | 18:08:38–18:08:45 | wheels; brief hold ~1.33 |
| hop | 18:09:15 |
| mobile | 18:10:33–18:10:45 | wheels |
| **hop ~10** | **18:10:47–18:10:56** | main burst (session name 18:11) |
| hop | 18:12:31–18:12:36 | |
| 推箱 / 到按钮 | **18:12:38–18:12:54** | hold ~`(1.35, …)` + wheels |
| 按钮 | **18:12:45–18:12:58** | depth clip |
| hop | 18:13:08–18:13:12 | after the button |

## Files

| File | What |
|---|---|
| `室内一套.csv` | LCM window, 18:08:00–18:13:22 BJ |
| `cheku1_1811_180819-181303.mp4` | full decodable RGB, 0:00 = 18:08:19 |
| `cheku1_1811_181000-181303.mp4` | RGB from 18:10:00 (2:50, ends ~18:12:50) |
| `cheku1_1811_to_button_181220-181250.mp4` | 30 s to the button |
| `cheku1_1811_button_depth_181245-181258.mp4` | button window, depth false-color |
