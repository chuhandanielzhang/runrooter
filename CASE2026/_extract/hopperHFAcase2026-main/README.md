# hopperHFAcase2026

Cao's classical ModeE controller for the hybrid hopper, plus the paper-relevant
experiment logs. This is a curated, self-contained snapshot for the paper:
the low-level C++ leg driver, the high-level Python controller, and the exact
logs the paper figures are generated from.

## Layout

```
.
├── main.cpp  CMakeLists.txt  rebuild.sh   # low level: C++ leg driver (hopper_driver)
├── src/  include/                          #   sources / headers
├── xbox_driver/                            #   gamepad input
├── hopper_lcm_types/                       #   LCM message definitions (.lcm + generated cpp)
├── model/                                  #   MuJoCo XML + meshes (used by the sim/upper layer)
├── hopper_controller/                      # high level: Cao ModeE controller
│   ├── run_modee.py                        #   main controller entry point
│   ├── modee/                              #   ModeE state machine + control
│   ├── forward_kinematics.py
│   ├── mujoco_lcm_fake_robot.py            #   sim-in-the-loop fake robot over LCM
│   ├── plot_paper_figures.py               #   paper figures  (reads ../logs/test2_1.csv)
│   ├── plot_paper_v2.py                    #   paper figures v2 (reads ../logs/*.csv)
│   └── record_modee_serial_*.sh            #   data-recording helpers
└── logs/                                   # paper-relevant logs (the ones the figures use)
    ├── test2_1.csv                         #   -> plot_paper_figures.py
    ├── tt5_20260228_0110.csv               #   -> plot_paper_v2.py (Exp1)
    ├── caotest3_20260226_1053.csv          #   -> plot_paper_v2.py (Exp2, outdoor)
    ├── caopengtask1_20260226_1055.csv      #   -> plot_paper_v2.py (Exp3)
    └── figs/                               #   pre-rendered reference figures (PNG)
```

The plot scripts read CSVs from `../logs/` relative to themselves, so the repo
is self-contained — no absolute paths required.

## Build (low level, legs-only)

The driver is the Jetson + Pixhawk legs-only build: 3x AK60 leg motors over
SocketCAN (`can0`). IMU and propeller PWM are owned by the Pixhawk, not this
binary.

```bash
chmod +x rebuild.sh hopper_lcm_types/scripts/make_types.sh
./rebuild.sh
sudo ./build/hopper_driver
```

## Reproduce paper figures

```bash
cd hopper_controller
python3 plot_paper_figures.py     # writes ../logs/paper_figs/
python3 plot_paper_v2.py          # writes ../logs/paper_figs_v2/
```

## Notes

- Only the four CSVs the paper figures actually consume are committed here; the
  full raw log archive (~20 GB) is intentionally excluded.
- Legacy Velox-F7 tooling (`tri_balance`, `motor_id_seq`) is retired and not
  built; see the commented blocks in `CMakeLists.txt`.
