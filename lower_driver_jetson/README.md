# HOPPER SRB RL

Pi firmware + PC RL bridge (`rl_deploy/`).

## 1. SCP (PC → PI)

```bash
unset LD_LIBRARY_PATH
rsync -avz --delete \
  --exclude='build/' --exclude='__pycache__/' --exclude='.git/' \
  ./Hopper_srbRL/ pi@PI_IP:~/Hopper_srbRL/
```

## 2. SSH (PC → PI)

```bash
unset LD_LIBRARY_PATH
ssh pi@PI_IP
```

## 3. LCM NETWORK (PI)

```bash
sudo ifconfig eth0 multicast
sudo route add -net 224.0.0.0 netmask 240.0.0.0 dev eth0
sudo ip link set can0 up type can bitrate 1000000 restart-ms 100
```

## 4. BUILD (PI)

```bash
cd ~/Hopper_srbRL
chmod +x rebuild.sh hopper_lcm_types/scripts/make_types.sh
./rebuild.sh
```

## 5. RUN (PI)

```bash
cd ~/Hopper_srbRL
sudo ./build/hopper_driver
```

Xbox: `B` = DAMP · `X` = PD · `X` then `A` = PWMPD · `A` alone is ignored so it never turns the leg off. PC bridge: `Y` = upper-layer reset.

## 6. LCM-SPY (PC)

```bash
cd /path/to/Hopper_srbRL
sudo route add -net 224.0.0.0 netmask 240.0.0.0 dev enp44s0
./hopper_lcm_types/scripts/launch_lcm_spy.sh
```

## 7. RUN RL (PC)

```bash
pip install -r rl_deploy/requirements.txt
cd /path/to/Hopper_srbRL
python rl_deploy/bridge/runner.py --lcm \
  --cmd_vx 0.0 --cmd_vy 0.0 --cmd_apex 0.80 \
  --pwm_max_us 1600 --tau_out_max_nm 20 \
  --steps 999999 --print_every 20 \
  --log_csv logs/clean_rl_hfa_$(date +%Y%m%d_%H%M%S).csv
```

## 8. Propeller Calibration (PC)

Fix the robot/propeller on a bench with a scale/load cell. On the Pi, run
`sudo ./build/hopper_driver`, then press `X` and `A` so prop PWM is enabled.

```bash
cd /path/to/Hopper_srbRL
python rl_deploy/scripts/calibrate_prop_pwm.py --armed \
  --props all --pwm_start 1050 --pwm_stop 1600 --pwm_step 50 \
  --units g --out logs/prop_pwm_calibration_$(date +%Y%m%d_%H%M%S).csv
```

## DEMO

<img src="./demo/v4_walkstop_sim.gif" width="600" alt="walk-stop" />

<img src="./demo/hfa_rollout_sim.gif" width="600" alt="rollout" />
