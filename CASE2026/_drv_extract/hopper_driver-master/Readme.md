# Hopper system
## Prerequisite
1. install [LCM](https://lcm-proj.github.io/lcm/) on your PC. 
## Connect to Pi
- Wire connection.
```bash
ssh pi@169.254.24.118
# pw: hopper123
```
## Start
1. connect to pi using ssh
2. run the code 
```bash
cd hopper_driver/build
./hopper_driver
```
3. On your PC:
- close WIFI
- compile the code
```bash
cd hopper_driver
sudo ./hopper_lcm_types/scripts/make_types.sh
mkdir build
cd build
cmake ..
make -j
```
- launch visualization:
```bash
./hopper_lcm_types/scripts/launch_lcm_spy.sh
```
You should see three channels:
- motor data channels
- imu channels
- gamepad channels

4. Use gamepad to control
- `X`: PD mode 
- `B` : Damping mode
- `ThumbL` + `ThumbR` + `LB` + `RB`: Set motor zero

