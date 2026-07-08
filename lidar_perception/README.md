# lidar_perception — Mid-360 巡视 / 建图 / 定位子系统

Livox Mid-360S(网线接 PC `enp44s0`,雷达 IP `192.168.1.186`,host `192.168.1.2`)
→ livox_ros_driver2 → Point-LIO(建图+里程计)→ **LCM 桥** → ModeE 控制器融合 + 巡视。

```
Mid-360S ─eth─> livox_ros_driver2 ─/livox/lidar,/livox/imu─> Point-LIO
                                                    │ /aft_mapped_to_init (odom)
                                                    v
                                    odom_lcm_bridge.py(外参 + FLU/ENU→FRD/NED)
                                                    │ hopper_odom_lcmt
                          ┌─────────────────────────┴──────────────┐
                          v                                        v
     modee/lcm_controller.py + core.py                      patrol.py(航点巡视)
     (XY 位置 + yaw 漂移校正, 慢互补)                          │ hopper_nav_cmd_lcmt
                          ^────────────────────────────────────────┘
```

## 一次性安装(需要密码)

```bash
sudo bash /home/abc/Hopper/lidar_ws/install_ros2_humble.sh   # ROS2 Humble + SDK2 → /usr/local
bash /home/abc/Hopper/lidar_ws/build_ws.sh                   # colcon 编译 driver + point_lio
pip install --user open3d                                    # 仅 relocalize.py 需要
```

## 日常使用

| 场景 | 命令 |
|---|---|
| 建图(手持/推着走) | `bash run_lidar_stack.sh map` → 走一圈 → Ctrl-C → `bash save_map.sh` |
| 巡视/定位(平时) | `bash run_lidar_stack.sh`(桥自动加载重定位变换,若有) |
| 开机重定位到已存地图 | 栈起来后机器人**静止**,`/usr/bin/python3 relocalize.py`,然后重启栈 |
| 记录航点 | 把机器人搬到目标点 → `python3 record_waypoint.py`(写入 `waypoints.yaml`) |
| 巡视 | `python3 patrol.py`(可与控制器同机常驻;手柄 **SELECT** 接入/退出) |

控制器侧(`run_modee.py` 照常启动)自动生效:
- 收到健康的 `hopper_odom_lcmt` 后,`core.py` 用慢互补把 `_p_hat_w` 的 XY 和 yaw
  拉向雷达位姿(z 和速度仍由腿运动学/KF 主导,不受影响);odom 停发/劣化 0.4 s
  自动回纯航位推算。状态行出现 `+LIDAR[x,y]`。
- **SELECT** 切换巡视:`patrol.py` 的速度指令替代右摇杆(上限 0.5 m/s);
  **任何摇杆输入或 B 立即夺回**;nav 指令超时 0.3 s 时保持原地(v=0)。
  状态行出现 `+PATROL(wpN)`。

## 关键文件

| 文件 | 作用 |
|---|---|
| `perception_config.yaml` | 外参(`extrinsic`)、健康门控、巡视参数,**装机后必改 `t_bl_m`/`rpy_deg`** |
| `odom_lcm_bridge.py` | ROS odom → `hopper_odom_lcmt`(坐标变换、杆臂、跳变/超时门控、重定位变换) |
| `patrol.py` | 航点 P 控制 → `hopper_nav_cmd_lcmt`(odom 不健康自动 active=0) |
| `relocalize.py` | 已存地图上的一次性重定位(FPFH+RANSAC → ICP),写 `maps/T_map_odom.npy` |
| `waypoints.yaml` / `record_waypoint.py` | 巡视航点(map 系 XY) |

## 坐标系约定(容易翻车,写死在这)

- **hopper world/map**:NED 风格,**+Z 向下**,x = 上电时机头方向;机体 FRD。
- Point-LIO odom(`camera_init`):重力对齐,**z 向上**。桥内用 `diag(1,-1,-1)`
  翻到 +Z down;雷达 FLU → 机体 FRD 由 `extrinsic.rpy_deg`(顶装正向 = `[180,0,0]`)。
- `extrinsic.t_bl_m` = 雷达原点在机体 FRD 系的坐标(**FRD 的 -Z 是上方**)。

## 上机联调顺序(M6)

1. 雷达装机,量外参填 `perception_config.yaml`;
2. 静置:`run_lidar_stack.sh rviz`,确认 odom 稳定、桥每 5 s 打印的 pos/yaw 合理;
3. 手提机器人走动:确认 `+LIDAR` 出现、`p_hat` 跟随不漂;
4. 原地跳(先低高度):盯桥的 `bad=` 计数和 odom 跳变(触地冲击考验 Point-LIO;
   若发散,调大 `mid360.yaml` 的 `satu_acc`/`satu_gyro` 或给雷达加减振);
5. 录航点 → SELECT 巡视,先 0.2 m/s(`patrol.v_max_mps`)。

## 已知边界

- Mid-360S 需要 SDK2 ≥1.3.1 / driver ≥1.2.6(本仓库克隆版本已满足,dev_type=35)。
- Point-LIO/driver/桥都跑在 PC 上;要无线跳跃需把整套迁到 Jetson(未做,CPU 余量待测)。
- 巡视第一版无避障:航点务必画在空旷处。
