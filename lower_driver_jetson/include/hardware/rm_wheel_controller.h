#pragma once
// ===== RoboMaster M2006 + C610 kiwi wheels (3x, MOBILE) ==================
// Replaces the DaMiao DM-H6215 velocity-mode hubs on the dedicated Jetson
// SocketCAN bus (default can1). The folding-arm M2006s stay on Pixhawk;
// these three wheel motors share can1 only.
//
// C610 speaks CURRENT (A), not native velocity, so this class closes a
// local speed PI on the ESC feedback and emits the RoboMaster current
// frame. Upper layer still publishes wheel_cmd_lcmt.speed_des_rad_s --
// the command interface is unchanged.
//
// Protocol (C610 User Guide):
//   cmd   : CAN ID 0x200, 8 bytes, BE int16 current for IDs 1..4
//           (raw -10000..+10000 <-> -10..+10 A)
//   fb    : CAN ID 0x200 + ESC_ID (0x201..), angle / rotor_rpm / iq
//   gear  : M2006 36:1 -- commanded / reported speeds are OUTPUT shaft
//           (wheel) rad/s = rotor_rpm * 2*pi/60 / 36
//
// Safety layering (same as the old Damiao path):
//   1. HopperHardware gates on driver mode: only PD/PWMPD arm the wheels.
//   2. wheel_cmd_lcmt.enable==0 or a stale command (>200 ms) disarms.
//   3. On disarm we stream 0 A (coast). C610 holds last current if the
//      bus dies while armed -- keep the 200 ms freshness watchdog tight.

#include <stdint.h>
#include <chrono>

class RmWheelController {
public:
    static constexpr int kNumWheels = 3;
    static constexpr int kFirstEscId = 1;   // C610 IDs 1..3 on can1

    ~RmWheelController();
    bool init(const char* ifname);
    void close_bus();
    bool bus_ok() const { return fd_ >= 0; }

    // Call once per control tick.
    //   armed = leg-class mode gate AND wheel_cmd.enable AND freshness.
    //   w_des_rad_s = 3 OUTPUT-SHAFT (wheel) angular speeds [rad/s].
    void update(bool armed, const float* w_des_rad_s);

    // Telemetry (updated by update()).
    float vel_rad_s[kNumWheels] = {0.0f, 0.0f, 0.0f};  // output shaft
    float iq_a[kNumWheels] = {0.0f, 0.0f, 0.0f};
    bool online[kNumWheels] = {false, false, false};

private:
    static constexpr float kGear = 36.0f;          // M2006 reduction
    static constexpr float kIqMaxA = 5.0f;         // continuous-ish cap
    static constexpr float kKpAPerRadS = 0.20f;    // speed PI
    static constexpr float kKiAPerRadS2 = 1.0f;
    static constexpr float kDtNom = 0.002f;        // ~500 Hz control

    int fd_ = -1;
    bool armed_prev_ = false;
    float iq_i_[kNumWheels] = {0.0f, 0.0f, 0.0f};
    std::chrono::steady_clock::time_point last_tick_t_;
    bool tick_inited_ = false;

    // Self-heal (2026-08-13): the wheel CANable is on a hub and sometimes
    // drops off USB / enumerates after the driver starts ("轮子动不了,腿能动").
    // update() lazily re-opens the bus every kReinitPeriodS once can1 exists
    // again, and send_currents() detects a dead netdev (ENODEV/ENXIO after a
    // USB re-enumeration gave can1 a new ifindex) and closes the stale socket
    // so the retry path can rebind. No hopper-driver restart needed.
    static constexpr float kReinitPeriodS = 2.0f;
    char ifname_[16] = "can1";
    bool init_logged_ok_ = false;    // print bring-up/loss once per transition
    std::chrono::steady_clock::time_point last_reinit_t_;
    bool reinit_t_valid_ = false;
    void try_reinit();

    void send_currents(const float* iq_a_cmd);
    void receive();
};
