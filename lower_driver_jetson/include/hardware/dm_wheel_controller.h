#pragma once
// ===== DaMiao DM-H6215 hub wheels (3x, MOBILE kiwi drive) ================
// VELOCITY mode on a DEDICATED SocketCAN bus (default can1) -- the leg MIT
// bus stays clean on can0 (see the note in utility.h).
//
// Protocol (DM-H6215 manual V1.0):
//   control frame : CAN ID = 0x200 + ESC_ID, data = float32 LE v_des (rad/s)
//   enable/disable: same ID, 8 bytes 0xFF..0xFC / 0xFF..0xFD
//   mode register : write 0x7FF frame, RID 0x0A (CTRL_MODE) = 3 (velocity);
//                   volatile, so init() re-writes it on every boot -- no
//                   dependency on what was saved with the tuner.
//   feedback      : CAN ID = MST_ID (default 0); data[0] = (ERR<<4)|ESC_ID,
//                   POS16/VEL12/T12 fixed-point + MOS/rotor temperatures.
//
// Safety layering (leg-class, same as rm_iq_des):
//   1. HopperHardware gates on driver mode: only PD/PWMPD arm the wheels.
//   2. wheel_cmd_lcmt.enable==0 or a stale command (>200 ms) disarms.
//   3. The motor's own comm-loss protection exits Enable Mode if command
//      frames stop, so a dead driver can never leave a wheel spinning.

#include <stdint.h>
#include <chrono>

class DmWheelController {
public:
    static constexpr int kNumWheels = 3;
    // ESC_IDs 1..3 (set per motor with the DaMiao tuner; MST_ID may stay at
    // the default 0 -- feedback is keyed on the ID nibble in data[0]).
    static constexpr int kFirstEscId = 1;

    ~DmWheelController();
    // Bring the interface up and open a nonblocking CAN_RAW socket. Returns
    // false (and disables all later calls) if the bus does not exist, so a
    // robot without the wheel bus attached still runs normally.
    bool init(const char* ifname);
    void close_bus();
    bool bus_ok() const { return fd_ >= 0; }

    // Call once per control tick.
    //   armed = leg-class mode gate (PD/PWMPD) AND wheel_cmd_lcmt.enable
    //           AND command freshness (all resolved by the caller).
    //   w_des_rad_s = 3 wheel angular speeds.
    // Rising edge / every 500 ms while armed: mode-register write + enable
    // (covers power-up and motor self-disable after a fault or comm loss).
    // Falling edge: zero speed then disable (freewheel).
    void update(bool armed, const float* w_des_rad_s);

    // Decoded feedback, telemetry only (updated by update()).
    float vel_rad_s[kNumWheels] = {0.0f, 0.0f, 0.0f};
    uint8_t status[kNumWheels] = {0, 0, 0};   // 0 disabled, 1 enabled, >=8 fault

private:
    // Feedback fixed-point mapping range -- must equal the VMAX register in
    // the motors (decode only; the velocity COMMAND is a raw float and does
    // not depend on this).
    static constexpr float kFbVMax = 45.0f;

    int fd_ = -1;
    bool armed_prev_ = false;
    std::chrono::steady_clock::time_point last_enable_t_;

    void send_ff_tail(int esc_id, uint8_t tail);      // 0xFF x7 + tail
    void send_ctrl_mode_velocity(int esc_id);         // 0x7FF RID 0x0A = 3
    void send_speed(int esc_id, float v_rad_s);
    void receive();
};
