#pragma once
//
// Tri-Coaxial (Y6) balance flight controller.
//
// Scope (user request): control ONLY balance (roll/pitch) + total thrust Fz.
//   - No yaw control / no yaw rate / no heading hold / no Mz commanded.
//   - Each coaxial pair is split equally (top = bottom = T_i/2). Residual yaw
//     drift is left to the natural counter-torque cancellation between the
//     two CW/CCW props on each arm.
//
// Geometry: 3 arms at 120° apart, each arm carries 2 coaxial counter-rotating props.
//   arm_idx 0,1,2  ->  pwm_top_idx[i] (CCW prop)
//                  ->  pwm_bot_idx[i] (CW  prop)
//
// Allocation (body frame, x forward, y left, z up):
//   F_z = sum_i  T_i                          (total vertical thrust)
//   M_x = L * sum_i  sin(theta_i) * T_i       (roll moment)
//   M_y = -L * sum_i cos(theta_i) * T_i       (pitch moment)
// where T_i = T_i_top + T_i_bot is the per-arm vertical thrust.
//
// Closed-form inverse for 3 arms at 0/120/240 deg (general 3x3 inverse used in code):
//   T_1 = Fz/3                  - (2/3) * (M_y/L)
//   T_2 = Fz/3 + (1/sqrt3)*(Mx/L) + (1/3) * (M_y/L)
//   T_3 = Fz/3 - (1/sqrt3)*(Mx/L) + (1/3) * (M_y/L)
//
// Thrust -> PWM (Hopper4 convention):
//   pwm_us = pwm_min + sqrt( max(thrust_n,0) / k_thrust )
//
// Cascade attitude PID (roll/pitch only):
//   outer (angle -> rate):  rate_des = kp_att * att_err   (quat-based, small-angle err)
//   inner (rate  -> torque): torque  = kp*err + ki*∫err + kd*err_dot  (anti-windup)
//
// Inputs (XBox):
//   Left  stick Y  -> throttle bias  (TOTAL thrust ADD)        [user request]
//   Right stick X  -> roll  desired   [rad]
//   Right stick Y  -> pitch desired   [rad]
//   A press        -> ARM (only if throttle stick at minimum)
//   B press        -> DISARM (always; failsafe)
//   Y press        -> reset integrators
//   Tilt > tilt_safe_rad => auto DISARM.
//
// All PWM channels failsafe to pwm_min on disarm / exit / tilt-fault.

#include <atomic>
#include <array>
#include <cstdint>
#include <string>

#include "VeloxF7Wrapper.h"
#include "xbox_controller.hpp"

class TriBalanceController {
public:
    struct Config {
        // ===== Geometry / mixer =====
        float arm_len_m = 0.30f;
        // arm angles (deg) measured from body +x, CCW (right-handed about +z)
        std::array<float,3> arm_angle_deg = {0.0f, 120.0f, 240.0f};
        // Velox PWM channel indices for top (CCW) and bottom (CW) prop on each arm.
        // Default: arm0->{0,1}, arm1->{2,3}, arm2->{4,5}.
        std::array<int,3> pwm_top_idx = {0, 2, 4};
        std::array<int,3> pwm_bot_idx = {1, 3, 5};

        // ===== Prop / ESC =====
        // thrust_n = k_thrust * (pwm_us - pwm_min)^2  (Hopper4-style square law)
        float k_thrust   = 1.47e-4f;
        float pwm_min    = 1000.0f;     // disarmed
        float pwm_max    = 1700.0f;     // hard cap
        float pwm_idle   = 1050.0f;     // armed-idle (avoid stop/start jitter)
        // Per-arm hard caps (N)
        float thrust_per_arm_max_n   = 60.0f;  // hard upper cap on T_i
        float thrust_per_arm_min_n   = 0.0f;   // floor (kept >= 0)

        // ===== Vehicle =====
        float mass_kg = 2.0f;
        float g = 9.81f;

        // ===== Attitude (outer): angle [rad] -> body rate [rad/s] =====
        float kp_att_rp = 6.0f;       // roll & pitch
        float max_tilt_rad = 0.4363f; // 25 deg

        // ===== Rate (inner): rate [rad/s] -> torque [Nm] =====
        // Roll/pitch only (yaw not controlled).
        float kp_rate_p = 0.20f, ki_rate_p = 0.20f, kd_rate_p = 0.005f;
        float kp_rate_q = 0.20f, ki_rate_q = 0.20f, kd_rate_q = 0.005f;
        float i_clip_pq = 0.5f;        // |Nm|
        float gyro_lpf_alpha = 0.4f;   // 0..1, smaller = more smoothing
        // Outer-loop rate command saturation (rad/s)
        float max_rate_rp_rad = 3.1416f;  // ~180 deg/s

        // ===== Throttle mapping (left stick Y) =====
        // Total Fz_des = m*g*hover_factor + stick_y * Fz_stick_n
        //   (stick_y in [-1,1]; positive when pushed forward/up)
        float hover_thrust_factor = 1.0f;
        float Fz_stick_n          = 30.0f;   // additive total thrust at full stick

        // ===== Stick shaping =====
        float stick_deadzone   = 0.06f;
        float roll_pitch_full_rad = 0.349f; // 20 deg at full stick
        float att_cmd_lpf_alpha   = 0.5f;   // smooth desired angles
        // Sign flips (set -1 if you want stick direction reversed)
        float sign_roll  = +1.0f;
        float sign_pitch = +1.0f;
        float sign_throttle = +1.0f;

        // ===== Safety =====
        float tilt_safe_rad   = 1.0472f; // 60 deg
        bool  auto_disarm_on_tilt = true;
        float arm_throttle_thresh = 0.05f; // require |stick_y| < this to ARM
        // Smooth ramp on ARM: ramp Fz from 0 -> commanded over `arm_ramp_s` seconds
        float arm_ramp_s = 0.6f;

        // ===== Loop =====
        int   loop_hz = 500;
        std::string velox_port = "/dev/ttyACM0";
        std::string xbox_path  = "/dev/input/js0";
        bool  print_status = true;
        int   print_every_steps = 250;  // ~2 Hz at 500 Hz
    };

    explicit TriBalanceController(const Config& cfg);
    ~TriBalanceController();

    // Blocking main loop. Returns when stop() is called or fatal error.
    void run();
    void stop();

    bool isArmed() const { return armed_; }

private:
    // ---- state ----
    Config cfg_;
    VeloxF7Wrapper velox_;
    XboxController xbox_;
    std::atomic<bool> running_{true};
    bool armed_{false};

    // attitude state (rad)
    float roll_ = 0.0f, pitch_ = 0.0f, yaw_ = 0.0f;
    float quat_w_ = 1.0f, quat_x_ = 0.0f, quat_y_ = 0.0f, quat_z_ = 0.0f;
    // body rates (rad/s), low-pass filtered
    float p_lpf_ = 0.0f, q_lpf_ = 0.0f, r_lpf_ = 0.0f;

    // setpoints
    float roll_cmd_ = 0.0f, pitch_cmd_ = 0.0f;
    float Fz_total_des_ = 0.0f;     // commanded total thrust [N]

    // PID state (roll & pitch only; yaw NOT controlled)
    float i_p_ = 0.0f, i_q_ = 0.0f;
    float prev_p_err_ = 0.0f, prev_q_err_ = 0.0f;

    // arming ramp
    double arm_ramp_phase_ = 1.0; // 0->1 ramp progress

    // gamepad edge detection
    int last_a_ = 0, last_b_ = 0, last_y_ = 0;

    int step_counter_ = 0;

    // ---- helpers ----
    void update_state();
    void update_setpoints(float dt);
    // Attitude controller: roll/pitch only -> (Mx, My).  Yaw is NOT controlled.
    void compute_torques(float dt, float& Mx, float& My);
    void mixer(float Fz, float Mx, float My,
               std::array<float,3>& T_arm) const;
    uint16_t thrust_to_pwm(float thrust_n) const;
    void send_disarm_pwm();
    void send_armed_pwm(const std::array<float,3>& T_arm);
    void try_arm();
    void disarm(const char* reason);

    static float clampf(float v, float lo, float hi) {
        return v < lo ? lo : (v > hi ? hi : v);
    }
    static float deadzone(float v, float dz) {
        if (v >  dz) return (v - dz) / (1.0f - dz);
        if (v < -dz) return (v + dz) / (1.0f - dz);
        return 0.0f;
    }
};
