#include "hardware/tri_balance_controller.h"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <iostream>
#include <thread>

namespace {

constexpr float kDeg2Rad = static_cast<float>(M_PI) / 180.0f;
constexpr float kRad2Deg = 180.0f / static_cast<float>(M_PI);

inline float wrap_pi(float a) {
    while (a >  static_cast<float>(M_PI)) a -= 2.0f * static_cast<float>(M_PI);
    while (a < -static_cast<float>(M_PI)) a += 2.0f * static_cast<float>(M_PI);
    return a;
}

// Quaternion error: q_err = q_des^* * q_cur, returned as small-angle vector ~ 2*[x,y,z].
// We treat (roll_des, pitch_des, yaw_des) as a desired ZYX-Euler frame.
inline void euler_zyx_to_quat(float roll, float pitch, float yaw,
                              float& w, float& x, float& y, float& z) {
    const float cr = std::cos(roll * 0.5f),  sr = std::sin(roll * 0.5f);
    const float cp = std::cos(pitch * 0.5f), sp = std::sin(pitch * 0.5f);
    const float cy = std::cos(yaw * 0.5f),   sy = std::sin(yaw * 0.5f);
    w = cr*cp*cy + sr*sp*sy;
    x = sr*cp*cy - cr*sp*sy;
    y = cr*sp*cy + sr*cp*sy;
    z = cr*cp*sy - sr*sp*cy;
}

inline void quat_mul(float aw, float ax, float ay, float az,
                     float bw, float bx, float by, float bz,
                     float& w, float& x, float& y, float& z) {
    w = aw*bw - ax*bx - ay*by - az*bz;
    x = aw*bx + ax*bw + ay*bz - az*by;
    y = aw*by - ax*bz + ay*bw + az*bx;
    z = aw*bz + ax*by - ay*bx + az*bw;
}

} // namespace

TriBalanceController::TriBalanceController(const Config& cfg)
    : cfg_(cfg),
      velox_(cfg.velox_port, cfg.loop_hz),
      xbox_(cfg.xbox_path)
{
    if (!velox_.connect()) {
        std::cerr << "[tri-balance] FATAL: cannot connect to Velox F7 @ "
                  << cfg_.velox_port << std::endl;
        running_ = false;
    }
    if (!xbox_.initialize()) {
        std::cerr << "[tri-balance] WARN: xbox controller init failed @ "
                  << cfg_.xbox_path << " (controller may auto-reconnect)" << std::endl;
        // do not abort: xbox.processInput() will retry
    }
    send_disarm_pwm();
    std::cout << "[tri-balance] Ready. Press A to ARM (throttle MUST be at minimum)."
              << std::endl;
}

TriBalanceController::~TriBalanceController() {
    armed_ = false;
    send_disarm_pwm();
    std::this_thread::sleep_for(std::chrono::milliseconds(50));
    velox_.disconnect();
}

void TriBalanceController::stop() { running_ = false; }

void TriBalanceController::send_disarm_pwm() {
    velox_.sendDirectMotorCommands(static_cast<uint16_t>(cfg_.pwm_min),
                                   static_cast<uint16_t>(cfg_.pwm_min),
                                   static_cast<uint16_t>(cfg_.pwm_min),
                                   static_cast<uint16_t>(cfg_.pwm_min),
                                   static_cast<uint16_t>(cfg_.pwm_min),
                                   static_cast<uint16_t>(cfg_.pwm_min));
}

uint16_t TriBalanceController::thrust_to_pwm(float thrust_n) const {
    float t = thrust_n;
    if (t < 0.0f) t = 0.0f;
    float pwm_us = cfg_.pwm_min + std::sqrt(t / cfg_.k_thrust);
    if (pwm_us < cfg_.pwm_idle) pwm_us = cfg_.pwm_idle;
    if (pwm_us > cfg_.pwm_max)  pwm_us = cfg_.pwm_max;
    return static_cast<uint16_t>(pwm_us);
}

void TriBalanceController::update_state() {
    VeloxImuData imu;
    velox_.getImuData(imu);

    roll_  = imu.euler.data[0] * kDeg2Rad;
    pitch_ = imu.euler.data[1] * kDeg2Rad;
    yaw_   = wrap_pi(imu.euler.data[2] * kDeg2Rad);
    quat_w_ = imu.quaternion.data[0];
    quat_x_ = imu.quaternion.data[1];
    quat_y_ = imu.quaternion.data[2];
    quat_z_ = imu.quaternion.data[3];

    // gyro from Velox is deg/s -> rad/s, with low-pass filter
    const float p_raw = imu.gyroIIBiasCalibrated.data[0] * kDeg2Rad;
    const float q_raw = imu.gyroIIBiasCalibrated.data[1] * kDeg2Rad;
    const float r_raw = imu.gyroIIBiasCalibrated.data[2] * kDeg2Rad;
    const float a = clampf(cfg_.gyro_lpf_alpha, 0.0f, 1.0f);
    p_lpf_ = a * p_raw + (1.0f - a) * p_lpf_;
    q_lpf_ = a * q_raw + (1.0f - a) * q_lpf_;
    r_lpf_ = a * r_raw + (1.0f - a) * r_lpf_;
}

void TriBalanceController::update_setpoints(float dt) {
    xbox_.processInput();
    const auto& m = xbox_.getMap();

    // axis values are int16-scaled; normalize to [-1, +1].
    // Yaw stick (lx) is intentionally NOT used: this controller only stabilizes
    // roll/pitch and total thrust Fz per user request.
    const float ly = clampf(static_cast<float>(m.ly) / 32767.0f, -1.0f, 1.0f);
    const float rx = clampf(static_cast<float>(m.rx) / 32767.0f, -1.0f, 1.0f);
    const float ry = clampf(static_cast<float>(m.ry) / 32767.0f, -1.0f, 1.0f);

    // Note: js axes are inverted on Y by convention (UP is negative). We invert sign here
    // so "stick up" -> positive throttle / pitch-up command.
    const float ly_n = -deadzone(ly, cfg_.stick_deadzone);
    const float ry_n = -deadzone(ry, cfg_.stick_deadzone);
    const float rx_n =  deadzone(rx, cfg_.stick_deadzone);

    // ---- Throttle: TOTAL thrust additive from left-stick Y (user request) ----
    const float Fz_hover = cfg_.hover_thrust_factor * cfg_.mass_kg * cfg_.g;
    const float Fz_stick = cfg_.sign_throttle * ly_n * cfg_.Fz_stick_n;
    Fz_total_des_ = Fz_hover + Fz_stick;
    if (Fz_total_des_ < 0.0f) Fz_total_des_ = 0.0f;

    // ---- Roll/Pitch attitude command from right stick (with smoothing) ----
    const float roll_cmd_raw  = cfg_.sign_roll  * rx_n * cfg_.roll_pitch_full_rad;
    const float pitch_cmd_raw = cfg_.sign_pitch * ry_n * cfg_.roll_pitch_full_rad;
    const float a = clampf(cfg_.att_cmd_lpf_alpha, 0.0f, 1.0f);
    roll_cmd_  = a * roll_cmd_raw  + (1.0f - a) * roll_cmd_;
    pitch_cmd_ = a * pitch_cmd_raw + (1.0f - a) * pitch_cmd_;
    roll_cmd_  = clampf(roll_cmd_,  -cfg_.max_tilt_rad, cfg_.max_tilt_rad);
    pitch_cmd_ = clampf(pitch_cmd_, -cfg_.max_tilt_rad, cfg_.max_tilt_rad);

    // ---- Buttons (rising edge) ----
    if (m.a && !last_a_) try_arm();
    if (m.b && !last_b_) disarm("B pressed");
    if (m.y && !last_y_) {
        i_p_ = i_q_ = 0.0f;
        prev_p_err_ = prev_q_err_ = 0.0f;
        std::cout << "[tri-balance] integrators reset" << std::endl;
    }
    last_a_ = m.a;
    last_b_ = m.b;
    last_y_ = m.y;

    // ---- Tilt safety ----
    const float tilt = std::sqrt(roll_*roll_ + pitch_*pitch_);
    if (cfg_.auto_disarm_on_tilt && armed_ && tilt > cfg_.tilt_safe_rad) {
        char buf[96];
        std::snprintf(buf, sizeof(buf), "tilt %.1f deg > %.1f deg",
                      tilt * kRad2Deg, cfg_.tilt_safe_rad * kRad2Deg);
        disarm(buf);
    }

    // ---- Arming ramp ----
    if (armed_ && arm_ramp_phase_ < 1.0) {
        arm_ramp_phase_ += dt / std::max(1e-3f, cfg_.arm_ramp_s);
        if (arm_ramp_phase_ > 1.0) arm_ramp_phase_ = 1.0;
        Fz_total_des_ *= static_cast<float>(arm_ramp_phase_);
    }
}

void TriBalanceController::try_arm() {
    if (armed_) return;
    const auto& m = xbox_.getMap();
    const float ly_n = -static_cast<float>(m.ly) / 32767.0f;
    if (std::fabs(ly_n) > cfg_.arm_throttle_thresh) {
        std::cout << "[tri-balance] ARM REJECTED: throttle stick not at minimum (|ly|="
                  << std::fabs(ly_n) << ")" << std::endl;
        return;
    }
    armed_ = true;
    arm_ramp_phase_ = 0.0;
    i_p_ = i_q_ = 0.0f;
    prev_p_err_ = prev_q_err_ = 0.0f;
    std::cout << "[tri-balance] ARMED" << std::endl;
}

void TriBalanceController::disarm(const char* reason) {
    if (armed_) {
        std::cout << "[tri-balance] DISARM (" << (reason ? reason : "?") << ")" << std::endl;
    }
    armed_ = false;
    arm_ramp_phase_ = 0.0;
    i_p_ = i_q_ = 0.0f;
    prev_p_err_ = prev_q_err_ = 0.0f;
}

void TriBalanceController::compute_torques(float dt, float& Mx, float& My) {
    // ----- Outer loop: roll/pitch angle -> body rate desired -----
    // Quaternion attitude error (small-angle approx): q_err = q_des^* * q_cur.
    // Yaw is NOT controlled, so we use the *current* yaw as the desired yaw,
    // which makes ez ≈ 0 and the controller never tries to correct heading.
    float qd_w, qd_x, qd_y, qd_z;
    euler_zyx_to_quat(roll_cmd_, pitch_cmd_, yaw_, qd_w, qd_x, qd_y, qd_z);
    // q_des^*  ->  conjugate
    qd_x = -qd_x; qd_y = -qd_y; qd_z = -qd_z;
    float qe_w, qe_x, qe_y, qe_z;
    quat_mul(qd_w, qd_x, qd_y, qd_z,
             quat_w_, quat_x_, quat_y_, quat_z_,
             qe_w, qe_x, qe_y, qe_z);
    if (qe_w < 0.0f) { qe_w = -qe_w; qe_x = -qe_x; qe_y = -qe_y; qe_z = -qe_z; }
    // small-angle vector (rad). ez (yaw error) is intentionally ignored.
    const float ex = 2.0f * qe_x;
    const float ey = 2.0f * qe_y;

    float p_des = -cfg_.kp_att_rp * ex;  // negative so rate drives the error to zero
    float q_des = -cfg_.kp_att_rp * ey;
    p_des = clampf(p_des, -cfg_.max_rate_rp_rad,  cfg_.max_rate_rp_rad);
    q_des = clampf(q_des, -cfg_.max_rate_rp_rad,  cfg_.max_rate_rp_rad);

    // ----- Inner loop: body rate -> torque (roll & pitch only) -----
    const float p_err = p_des - p_lpf_;
    const float q_err = q_des - q_lpf_;

    i_p_ = clampf(i_p_ + cfg_.ki_rate_p * p_err * dt, -cfg_.i_clip_pq, cfg_.i_clip_pq);
    i_q_ = clampf(i_q_ + cfg_.ki_rate_q * q_err * dt, -cfg_.i_clip_pq, cfg_.i_clip_pq);

    const float dp = (p_err - prev_p_err_) / std::max(dt, 1e-4f);
    const float dq = (q_err - prev_q_err_) / std::max(dt, 1e-4f);
    prev_p_err_ = p_err; prev_q_err_ = q_err;

    Mx = cfg_.kp_rate_p * p_err + i_p_ + cfg_.kd_rate_p * dp;
    My = cfg_.kp_rate_q * q_err + i_q_ + cfg_.kd_rate_q * dq;
    // Mz is NOT computed -- yaw is uncontrolled per user request.
}

void TriBalanceController::mixer(float Fz, float Mx, float My,
                                 std::array<float,3>& T_arm) const {
    // Per-arm vertical thrust from (Fz, Mx, My) only -- yaw is NOT controlled.
    // Generic 3x3 inverse for arbitrary {theta_1, theta_2, theta_3} arm layout:
    //   A * T = [Fz, Mx, My]^T,
    //   A = [[1, 1, 1],
    //        [L*sin(t1), L*sin(t2), L*sin(t3)],
    //        [-L*cos(t1), -L*cos(t2), -L*cos(t3)]]
    const float L = std::max(cfg_.arm_len_m, 1e-3f);
    const float t1 = cfg_.arm_angle_deg[0] * kDeg2Rad;
    const float t2 = cfg_.arm_angle_deg[1] * kDeg2Rad;
    const float t3 = cfg_.arm_angle_deg[2] * kDeg2Rad;
    const float A[3][3] = {
        {1.0f, 1.0f, 1.0f},
        {L*std::sin(t1), L*std::sin(t2), L*std::sin(t3)},
        {-L*std::cos(t1), -L*std::cos(t2), -L*std::cos(t3)}
    };
    const float b[3] = { Fz, Mx, My };
    const float det =
        A[0][0]*(A[1][1]*A[2][2]-A[1][2]*A[2][1])
      - A[0][1]*(A[1][0]*A[2][2]-A[1][2]*A[2][0])
      + A[0][2]*(A[1][0]*A[2][1]-A[1][1]*A[2][0]);
    float T[3] = {Fz/3.0f, Fz/3.0f, Fz/3.0f};
    if (std::fabs(det) > 1e-9f) {
        float inv[3][3];
        inv[0][0] =  (A[1][1]*A[2][2]-A[1][2]*A[2][1]) / det;
        inv[0][1] = -(A[0][1]*A[2][2]-A[0][2]*A[2][1]) / det;
        inv[0][2] =  (A[0][1]*A[1][2]-A[0][2]*A[1][1]) / det;
        inv[1][0] = -(A[1][0]*A[2][2]-A[1][2]*A[2][0]) / det;
        inv[1][1] =  (A[0][0]*A[2][2]-A[0][2]*A[2][0]) / det;
        inv[1][2] = -(A[0][0]*A[1][2]-A[0][2]*A[1][0]) / det;
        inv[2][0] =  (A[1][0]*A[2][1]-A[1][1]*A[2][0]) / det;
        inv[2][1] = -(A[0][0]*A[2][1]-A[0][1]*A[2][0]) / det;
        inv[2][2] =  (A[0][0]*A[1][1]-A[0][1]*A[1][0]) / det;
        for (int i = 0; i < 3; i++) {
            T[i] = inv[i][0]*b[0] + inv[i][1]*b[1] + inv[i][2]*b[2];
        }
    }

    // Per-arm clamp (keep >= floor, <= cap). When saturated, the corresponding
    // moment contribution is lost -- standard "saturated mixer" behaviour.
    for (int i = 0; i < 3; i++) {
        T_arm[i] = clampf(T[i], cfg_.thrust_per_arm_min_n, cfg_.thrust_per_arm_max_n);
    }
}

void TriBalanceController::send_armed_pwm(const std::array<float,3>& T_arm) {
    // No yaw control: split each arm equally between top and bottom prop.
    // The CW/CCW counter-rotation of the coaxial pair cancels reaction torque
    // naturally on a balanced rig.
    float pwm[6];
    for (int i = 0; i < 6; i++) pwm[i] = cfg_.pwm_min;
    for (int i = 0; i < 3; i++) {
        const float t_half = 0.5f * T_arm[i];
        const int it = cfg_.pwm_top_idx[i];
        const int ib = cfg_.pwm_bot_idx[i];
        if (it >= 0 && it < 6) pwm[it] = static_cast<float>(thrust_to_pwm(t_half));
        if (ib >= 0 && ib < 6) pwm[ib] = static_cast<float>(thrust_to_pwm(t_half));
    }
    velox_.sendDirectMotorCommands(
        static_cast<uint16_t>(pwm[0]), static_cast<uint16_t>(pwm[1]),
        static_cast<uint16_t>(pwm[2]), static_cast<uint16_t>(pwm[3]),
        static_cast<uint16_t>(pwm[4]), static_cast<uint16_t>(pwm[5]));
}

void TriBalanceController::run() {
    using clock = std::chrono::steady_clock;
    const auto period = std::chrono::microseconds(1000000 / std::max(1, cfg_.loop_hz));
    auto next = clock::now();
    auto t_prev = next;

    while (running_) {
        const auto t_now = clock::now();
        float dt = std::chrono::duration<float>(t_now - t_prev).count();
        if (dt < 1e-4f) dt = 1.0f / cfg_.loop_hz;
        if (dt > 0.05f) dt = 0.05f;
        t_prev = t_now;

        update_state();
        update_setpoints(dt);

        if (!armed_) {
            send_disarm_pwm();
        } else {
            float Mx, My;
            compute_torques(dt, Mx, My);

            std::array<float,3> T_arm{};
            mixer(Fz_total_des_, Mx, My, T_arm);
            send_armed_pwm(T_arm);
        }

        if (cfg_.print_status && (step_counter_ % cfg_.print_every_steps == 0)) {
            std::printf(
                "[tri] %s rpy=%+5.1f,%+5.1f,%+5.1f deg | "
                "des r/p=%+5.1f,%+5.1f deg | "
                "Fz*=%5.1fN | rates p,q=%+5.1f,%+5.1f dps\n",
                armed_ ? "ARMED  " : "DISARM ",
                roll_*kRad2Deg, pitch_*kRad2Deg, yaw_*kRad2Deg,
                roll_cmd_*kRad2Deg, pitch_cmd_*kRad2Deg,
                Fz_total_des_,
                p_lpf_*kRad2Deg, q_lpf_*kRad2Deg);
            std::fflush(stdout);
        }

        step_counter_++;
        next += period;
        if (next < clock::now()) next = clock::now();  // recover from overrun
        std::this_thread::sleep_until(next);
    }

    // graceful shutdown
    send_disarm_pwm();
}
