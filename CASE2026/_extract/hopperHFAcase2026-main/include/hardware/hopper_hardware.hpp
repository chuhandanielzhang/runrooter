#pragma once
#include "orientation_tools.h"

#include "PeriodicTask.h"
#include <cmath>
#include <array>
#include <chrono>
#include <mutex>
#include <vector>

#include <lcm/lcm-cpp.hpp>
#include "gamepad_lcmt.hpp"
#include "hopper_cmd_lcmt.hpp"
#include "hopper_data_lcmt.hpp"
#include "motor_pwm_lcmt.hpp"
#include "ak60_controller.h"
#include "xbox_controller.hpp"

#define NUM_MOTORS 3
using namespace ori;
class HopperHardware
{
    public:
        HopperHardware(bool is_publish_lcm_data);
        ~HopperHardware();
        PeriodicTaskManager task_manager_;
        PeriodicMemberFunction<HopperHardware> lcmreceiveTask;
        PeriodicMemberFunction<HopperHardware> lcmsendTask;
        PeriodicMemberFunction<HopperHardware> gamepadTask;
        void _thread_lcmrec_run();
        void _thread_lcmsen_run();
        void _thread_xbox_run();
        void set_value_zero();
        void start_threads();
        void step_with_only_receiving();
        void step_with_damping();
        void step_with_pd_control();
        void step_with_pd_pwm_control();
        // Propeller-only mode: leg motors are left free (zero torque, kp=kd=0)
        // while propellers are driven by motor_pwm_lcmt (gated by control_mode > 0).
        // Use this to test/tune props independently of the leg.
        void step_with_pwm_only();
        void step_with_set_zero_mode();
        XboxController::XboxMap get_xbox_map();
        // Remote safety override (from controller PC):
        // If motor_pwm_lcmt.control_mode < 0, main.cpp can force DAMP mode (same as pressing B).
        int get_motor_pwm_control_mode();
        // One-shot remote safety request: after main.cpp consumes a negative control_mode and forces DAMP,
        // clear the stored value so the gamepad can freely switch modes afterwards.
        void clear_motor_pwm_control_mode();
        // Counter to track number of steps and prevent overwhelming the system
        // Used to rate limit control loop execution
        int step_counter = 0;
    private:
        lcm::LCM Controller2Robot;
        lcm::LCM Robot2Controller;
        void handleController2RobotLCM(const lcm::ReceiveBuffer* rbuf,
                                const std::string& chan,
                                const hopper_cmd_lcmt* msg);
        void handleMotorPwmLCM(const lcm::ReceiveBuffer* rbuf,
                               const std::string& chan,
                               const motor_pwm_lcmt* msg);
        void handleGamepadLCM(const lcm::ReceiveBuffer* rbuf,
                              const std::string& chan,
                              const gamepad_lcmt* msg);
        gamepad_lcmt gamepad_cmd_lcmt_;
        hopper_cmd_lcmt hopper_cmd_lcmt_;
        hopper_data_lcmt hopper_data_lcmt_;
        // motor_pwm_lcmt is still subscribed: we only read control_mode for the
        // remote SAFE override (control_mode < 0 -> force DAMP). Propeller PWM
        // itself is now produced on the Pixhawk side (px4_bridge), not here.
        motor_pwm_lcmt motor_pwm_lcmt_;
        AK60Controller* ak60_controller_ptr_ = nullptr;
        XboxController* xbox_controller_ptr_ = nullptr;
        bool is_publish_lcm_data=false;
        void _fill_in_motor_data_to_lcm();
        void _fill_in_gamepad_data_to_lcm();
        std::vector<float> motor_pos;
        std::vector<float> motor_vel;
        std::vector<float> motor_tau;
        std::mutex lcm_cmd_mutex;
        // Protect motor_pos/motor_vel/motor_tau which are written in the main control loop
        // and read by the LCM send task in a different thread.
        std::mutex motor_state_mutex;

        // ===== qd-from-position estimator state (LCM convention) =====
        // We publish:
        //   q_lcm  = -motor_pos + offset
        // We want:
        //   qd_lcm = d/dt (q_lcm)
        // computed from q_lcm difference + EWMA smoothing.
        std::array<float, 3> q_lcm_prev_{{0.0f, 0.0f, 0.0f}};
        std::array<float, 3> qd_lcm_ema_{{0.0f, 0.0f, 0.0f}};
        bool qd_est_inited_ = false;
        bool qd_last_ts_inited_ = false;
        std::chrono::steady_clock::time_point qd_last_ts_;
};