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
#include "hopper_imu_lcmt.hpp"
#include "motor_pwm_lcmt.hpp"
#include "rm_esc_cmd_lcmt.hpp"
#include "rm_esc_data_lcmt.hpp"
#include "ak60_controller.h"
#include "ImuWrapper.h"
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
        PeriodicMemberFunction<HopperHardware> imuTask;
        PeriodicMemberFunction<HopperHardware> gamepadTask;
        void _thread_lcmrec_run();
        void _thread_lcmsen_run();
        void _thread_imu_run();
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
        void handleRmEscDataLCM(const lcm::ReceiveBuffer* rbuf,
                                const std::string& chan,
                                const rm_esc_data_lcmt* msg);
        void handleGamepadLCM(const lcm::ReceiveBuffer* rbuf,
                              const std::string& chan,
                              const gamepad_lcmt* msg);
        gamepad_lcmt gamepad_cmd_lcmt_;
        hopper_cmd_lcmt hopper_cmd_lcmt_;
        hopper_data_lcmt hopper_data_lcmt_;
        hopper_imu_lcmt hopper_imu_lcmt_;
        // motor_pwm_lcmt is still subscribed: we only read control_mode for the
        // remote SAFE override (control_mode < 0 -> force DAMP). Propeller PWM
        // itself is now produced on the Pixhawk side (px4_bridge), not here.
        motor_pwm_lcmt motor_pwm_lcmt_;
        // ---- RM M2006/C610 (Pixhawk rm_c610 <-> px4_dds_bridge) ----
        // The M2006s are LEG-CLASS actuators: their current command is gated by
        // the SAME gamepad mode machine as the AK60s (X arms PD -> forward
        // hopper_cmd_lcmt.rm_iq_des; B/DAMP/OFF/anything else -> stream 0 A).
        // This driver is the ONLY publisher of rm_esc_cmd_lcmt; the PC never
        // talks to the bridge directly, so B always cuts the motors.
        rm_esc_data_lcmt rm_esc_data_lcmt_;
        std::chrono::steady_clock::time_point rm_data_rx_t_;
        bool rm_data_seen_ = false;
        // M2006 shaft-angle zero offset (rad). The PX4 rm_c610 module reports a
        // multi-turn angle that starts at an arbitrary value on power-up; the
        // gamepad SET_ZERO action latches the current angle here so that
        // hopper_data_lcmt.rm_q reads 0 at the zeroing pose (same UX as AK60s).
        float rm_q_offset_[3] = {0.0f, 0.0f, 0.0f};
        // Edge detector for hopper_cmd_lcmt.rm_set_zero (LCM-triggered re-zero).
        bool rm_set_zero_prev_ = false;
        // Publish the (gated) M2006 current command. Call once per control step:
        // enabled=true forwards rm_iq_des, enabled=false streams zero current.
        void _publish_rm_cmd(bool enabled);
        AK60Controller* ak60_controller_ptr_ = nullptr;
        ImuWrapper* imu_wrapper_ptr_ = nullptr;
        IG1ImuDataI imu_raw_data_;
        IG1ImuDataI imu_raw_data_compensated_;
        void imu_compensation(IG1ImuDataI* source, IG1ImuDataI* final);
        XboxController* xbox_controller_ptr_ = nullptr;
        bool is_publish_lcm_data=false;
        void _fill_in_motor_data_to_lcm();
        void _fill_in_imu_data_to_lcm();
        void _fill_in_gamepad_data_to_lcm();
        std::vector<float> motor_pos;
        std::vector<float> motor_vel;
        std::vector<float> motor_tau;
        // qd published on LCM = numerical differentiation of motor position
        // (2026-07-06: CAN-reported velocity is NOT used anywhere -- the AK60
        // internal estimate measured ~14% low in amplitude and lagged).
        std::vector<float> motor_vel_diff;
        std::vector<float> motor_pos_prev;
        bool motor_diff_init_ = false;
        std::chrono::steady_clock::time_point motor_diff_prev_t_;
        // Store one control-tick snapshot of motor state (and update the
        // position-derivative velocity). Call once per step after getMotorState.
        void _store_motor_state(const float* temp_pos, const float* temp_vel, const float* temp_tau);
        std::mutex lcm_cmd_mutex;
        // Protect motor_pos/motor_vel/motor_tau which are written in the main control loop
        // and read by the LCM send task in a different thread.
        std::mutex motor_state_mutex;

        Eigen::Matrix3f rot_offset_;
        Eigen::Vector3f eigen_gyro_, eigen_acc_, eigen_rpy_;
        Eigen::Vector3f eigen_gyro_rotated_, eigen_acc_rotated_;
        Eigen::Vector4f eigen_quat_, eigen_quat_rotated_;
};
