#pragma once
// Ethercat headfiles
#include "orientation_tools.h"

// Common headfiles
#include "PeriodicTask.h"
#include <mutex>
#include <cmath>
// LCM headfiles
#include <lcm/lcm-cpp.hpp>
#include "gamepad_lcmt.hpp"
#include "hopper_cmd_lcmt.hpp"
#include "hopper_data_lcmt.hpp"
#include "hopper_imu_lcmt.hpp"
#include "ak60_controller.h"
#include "ImuWrapper.h"
#include "xbox_controller.hpp"
#include <mutex>
#define RST  "\x1B[0m"
#define KRED  "\x1B[31m"
#define KGRN  "\x1B[32m"
#define KYEL  "\x1B[33m"
#define KBLU  "\x1B[34m"
#define KMAG  "\x1B[35m"
#define KCYN  "\x1B[36m"
#define KWHT  "\x1B[37m"
# define NUM_MOTORS 3
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
        void step_with_set_zero_mode();
        XboxController::XboxMap get_xbox_map();
        // Counter to track number of steps and prevent overwhelming the system
        // Used to rate limit control loop execution
        int step_counter = 0;
    private:
        lcm::LCM Controller2Robot;
        lcm::LCM Robot2Controller;
        void handleController2RobotLCM(const lcm::ReceiveBuffer* rbuf,
                                const std::string& chan,
                                const hopper_cmd_lcmt* msg);
        gamepad_lcmt gamepad_cmd_lcmt_;
        hopper_cmd_lcmt hopper_cmd_lcmt_;
        hopper_data_lcmt hopper_data_lcmt_;
        hopper_imu_lcmt hopper_imu_lcmt_;
        AK60Controller* ak60_controller_ptr_;
        ImuWrapper* imu_wrapper_ptr;
        IG1ImuDataI imu_raw_data;
        IG1ImuDataI imu_raw_data_compensated;
        void imu_compensation(IG1ImuDataI* source, IG1ImuDataI* final);
        XboxController* xbox_controller_ptr_;
        bool is_publish_lcm_data=false;
        void _fill_in_motor_data_to_lcm();
        void _fill_in_imu_data_to_lcm();
        void _fill_in_gamepad_data_to_lcm();
        std::vector<float> motor_pos;
        std::vector<float> motor_vel;
        std::vector<float> motor_tau;
        std::mutex lcm_cmd_mutex;
        Eigen::Matrix3f rot_offset;
        Eigen::Vector3f eigen_gyro, eigen_acc,eigen_rpy;
        Eigen::Vector3f eigen_gyro_rotated, eigen_acc_rotated;
        Eigen::Vector4f eigen_quat, eigen_quat_rotated;
};
