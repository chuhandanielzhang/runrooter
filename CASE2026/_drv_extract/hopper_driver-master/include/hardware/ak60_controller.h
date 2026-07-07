#ifndef HOPPER_CONTROLLER_H
#define HOPPER_CONTROLLER_H

#include <chrono>
#include <thread>
#include <cmath>
#include <vector>
#include "utility.h"  // Your CAN interface class
#include <mutex>
// Constants
const int MOTOR_COUNT_PER_PORT = 3;  // Number of motors per CAN port
const int CAN_PORT_COUNT = 2;        // Number of CAN ports
const int TOTAL_MOTORS = MOTOR_COUNT_PER_PORT * CAN_PORT_COUNT;  // Total motors
const int LOOP_FREQ = 500;           // 500Hz control loop
const float Kp = 5.0;               // Proportional gain
const float Kd = 0.5;               // Derivative gain
const float T_ff = 0.0;             // Feedforward torque (optional)

class AK60Controller {
private:
    std::vector<canChannel> can_ports;
    std::vector<float> target_positions;
    std::vector<float> motor_states;
    std::vector<float> target_vel;
    std::vector<float> pos_gain;
    std::vector<float> vel_gain;
    std::vector<float> tau_ff;
    int history_size = 5;
    std::vector<std::vector<float>> dof_vel_history;  // History of velocities for each DOF
    std::vector<float> dof_vel_history_weight = {0.06,0.25,0.38,0.25,0.06};
    std::chrono::high_resolution_clock::time_point loop_start;
    
    // Motor state buffer: [pos, vel, torque] for each motor
    // Indexed as: motor_states[motor_id * 3 + state_index]
    // where motor_id goes from 0 to TOTAL_MOTORS-1
    // and state_index: 0=position, 1=velocity, 2=torque

public:
    AK60Controller();
    ~AK60Controller();
    
    // Set target position for a specific motor
    void setMotorParams(int motor_id, float position, float velocity, float tau, float kp, float kd);
    void getAllMotorState(std::vector<float>& pos, std::vector<float>& vel, std::vector<float>& tau);
    // Get motor state (position, velocity, torque)
    void getMotorState(int motor_id, float& position, float& velocity, float& torque);
    

    // Update motor states from CAN data
    void updateMotorStates();
    
    // Send motor commands to all motors
    void sendMotorCommands();
    
    // Main control loop
    void runControlLoop();

    // MIT mode control loop
    void runControlLoopMIT();
    void sendMotorCommandsMIT();
    // Alternative: run control loop for a specified number of iterations
    void runControlLoop(int iterations);

    void enableMotors();
    void disableMotors();
    void setZero();
    std::mutex motor_mutex;  // Mutex for thread-safe access to motor states and commands

};

#endif // HOPPER_CONTROLLER_H 