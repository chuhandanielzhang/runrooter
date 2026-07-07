#include <chrono>
#include <thread>
#include <cmath>
#include <vector>
#include "utility.h"  // Your CAN interface class

// Constants
const int MOTOR_COUNT_PER_PORT = 3;  // Number of motors per CAN port
const int CAN_PORT_COUNT = 2;        // Number of CAN ports
const int TOTAL_MOTORS = MOTOR_COUNT_PER_PORT * CAN_PORT_COUNT;  // Total motors
const int LOOP_FREQ = 500;           // 500Hz control loop
const float Kp = 5.0;               // Proportional gain
const float Kd = 0.5;               // Derivative gain
const float T_ff = 0.0;             // Feedforward torque (optional)

class HopperController {
private:
    std::vector<canChannel> can_ports;
    std::vector<float> target_positions;
    std::vector<float> motor_states;
    std::chrono::high_resolution_clock::time_point loop_start;
    
    // Motor state buffer: [pos, vel, torque] for each motor
    // Indexed as: motor_states[motor_id * 3 + state_index]
    // where motor_id goes from 0 to TOTAL_MOTORS-1
    // and state_index: 0=position, 1=velocity, 2=torque

public:
    HopperController() : can_ports(CAN_PORT_COUNT), 
                        target_positions(TOTAL_MOTORS, 0.5f),
                        motor_states(TOTAL_MOTORS * 3, 0.0f) {
        // Initialize CAN ports
        for (int port = 0; port < CAN_PORT_COUNT; port++) {
            can_ports[port].comm_init(port + 1, MOTOR_COUNT_PER_PORT);  // +1 because port_num is 1-based
        }
        
        // Enable motors (exit safe mode) for all ports
        for (int port = 0; port < CAN_PORT_COUNT; port++) {
            for (int motor = 0; motor < MOTOR_COUNT_PER_PORT; motor++) {
                can_ports[port].send_EnterMotorMode(motor + 1);  // Motors are 1-indexed
            }
        }
        
        loop_start = std::chrono::high_resolution_clock::now();
    }
    
    ~HopperController() {
        // Disable motors for all ports
        for (int port = 0; port < CAN_PORT_COUNT; port++) {
            for (int motor = 0; motor < MOTOR_COUNT_PER_PORT; motor++) {
                can_ports[port].send_ExitMotorMode(motor + 1);
            }
            can_ports[port].comm_close();
        }
    }
    
    // Set target position for a specific motor
    void setTargetPosition(int motor_id, float position) {
        if (motor_id >= 0 && motor_id < TOTAL_MOTORS) {
            target_positions[motor_id] = position;
        }
    }
    
    // Get motor state (position, velocity, torque)
    void getMotorState(int motor_id, float& position, float& velocity, float& torque) {
        if (motor_id >= 0 && motor_id < TOTAL_MOTORS) {
            int base_index = motor_id * 3;
            position = motor_states[base_index];
            velocity = motor_states[base_index + 1];
            torque = motor_states[base_index + 2];
        }
    }
    
    // Update motor states from CAN data
    void updateMotorStates() {
        for (int port = 0; port < CAN_PORT_COUNT; port++) {
            can_ports[port].receive_process_frame();
            
            // Copy motor states from this port to our buffer
            for (int motor = 0; motor < MOTOR_COUNT_PER_PORT; motor++) {
                int global_motor_id = port * MOTOR_COUNT_PER_PORT + motor;
                int base_index = global_motor_id * 3;
                
                // Copy from canChannel's motor_state (1-indexed) to our buffer
                motor_states[base_index] = can_ports[port].motor_state[motor + 1][0];     // position
                motor_states[base_index + 1] = can_ports[port].motor_state[motor + 1][1]; // velocity
                motor_states[base_index + 2] = can_ports[port].motor_state[motor + 1][2]; // torque
            }
        }
    }
    
    // Send motor commands to all motors
    void sendMotorCommands() {
        float motor_cmd[5];  // [p_des, v_des, Kp, Kd, T_ff]
        
        for (int port = 0; port < CAN_PORT_COUNT; port++) {
            for (int motor = 0; motor < MOTOR_COUNT_PER_PORT; motor++) {
                int global_motor_id = port * MOTOR_COUNT_PER_PORT + motor;
                
                // Pack motor command
                motor_cmd[0] = target_positions[global_motor_id];  // Position setpoint
                motor_cmd[1] = 0.0;                               // Velocity damping (D term)
                motor_cmd[2] = Kp;                                // Position gain
                motor_cmd[3] = Kd;                                // Velocity gain
                motor_cmd[4] = T_ff;                              // Feedforward torque
                
                // Send command to this motor on this port
                can_ports[port].send_motor_cmd(motor + 1, motor_cmd);
            }
        }
    }
    
    // Main control loop
    void runControlLoop() {
        while (true) {
            // Update motor states from CAN
            updateMotorStates();
            
            // Send motor commands
            sendMotorCommands();
            
            // Print motor 0's position (for debugging)
            float pos, vel, torque;
            getMotorState(0, pos, vel, torque);
            printf("Motor 0 Position: %.3f\n", pos);
            
            // Strict timing for 500Hz loop
            loop_start += std::chrono::microseconds(2000);  // 500Hz = 2000µs
            std::this_thread::sleep_until(loop_start);
        }
    }
    
    // Alternative: run control loop for a specified number of iterations
    void runControlLoop(int iterations) {
        for (int i = 0; i < iterations; i++) {
            updateMotorStates();
            sendMotorCommands();
            
            // Print motor 0's position (for debugging)
            float pos, vel, torque;
            getMotorState(0, pos, vel, torque);
            printf("Motor 0 Position: %.3f\n", pos);
            
            // Strict timing for 500Hz loop
            loop_start += std::chrono::microseconds(2000);  // 500Hz = 2000µs
            std::this_thread::sleep_until(loop_start);
        }
    }
};

int main() {
    // Create and run the hopper controller
    HopperController controller;
    
    // Set different target positions for each motor (optional)
    controller.setTargetPosition(0, 0.5f);   // Motor 0 on CAN0
    controller.setTargetPosition(1, 0.3f);   // Motor 1 on CAN0
    controller.setTargetPosition(2, 0.7f);   // Motor 2 on CAN0
    controller.setTargetPosition(3, 0.2f);   // Motor 0 on CAN1
    controller.setTargetPosition(4, 0.4f);   // Motor 1 on CAN1
    controller.setTargetPosition(5, 0.6f);   // Motor 2 on CAN1
    
    // Run the control loop
    controller.runControlLoop();
    
    return 0;
}