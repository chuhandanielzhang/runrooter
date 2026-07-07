#include "ak60_controller.h"
#include <cstdio>

static float _read_env_float(const char* name, float def, float lo, float hi) {
    if (const char* s = std::getenv(name)) {
        const float v = std::strtof(s, nullptr);
        if (std::isfinite(v) && v >= lo && v <= hi) return v;
    }
    return def;
}

static float _read_env_float_pos(const char* name, float def, float hi) {
    return _read_env_float(name, def, 0.0f, hi);
}

AK60Controller::AK60Controller() : can_ports(CAN_PORT_COUNT), 
                        target_positions(TOTAL_MOTORS, 0.0f),
                        target_vel(TOTAL_MOTORS, 0.0f),
                        pos_gain(TOTAL_MOTORS, Kp),
                        vel_gain(TOTAL_MOTORS, Kd),
                        tau_ff(TOTAL_MOTORS, 0.0f),
                        position_offsets(TOTAL_MOTORS, 0.0f),  // +60° = +1.047 rad
                        motor_states(TOTAL_MOTORS * 3, 0.0f),
                        dof_vel_history(TOTAL_MOTORS, std::vector<float>(history_size, 0.0f)),
                        motor_pos_prev(TOTAL_MOTORS, 0.0f),
                        motor_pos_prev2(TOTAL_MOTORS, 0.0f),
                        filter_initialized(TOTAL_MOTORS, false),
                        prev2_initialized(TOTAL_MOTORS, false),
                        motor_vel_ewma(TOTAL_MOTORS, 0.0f)
                         {
    // ========== Velocity Estimation Setup ==========
    // "Super smooth, super advanced" implementation using Kalman Filter.
    // Fuses high-precision encoder position with noisy finite-difference velocity.
    //
    // Tuning notes (bigger = smoother but more lag):
    // - meas_vel_std (measurement noise of diff-velocity): increase to "strengthen filtering"
    // - proc_vel_std (process noise of velocity): decrease to "strengthen filtering"
    //
    // Runtime tuning (no recompile) via env vars:
    //   export HOPPER_DQ_KF_MEAS_VEL_STD=5.0   # stronger smoothing (default: 1.0)
    //   export HOPPER_DQ_KF_PROC_VEL_STD=0.005 # stronger smoothing (default: 0.01)
    const float kf_proc_pos_std = _read_env_float_pos("HOPPER_DQ_KF_PROC_POS_STD", 1e-4f, 1.0f);
    const float kf_proc_vel_std = _read_env_float_pos("HOPPER_DQ_KF_PROC_VEL_STD", 1e-2f, 10.0f);
    const float kf_meas_pos_std = _read_env_float_pos("HOPPER_DQ_KF_MEAS_POS_STD", 1e-6f, 1.0f);
    const float kf_meas_vel_std = _read_env_float_pos("HOPPER_DQ_KF_MEAS_VEL_STD", 1.0f, 1000.0f);
    kalman_filter = new KalmanFilter3D(
        0.001f,   // dt (1000Hz)
        kf_proc_pos_std,
        kf_proc_vel_std,
        kf_meas_pos_std,
        kf_meas_vel_std
    );
    filter_type = FILTER_KALMAN;

    // Initialize CAN ports
    for (int port = 0; port < CAN_PORT_COUNT; port++) {
        can_ports[port].comm_init(port + 1, MOTOR_COUNT_PER_PORT);  // +1 because port_num is 1-based
    }
    
    // In old codebase this was enabled right away. Let's do it to wake up the bus.
    enableMotors();
    
    loop_start = std::chrono::high_resolution_clock::now();
}

void AK60Controller::enableMotors() {
    for (int port = 0; port < CAN_PORT_COUNT; port++) {
        for (int motor = 0; motor < MOTOR_COUNT_PER_PORT; motor++) {
            can_ports[port].send_EnterMotorMode(motor + 1);  // CAN ID = 1, 2, 3
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
}

void AK60Controller::disableMotors() {
    for (int port = 0; port < CAN_PORT_COUNT; port++) {
        for (int motor = 0; motor < MOTOR_COUNT_PER_PORT; motor++) {
            can_ports[port].send_ExitMotorMode(motor + 1);  // CAN ID = 1, 2, 3
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
    }
}

void AK60Controller::setZero() {
    for (int port = 0; port < CAN_PORT_COUNT; port++) {
        for (int motor = 0; motor < MOTOR_COUNT_PER_PORT; motor++) {
            can_ports[port].send_SetZero(motor + 1);  // CAN ID = 1, 2, 3
            std::this_thread::sleep_for(std::chrono::milliseconds(500));
        }
    }
}

AK60Controller::~AK60Controller() {
    if (kalman_filter) {
        delete kalman_filter;
        kalman_filter = nullptr;
    }
    // Disable motors for all ports
    disableMotors();
    for (int port = 0; port < CAN_PORT_COUNT; port++) {
        can_ports[port].comm_close();
    }
}

// Set target position for a specific motor
void AK60Controller::setMotorParams(int motor_id, float position, float velocity, float tau, float kp, float kd) {
    if (motor_id >= 0 && motor_id < TOTAL_MOTORS) {
        target_positions[motor_id] = position;
        target_vel[motor_id] = velocity;
        tau_ff[motor_id] = tau;
        pos_gain[motor_id] = kp;
        vel_gain[motor_id] = kd;
    }
}

// Get motor state (position, velocity, torque)
void AK60Controller::getMotorState(int motor_id, float& position, float& velocity, float& torque) {
    if (motor_id >= 0 && motor_id < TOTAL_MOTORS) {
        int base_index = motor_id * 3;
        position = motor_states[base_index];
        velocity = motor_states[base_index + 1];
        torque = motor_states[base_index + 2];
    }
}

void AK60Controller::setPositionOffset(int motor_id, float offset) {
    if (motor_id >= 0 && motor_id < TOTAL_MOTORS) {
        position_offsets[motor_id] = offset;
    }
}

void AK60Controller::setAllPositionOffsets(float offset) {
    for (int i = 0; i < TOTAL_MOTORS; i++) {
        position_offsets[i] = offset;
    }
}

void AK60Controller::getAllMotorState(std::vector<float>& pos, std::vector<float>& vel, std::vector<float>& tau) {
    for(int i = 0; i < TOTAL_MOTORS; i++){
        getMotorState(i, pos[i], vel[i], tau[i]);
    }
}

// Update motor states from CAN data
void AK60Controller::updateMotorStates() {
    motor_mutex.lock();

    // Use measured dt to reduce spikes if the loop jitters.
    const auto now = std::chrono::high_resolution_clock::now();
    float dt_meas = dt;
    if (vel_last_ts_initialized) {
        dt_meas = std::chrono::duration<float>(now - vel_last_ts).count();
        if (dt_meas < 1e-4f) dt_meas = 1e-4f;
        if (dt_meas > 5e-2f) dt_meas = 5e-2f;
    } else {
        vel_last_ts_initialized = true;
    }
    vel_last_ts = now;

    // For 2-sample finite difference (reduces quantization noise):
    // dq_raw[k] = (q[k] - q[k-2]) / (dt[k] + dt[k-1])
    float dt2 = dt_meas;
    if (dt_prev_initialized) {
        dt2 = dt_meas + dt_prev;
        if (dt2 < 1e-4f) dt2 = 1e-4f;
    }
    dt_prev = dt_meas;
    dt_prev_initialized = true;

    // Arrays for Kalman Filter (first 3 motors)
    float meas_pos[3];
    float meas_vel[3]; // Raw finite difference
    
    for (int port = 0; port < CAN_PORT_COUNT; port++) {
        can_ports[port].receive_process_frame();

        // Copy motor states from this port to our buffer
        for (int motor = 0; motor < MOTOR_COUNT_PER_PORT; motor++) {
            int global_motor_id = port * MOTOR_COUNT_PER_PORT + motor;
            int base_index = global_motor_id * 3;
            int can_state_index = motor + 1;  // CAN ID = 1, 2, 3

            // Apply position offset
            float motor_pos_current = can_ports[port].motor_state[can_state_index][0] + position_offsets[global_motor_id];
            motor_states[base_index] = motor_pos_current;  // position with offset
            motor_states[base_index + 2] = can_ports[port].motor_state[can_state_index][2]; // torque

            // Compute raw finite difference velocity for estimator input
            float raw_velocity = 0.0f;
            if (!filter_initialized[global_motor_id]) {
                motor_pos_prev[global_motor_id] = motor_pos_current;
                motor_pos_prev2[global_motor_id] = motor_pos_current;
                prev2_initialized[global_motor_id] = false;
                filter_initialized[global_motor_id] = true;
                raw_velocity = 0.0f;
            } else {
                if (!prev2_initialized[global_motor_id]) {
                    // First derivative sample: use 1-step diff.
                    raw_velocity = (motor_pos_current - motor_pos_prev[global_motor_id]) / dt_meas;
                    motor_pos_prev2[global_motor_id] = motor_pos_prev[global_motor_id];
                    motor_pos_prev[global_motor_id] = motor_pos_current;
                    prev2_initialized[global_motor_id] = true;
                } else {
                    // 2-sample diff: q[k] - q[k-2] over dt[k] + dt[k-1]
                    raw_velocity = (motor_pos_current - motor_pos_prev2[global_motor_id]) / dt2;
                    motor_pos_prev2[global_motor_id] = motor_pos_prev[global_motor_id];
                    motor_pos_prev[global_motor_id] = motor_pos_current;
                }
            }

            // Clamp raw velocity to sanity bounds
            if (raw_velocity > V_MAX) raw_velocity = V_MAX;
            if (raw_velocity < V_MIN) raw_velocity = V_MIN;

            // Store for filter
            if (global_motor_id < 3) {
                meas_pos[global_motor_id] = motor_pos_current;
                meas_vel[global_motor_id] = raw_velocity;
            }
        }
    }

    // Apply Kalman Filter for the first 3 motors (Leg)
    if (filter_type == FILTER_KALMAN && kalman_filter) {
        float filt_pos[3];
        float filt_vel[3];
        kalman_filter->filter(meas_pos, meas_vel, filt_pos, filt_vel);
        
        for (int i = 0; i < 3; i++) {
            // Update the state buffer with the optimal filtered velocity
            motor_states[i * 3 + 1] = filt_vel[i];
            
            // Optionally, we could also replace position with filtered position if desired,
            // but usually raw encoder position is preferred for kinematics unless very noisy.
            // motor_states[i * 3 + 0] = filt_pos[i]; 
        }
    } else {
        // Fallback or for other motors: Just use raw diff (or add EWMA here if needed)
        // For now, we assume only motors 0-2 matter for control.
    }

    motor_mutex.unlock();
}

// Send motor commands to all motors
void AK60Controller::sendMotorCommands() {    
    float motor_cmd[5];  // [p_des, v_des, Kp, Kd, T_ff]
    
    for (int port = 0; port < CAN_PORT_COUNT; port++) {
        for (int motor = 0; motor < MOTOR_COUNT_PER_PORT; motor++) {
            int global_motor_id = port * MOTOR_COUNT_PER_PORT + motor;
            
            // Pack motor command
            // Apply offset: add offset to target position
            // When software sends target_pos = 0, motor goes to physical position = offset
            motor_cmd[0] = target_positions[global_motor_id] + position_offsets[global_motor_id];  // Position setpoint with offset
            motor_cmd[1] = target_vel[global_motor_id];                        // Velocity setpoint
            motor_cmd[2] = pos_gain[global_motor_id];                          // Position gain
            motor_cmd[3] = vel_gain[global_motor_id];                          // Velocity gain
            motor_cmd[4] = tau_ff[global_motor_id];                              // Feedforward torque
            
            // Send command to this motor on this port - CAN ID = 1, 2, 3
            can_ports[port].send_motor_cmd(motor + 1, motor_cmd);
        }
    }
}

void AK60Controller::sendMotorCommandsMIT() {
    float motor_cmd[5];  // [p_des, v_des, Kp, Kd, T_ff]
    
    for (int port = 0; port < CAN_PORT_COUNT; port++) {
        for (int motor = 0; motor < MOTOR_COUNT_PER_PORT; motor++) {
            int global_motor_id = port * MOTOR_COUNT_PER_PORT + motor;
            
            // Pack motor command
            motor_cmd[0] = 0.0f;  // Position setpoint
            motor_cmd[1] = 0.0f;                        // Velocity setpoint
            motor_cmd[2] = 0.0f;                          // Position gain
            motor_cmd[3] = 0.0f;                          // Velocity gain
            motor_cmd[4] = tau_ff[global_motor_id];                              // Feedforward torque
            
            // Send command to this motor on this port - CAN ID = 1, 2, 3
            can_ports[port].send_motor_cmd(motor + 1, motor_cmd);
        }
    }
}
// MIT mode control loop - runs with MIT-style impedance control. 
void AK60Controller::runControlLoopMIT() {
    while (true) {
        // Update motor states from CAN
        updateMotorStates();
        // Update velocity history
        for (int motor_id = 0; motor_id < TOTAL_MOTORS; motor_id++) {
            float pos, vel, torque;
            getMotorState(motor_id, pos, vel, torque);
            dof_vel_history[motor_id].push_back(vel);
            if (dof_vel_history[motor_id].size() > history_size) {
                dof_vel_history[motor_id].erase(dof_vel_history[motor_id].begin());
            }
        }
        // Calculate filtered velocity for each motor using weighted history
        std::vector<float> filtered_vel(TOTAL_MOTORS, 0.0f);
        for (int motor_id = 0; motor_id < TOTAL_MOTORS; motor_id++) {
            // Apply weights to velocity history
            for (size_t i = 0; i < dof_vel_history[motor_id].size(); i++) {
                filtered_vel[motor_id] += dof_vel_history[motor_id][i] * dof_vel_history_weight[i];
            }
        }
        printf("MIT Mode - Filtered Velocities: ");
        for(int i = 0; i < 3; i++) {
            printf("M%d=%.3f ", i, filtered_vel[i]);
        }
        printf("\n");
        // For each motor, compute MIT-style impedance control
        for (int motor_id = 0; motor_id < TOTAL_MOTORS; motor_id++) {
            float pos, vel, torque;
            getMotorState(motor_id, pos, vel, torque);
            
            // Compute position and velocity errors
            float pos_error = target_positions[motor_id] - pos;
            float vel_error = target_vel[motor_id] - vel;
            
            // MIT-style impedance control law:
            // tau = Kp*(pos_error) + Kd*(vel_error) + tau_ff
            tau_ff[motor_id] = pos_gain[motor_id] * pos_error + vel_gain[motor_id] * vel_error;
        }
        
        // Send motor commands
        sendMotorCommandsMIT();
        
        // Print all motors' states
        printf("MIT Mode - Motors State:\n");
        for(int i = 0; i < 3; i++) {
            float pos, vel, torque;
            getMotorState(i, pos, vel, torque);
            printf("  Motor %d: Pos=%.3f, Vel=%.3f, Torque=%.3f\n", i, pos, vel, torque);
        }

        // Strict timing for 10000Hz loop
        loop_start += std::chrono::microseconds(1000);  // 1000Hz = 1000µs
        std::this_thread::sleep_until(loop_start);
        // Print loop timing informatio
        
        auto current_time = std::chrono::high_resolution_clock::now();
        auto time_since_start = std::chrono::duration_cast<std::chrono::microseconds>(current_time - loop_start);
        printf("Current time: %lld ms\n", std::chrono::duration_cast<std::chrono::milliseconds>(current_time.time_since_epoch()).count());

    }
}

// Main control loop
void AK60Controller::runControlLoop() {
    while (true) {
        // Update motor states from CAN
        updateMotorStates();
        
        // Send motor commands
        sendMotorCommands();
        
        // Print all motors' states
        printf("Motors State:\n");
        for(int i = 0; i < 3; i++) {
            float pos, vel, torque;
            getMotorState(i, pos, vel, torque);
            printf("  Motor %d: Pos=%.3f, Vel=%.3f, Torque=%.3f\n", i, pos, vel, torque);
        }
        
        // Strict timing for 10000Hz loop
        loop_start += std::chrono::microseconds(1000);  // 1000Hz = 1000µs
        std::this_thread::sleep_until(loop_start);
    }
}

// Alternative: run control loop for a specified number of iterations
void AK60Controller::runControlLoop(int iterations) {
    for (int i = 0; i < iterations; i++) {
        updateMotorStates();
        sendMotorCommands();
        
        // Print all motors' states
        printf("Motors State:\n");
        for(int j = 0; j < 3; j++) {
            float pos, vel, torque;
            getMotorState(j, pos, vel, torque);
            printf("  Motor %d: Pos=%.3f, Vel=%.3f, Torque=%.3f\n", j, pos, vel, torque);
        }
        
        // Strict timing for 10000Hz loop
        loop_start += std::chrono::microseconds(1000);  // 1000Hz = 1000µs
        std::this_thread::sleep_until(loop_start);
    }
} 


