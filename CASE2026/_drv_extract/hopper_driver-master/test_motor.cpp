#include "ak60_controller.h"
#include <signal.h>

// Global pointer to controller for signal handler
AK60Controller* controller_ptr = nullptr;

// Signal handler for Ctrl+C
void signalHandler(int signum) {
    if (controller_ptr) {
        delete controller_ptr;
        controller_ptr = nullptr;
    }
    exit(signum);
}

int main() {
    // Register signal handler
    signal(SIGINT, signalHandler);
    
    // Create controller on heap and store pointer
    controller_ptr = new AK60Controller();
    
    int motor_id = 0;
    float kp = 0.0f;
    float kd = 0.0f;
    float tau = 0.5f;
    float pos = 0.0f;
    float vel = 0.0f;
    
    // Set different target positions for each motor (optional)
    controller_ptr->setMotorParams(motor_id, pos, vel, tau, kp, kd);   // Motor 0 on CAN0
    // controller_ptr->setMotorParams(1, 0.3f, 0.0f, 5.0f, 0.5f);   // Motor 1 on CAN0
    // controller_ptr->setMotorParams(2, 0.7f, 0.0f, 5.0f, 0.5f);   // Motor 2 on CAN0
    // controller_ptr->setMotorParams(3, 0.2f, 0.0f, 5.0f, 0.5f);   // Motor 0 on CAN1
    // controller_ptr->setMotorParams(4, 0.4f, 0.0f, 5.0f, 0.5f);   // Motor 1 on CAN1
    // controller_ptr->setMotorParams(5, 0.6f, 0.0f, 5.0f, 0.5f);   // Motor 2 on CAN1
    
    // Run the control loop
    controller_ptr->runControlLoopMIT();
    
    // Clean up (this will only execute if runControlLoop() exits normally)
    delete controller_ptr;
    return 0;
}