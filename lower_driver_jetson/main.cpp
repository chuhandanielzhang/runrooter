#include "hopper_hardware.hpp"
#include <signal.h>
namespace HOPPER_mode{
  constexpr int OFF = 0;
  constexpr int DAMP = 1;
  constexpr int PD = 2;
  constexpr int PWMPD = 3;  // PD (leg) + PWM (propellers) -- legacy combined mode (no key bound by default)
  constexpr int SET_ZERO_MODE = 4;
  constexpr int PWM_ONLY = 5;  // PWM (propellers) only; leg motors free (zero torque, kp=kd=0)
}
// Global pointer for signal handler
HopperHardware* hopper_ptr = nullptr;
int mode = HOPPER_mode::OFF;

void deal_with_mode_change(XboxController::XboxMap xbox_map){
    static bool last_b = false;
    static bool last_x = false;
    static bool last_a = false;
    
    if(xbox_map.b && !last_b && mode != HOPPER_mode::DAMP){
        mode = HOPPER_mode::DAMP;
        std::cout << "change to DAMP (legs damping, props OFF)" << std::endl;
    }else if(xbox_map.x && !last_x && mode != HOPPER_mode::PD){
        mode = HOPPER_mode::PD;
        std::cout << "change to PD (legs only, props OFF)" << std::endl;
    }else if(xbox_map.a && !last_a){
        // A controls propellers without changing the leg state:
        //   PD    -> PWMPD  (add props, keep legs active)
        //   PWMPD -> PD     (remove props, keep legs active)
        //   OFF/DAMP/PWM_ONLY: ignore A so it never turns leg damping/PD off.
        if(mode == HOPPER_mode::PD){
            mode = HOPPER_mode::PWMPD;
            std::cout << "change to PWMPD (legs stay PD + props PWM)" << std::endl;
        }else if(mode == HOPPER_mode::PWMPD){
            mode = HOPPER_mode::PD;
            std::cout << "change to PD (props OFF, legs stay PD)" << std::endl;
        }else{
            std::cout << "A ignored: press X first to arm legs, then A toggles props" << std::endl;
        }
    }else if(xbox_map.thumbr && xbox_map.thumbl && xbox_map.lb && xbox_map.rb && mode == HOPPER_mode::OFF){
        mode = HOPPER_mode::SET_ZERO_MODE;
        std::cout << "change to SET_ZERO_MODE" << std::endl;
    }else if(xbox_map.start && mode == HOPPER_mode::DAMP){
        mode = HOPPER_mode::OFF;
        std::cout << "change to OFF" << std::endl;
    }
    
    last_b = xbox_map.b;
    last_x = xbox_map.x;
    last_a = xbox_map.a;
}
volatile bool g_running = true;

// Signal handler for Ctrl+C
void signalHandler(int signum) {
    std::cout << "\nShutting down..." << std::endl;
    g_running = false;
}

int main(int argc, char** argv) {
    // Register signal handler
    signal(SIGINT, signalHandler);
    // Create hardware on heap and store pointer
    hopper_ptr = new HopperHardware(true);
    while (g_running) {
        deal_with_mode_change(hopper_ptr->get_xbox_map());
        // Remote safety override (ONE-SHOT):
        // Controller PC can request DAMP by sending motor_pwm_lcmt.control_mode < 0.
        // User request: only honor this when we are actively controlling (PD/PWMPD).
        // We always consume (clear) the request so it doesn't latch and surprise-trigger later.
        if (hopper_ptr->get_motor_pwm_control_mode() < 0) {
            if (mode == HOPPER_mode::PD || mode == HOPPER_mode::PWMPD || mode == HOPPER_mode::PWM_ONLY) {
                std::cout << "force DAMP (remote SAFE flag)" << std::endl;
                mode = HOPPER_mode::DAMP;
            }
            hopper_ptr->clear_motor_pwm_control_mode();
        }
        if(mode == HOPPER_mode::OFF){
            hopper_ptr->step_with_only_receiving();
        }else if(mode == HOPPER_mode::DAMP){
            hopper_ptr->step_with_damping();
        } else if(mode == HOPPER_mode::PD){
            hopper_ptr->step_with_pd_control();
        } else if(mode == HOPPER_mode::PWMPD){
            hopper_ptr->step_with_pd_pwm_control();
        } else if(mode == HOPPER_mode::PWM_ONLY){
            hopper_ptr->step_with_pwm_only();
        } else if(mode == HOPPER_mode::SET_ZERO_MODE){
            hopper_ptr->step_with_set_zero_mode();
            mode = HOPPER_mode::OFF;
        }
        std::this_thread::sleep_for(std::chrono::microseconds(2000));  // 500Hz
        if(hopper_ptr->step_counter%10000==0){
            std::cout<<"current mode: ";
            switch(mode) {
                case HOPPER_mode::OFF:
                    std::cout<<"OFF";
                    break;
                case HOPPER_mode::DAMP:
                    std::cout<<"DAMP";
                    break;
                case HOPPER_mode::PD:
                    std::cout<<"PD";
                    break;
                case HOPPER_mode::PWMPD:
                    std::cout<<"PWMPD";
                    break;
                case HOPPER_mode::PWM_ONLY:
                    std::cout<<"PWM_ONLY";
                    break;
                case HOPPER_mode::SET_ZERO_MODE:
                    std::cout<<"SET_ZERO_MODE";
                    break;
                default:
                    std::cout<<"UNKNOWN";
            }
            std::cout<<std::endl;
        }
    }

    // Clean up if loop exits normally
    delete hopper_ptr;
    return 0;
}