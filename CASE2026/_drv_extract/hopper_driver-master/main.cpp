#include "hopper_hardware.hpp"
#include <signal.h>
namespace HOPPER_mode{
  constexpr int OFF = 0;
  constexpr int DAMP = 1;
  constexpr int PD = 2;
  constexpr int SET_ZERO_MODE = 3;
}
// Global pointer for signal handler
HopperHardware* hopper_ptr = nullptr;
int mode = HOPPER_mode::OFF;

void deal_with_mode_change(XboxController::XboxMap xbox_map){
    // std::cout << "xbox_map.b: " << xbox_map.b << std::endl;
    // std::cout << "xbox_map.x: " << xbox_map.x << std::endl;
    // std::cout << "xbox_map.thumbr: " << xbox_map.thumbr << std::endl;
    // std::cout << "xbox_map.thumbl: " << xbox_map.thumbl << std::endl;
    // std::cout << "xbox_map.lb: " << xbox_map.lb << std::endl;
    // std::cout << "xbox_map.rb: " << xbox_map.rb << std::endl;
    if(xbox_map.b && mode != HOPPER_mode::DAMP){
        mode = HOPPER_mode::DAMP;
        std::cout << "change to DAMP" << std::endl;
    }else if(xbox_map.x && mode != HOPPER_mode::PD){
        mode = HOPPER_mode::PD;
        std::cout << "change to PD" << std::endl;
    }else if(xbox_map.thumbr && xbox_map.thumbl && xbox_map.lb && xbox_map.rb && mode == HOPPER_mode::OFF){
        mode = HOPPER_mode::SET_ZERO_MODE;
        std::cout << "change to SET_ZERO_MODE" << std::endl;
    }else if(xbox_map.start && mode == HOPPER_mode::DAMP){
        mode = HOPPER_mode::OFF;
        std::cout << "change to OFF" << std::endl;
    }
}
// Signal handler for Ctrl+C
void signalHandler(int signum) {
    if (hopper_ptr) {
        delete hopper_ptr;
        hopper_ptr = nullptr;
    }
    exit(signum);
}

int main(int argc, char** argv) {
    // Register signal handler
    signal(SIGINT, signalHandler);
    // Create hardware on heap and store pointer
    hopper_ptr = new HopperHardware(true);
    while (true) {
        deal_with_mode_change(hopper_ptr->get_xbox_map());
        if(mode == HOPPER_mode::OFF){
            hopper_ptr->step_with_only_receiving();
        }else if(mode == HOPPER_mode::DAMP){
            hopper_ptr->step_with_damping();
        } else if(mode == HOPPER_mode::PD){
            hopper_ptr->step_with_pd_control();
        } else if(mode == HOPPER_mode::SET_ZERO_MODE){
            hopper_ptr->step_with_set_zero_mode();
            mode = HOPPER_mode::OFF;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        if(hopper_ptr->step_counter%1000==0){
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