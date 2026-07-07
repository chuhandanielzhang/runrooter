/*
 * @Author: linzhu
 * @Date: 2024-04-27 21:00:14
 * @LastEditors: linzhuyue 
 * @LastEditTime: 2025-01-15 14:43:30
 * @FilePath: /lrl_lowlevel_sdk/examples/xbox_controller_example.cpp
 * @Description: Example usage of Xbox Controller Class
 * 
 * Copyright (c) 2024 by linzhu, All Rights Reserved. 
 */

#include "xbox_controller.hpp"
#include <iostream>
#include <chrono>
#include <thread>

int main() {
    std::cout << "Xbox Controller Example" << std::endl;
    std::cout << "=======================" << std::endl;

    // Create Xbox controller instance
    XboxController xbox("/dev/input/js0");
    
    // Initialize the controller
    if (!xbox.initialize()) {
        std::cerr << "Failed to initialize Xbox controller!" << std::endl;
        return -1;
    }

    std::cout << "Xbox controller initialized successfully!" << std::endl;
    std::cout << "Press Ctrl+C to exit..." << std::endl;

    // Main loop to process input
    while (xbox.isConnected()) {
        // Process input from the controller
        xbox.processInput();
        
        // Get current controller state
        const auto& map = xbox.getMap();
        
        // Print some basic information (you can modify this as needed)
        if (map.a || map.b || map.x || map.y) {
            std::cout << "Buttons - A:" << map.a << " B:" << map.b 
                      << " X:" << map.x << " Y:" << map.y << std::endl;
        }
        // Print other button states if pressed
        if (map.lb || map.rb || map.thumbl || map.thumbr || map.home || map.point || map.select || map.start) {
            std::cout << "Other Buttons - LB:" << map.lb << " RB:" << map.rb 
                      << " L-:" << map.thumbl << " R+:" << map.thumbr
                      << " Home:" << map.home << " Point:" << map.point << " Select:" << map.select << " Start:" << map.start << std::endl;
        }

        // Print D-pad values if pressed
        if (map.xx != 0 || map.yy != 0) {
            std::cout << "D-pad - X:" << map.xx << " Y:" << map.yy << std::endl;
        }

        // Print L4/R4 button states if pressed
        if (map.l4 || map.r4) {
            std::cout << "L4/R4 - L4:" << map.l4 << " R4:" << map.r4 << std::endl;
        }
        // Print stick positions if they're moved significantly
        if (abs(map.lx) > 1000 || abs(map.ly) > 1000) {
            std::cout << "Left Stick - X:" << map.lx << " Y:" << map.ly << std::endl;
        }
        
        if (abs(map.rx) > 1000 || abs(map.ry) > 1000) {
            std::cout << "Right Stick - X:" << map.rx << " Y:" << map.ry << std::endl;
        }
        
        // Print trigger values if they're pressed
        if (map.lt > 1000 || map.rt > 1000) {
            std::cout << "Triggers - LT:" << map.lt << " RT:" << map.rt << std::endl;
        }
    
        // Sleep for a short time to avoid overwhelming the output
        // std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    std::cout << "Xbox controller disconnected." << std::endl;
    return 0;
} 