/*
 * @Author: linzhu
 * @Date: 2024-04-27 21:00:14
 * @LastEditors: linzhuyue 
 * @LastEditTime: 2025-01-15 14:43:30
 * @FilePath: /lrl_lowlevel_sdk/src/hardware/xbox_controller.cpp
 * @Description: Xbox Controller Class Implementation - Wrapped from rt_rc_interface.cpp
 * 
 * Copyright (c) 2024 by linzhu, All Rights Reserved. 
 */

#include "xbox_controller.hpp"

#include <string.h> // memcpy
#include <stdio.h>
#include <unistd.h>  
#include <sys/types.h>  
#include <sys/stat.h>  
#include <fcntl.h>  
#include <errno.h>  
#include <linux/joystick.h>  
#include <iostream>
#include <cstring>

XboxController::XboxController(const std::string& device_path)
    : fd_(-1)
    , js_gait_(9)
{
    device_path_ = device_path;
    // Initialize mutex
    pthread_mutex_init(&lcm_mutex_, NULL);
    
    // Initialize map to zero
    memset(&map_, 0, sizeof(XboxMap));
    // Try to open the device
    connecting();
}
void XboxController::connecting(){
    if (!openDevice(device_path_)) {
    std::cerr << "Failed to open Xbox controller at: " << device_path_ << std::endl;
    // Keep trying to open the device until successful
    int retry_count = 0;
    while (fd_ < 0) {
        std::cout << "Retrying to open Xbox controller... (Attempt " << retry_count + 1 << ")" << std::endl;
        fd_ = open(device_path_.c_str(), O_RDONLY);
        if (fd_ < 0) {
            retry_count++;
            sleep(1); // Wait 1 second between retries
        }
    }
    if (fd_ < 0) {
        std::cerr << "Failed to open Xbox controller after " << retry_count << " attempts" << std::endl;
    }
    }
    else{
        std::cout << "Xbox controller initialized successfully!" << std::endl;
    }
}
XboxController::~XboxController() {
    close();
    pthread_mutex_destroy(&lcm_mutex_);
}

bool XboxController::initialize() {
    if (fd_ > 0) {
        memset(&map_, 0, sizeof(XboxMap));
        printf("Xbox joystick opened successfully!\n");
        return true;
    } else {
        printf("Xbox joystick open failed!\n");
        return false;
    }
}

bool XboxController::isConnected() const {
    return fd_ > 0;
}

bool XboxController::openDevice(const std::string& device_path) {
    fd_ = open(device_path.c_str(), O_RDONLY);
    if (fd_ < 0) {
        perror("open");
        return false;
    }
    return true;
}

int XboxController::readMap() {
    int len, type, number, value;
    struct js_event js;

    len = read(fd_, &js, sizeof(struct js_event));
    if (len < 0) {
        throw std::runtime_error("Disconnected");
        return -1; // Unreachable but kept for consistency
    }

    type = js.type;
    number = js.number;
    value = js.value;

    map_.time = js.time;

    if (type == JS_EVENT_BUTTON) {
        // printf("Button %d: %d\n", number, value);
        switch (number) {
            case XBOX_BUTTON_A:
                map_.a = value;
                break;
            case XBOX_BUTTON_B:
                map_.b = value;
                break;
            case XBOX_BUTTON_X:
                map_.x = value;
                break;
            case XBOX_BUTTON_Y:
                map_.y = value;
                break;
            case XBOX_BUTTON_LB:
                map_.lb = value;
                break;
            case XBOX_BUTTON_RB:
                map_.rb = value;
                break;
            case XBOX_BUTTON_SELECT:
                map_.select = value;
                break;
            case XBOX_BUTTON_START:
                map_.start = value;
                break;
            case XBOX_BUTTON_ThumbR:
                map_.thumbr = value;
                break;
            case XBOX_BUTTON_ThumbL:
                map_.thumbl = value;
                break;
            case XBOX_BUTTON_HOME:
                map_.home = value;
                break;
            case XBOX_BUTTON_L4:
                map_.l4 = value;
                break;
            case XBOX_BUTTON_R4:
                map_.r4 = value;
                break;
            case XBOX_BUTTON_POINT:
                map_.point = value;
                break;
            default:
                break;
        }
    } else if (type == JS_EVENT_AXIS) {
        switch(number) {
            case XBOX_AXIS_LX:
                map_.lx = value;
                break;
            case XBOX_AXIS_LY:
                map_.ly = value;
                break;
            case XBOX_AXIS_RX:
                map_.rx = value;
                break;
            case XBOX_AXIS_RY:
                map_.ry = value;
                break;
            case XBOX_AXIS_LT:
                map_.lt = value;
                break;
            case XBOX_AXIS_RT:
                map_.rt = value;
                break;
            case XBOX_AXIS_XX:  // D-pad
                map_.xx = value;
                break;
            case XBOX_AXIS_YY:
                map_.yy = value;
                break;
            default:
                break;
        }
    }

    return len;
}


void XboxController::processInput() {
    try {
        int len = readMap();
        
        if (len < 0) {
            memset(&map_, 0, sizeof(XboxMap));
            return;
        }
        
        // Update gamepad data structure
        // updateGamepadData();
        
        // Publish the data


    } catch (const std::exception& e) {
        std::cerr << "Error processing Xbox controller input: " << e.what() << std::endl;
        close();
        fd_ = -1;
        // try reconnecting
        connecting();
        }
}

void XboxController::close() {
    if (fd_ > 0) {
        ::close(fd_);
        fd_ = -1;
    }
}

void* XboxController::v_memcpy(void* dest, volatile void* src, size_t n) {
    void* src_2 = (void*)src;
    return memcpy(dest, src_2, n);
} 