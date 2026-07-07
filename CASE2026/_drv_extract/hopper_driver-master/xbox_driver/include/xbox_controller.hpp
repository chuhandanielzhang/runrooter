/*
 * @Author: linzhu
 * @Date: 2024-04-27 21:00:14
 * @LastEditors: linzhuyue 
 * @LastEditTime: 2025-01-15 14:43:30
 * @FilePath: /lrl_lowlevel_sdk/include/hardware/xbox_controller.hpp
 * @Description: Xbox Controller Class - Wrapped from rt_rc_interface.cpp
 * 
 * Copyright (c) 2024 by linzhu, All Rights Reserved. 
 */

#ifndef _XBOX_CONTROLLER_HPP
#define _XBOX_CONTROLLER_HPP

#include <string>
#include <memory>
#include <pthread.h>

// Forward declaration
struct rc_control_settings;

class XboxController {
public:
    // Xbox button definitions
    static constexpr int XBOX_TYPE_BUTTON = 1;
    static constexpr int XBOX_TYPE_AXIS = 2;

    static constexpr int XBOX_BUTTON_A = 0;
    static constexpr int XBOX_BUTTON_B = 1;
    static constexpr int XBOX_BUTTON_X = 2;
    static constexpr int XBOX_BUTTON_Y = 3;
    static constexpr int XBOX_BUTTON_LB = 4;
    static constexpr int XBOX_BUTTON_RB = 5;
    static constexpr int XBOX_BUTTON_SELECT = 6;
    static constexpr int XBOX_BUTTON_START = 7;
    static constexpr int XBOX_BUTTON_ThumbR = 10;    // +
    static constexpr int XBOX_BUTTON_ThumbL = 9;  // -
    
#ifdef USEARCH_64
    // if in arch64 
    static constexpr int XBOX_BUTTON_HOME = 12;   // M
    static constexpr int XBOX_BUTTON_L4 = 10;     // L4
    static constexpr int XBOX_BUTTON_R4 = 11;     // R4
#else
    // if in x86-64 
    static constexpr int XBOX_BUTTON_HOME = 8;   // M
    static constexpr int XBOX_BUTTON_L4 = 11;     // L4
    static constexpr int XBOX_BUTTON_R4 = 12;     // R4
#endif

    static constexpr int XBOX_BUTTON_POINT = 13;  // Point
    static constexpr int XBOX_BUTTON_ON = 1;
    static constexpr int XBOX_BUTTON_OFF = 0;

    // Xbox axis definitions
    static constexpr int XBOX_AXIS_LX = 0;    // Left stick X axis
    static constexpr int XBOX_AXIS_LY = 1;    // Left stick Y axis
    static constexpr int XBOX_AXIS_RX = 3;    // Right stick X axis
    static constexpr int XBOX_AXIS_RY = 4;    // Right stick Y axis
    static constexpr int XBOX_AXIS_LT = 2;    // Left trigger
    static constexpr int XBOX_AXIS_RT = 5;    // Right trigger
    static constexpr int XBOX_AXIS_XX = 6;    // D-pad X axis
    static constexpr int XBOX_AXIS_YY = 7;    // D-pad Y axis

    // Axis value constants
    static constexpr int XBOX_AXIS_VAL_UP = -32767;
    static constexpr int XBOX_AXIS_VAL_DOWN = 32767;
    static constexpr int XBOX_AXIS_VAL_LEFT = -32767;
    static constexpr int XBOX_AXIS_VAL_RIGHT = 32767;
    static constexpr int XBOX_AXIS_VAL_MIN = -32767;
    static constexpr int XBOX_AXIS_VAL_MAX = 32767;
    static constexpr int XBOX_AXIS_VAL_MID = 0x00;

    // Xbox map structure
    struct XboxMap {
        int time;
        int a, b, x, y;
        int lb, rb;
        int start, select;
        int home, thumbr, thumbl;
        int l4, r4, point;
        int lx, ly, rx, ry;
        int lt, rt;
        int xx, yy;
    };

    // Constructor and destructor
    explicit XboxController(const std::string& device_path = "/dev/input/js0");
    ~XboxController();

    // Disable copy constructor and assignment operator
    XboxController(const XboxController&) = delete;
    XboxController& operator=(const XboxController&) = delete;

    // Public methods
    bool initialize();
    bool isConnected() const;
    void processInput();
    void close();
    void connecting();
    // Getters
    const XboxMap& getMap() const { return map_; }
    int getFileDescriptor() const { return fd_; }
    int getGait() const { return js_gait_; }

private:
    // Private methods
    bool openDevice(const std::string& device_path);
    int readMap();
    std::string device_path_;
    // Member variables
    int fd_;                                    // File descriptor for the joystick device
    XboxMap map_;                              // Current Xbox controller state
    int js_gait_;                             // Gait setting
    pthread_mutex_t lcm_mutex_;               // Mutex for thread safety
    
    // Static helper function
    static void* v_memcpy(void* dest, volatile void* src, size_t n);
};

#endif // _XBOX_CONTROLLER_HPP 