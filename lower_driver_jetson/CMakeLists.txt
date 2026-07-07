cmake_minimum_required(VERSION 3.10)
project(hopper_driver)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w -O2 -O3")
set(CMAKE_BUILD_TYPE Release)

set(CMAKE_PREFIX_PATH "/usr/local" ${CMAKE_PREFIX_PATH})
include_directories(/usr/local/include)
link_directories(/usr/local/lib)

set(ENV{PKG_CONFIG_PATH} "/usr/local/lib/pkgconfig:$ENV{PKG_CONFIG_PATH}")

find_package(Threads REQUIRED)

add_subdirectory(Lpms_IMU)
add_subdirectory(xbox_driver)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/common
    ${CMAKE_CURRENT_SOURCE_DIR}/include/common/Math
    ${CMAKE_CURRENT_SOURCE_DIR}/include/hardware
    ${CMAKE_CURRENT_SOURCE_DIR}/hopper_lcm_types/lcm_types/cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/Lpms_IMU/include
    ${CMAKE_CURRENT_SOURCE_DIR}/xbox_driver/include
)

# Jetson driver: 3x AK60 legs + Lpms IG1 IMU + LCM @ 500Hz (CASE hopper_driver-master style).
# Propellers remain on Pixhawk (px4_bridge). Do NOT run px4-dds-bridge at the same time
# (it also publishes hopper_imu_lcmt and will fight this driver).
add_executable(hopper_driver
    main.cpp
    src/common/PeriodicTask.cpp
    src/common/Utilities_print.cpp
    src/common/utilities.cpp
    src/hardware/utility.cpp
    src/hardware/hopper_hardware.cpp
    src/hardware/ak60_controller.cpp
    src/hardware/linmot_util.cpp
)

target_link_libraries(hopper_driver 
    Threads::Threads
    rt
    lcm
    Lpms_IMU
    xbox_controller
    pthread
    udev
)

message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")
