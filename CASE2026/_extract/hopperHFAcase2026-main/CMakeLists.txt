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

add_subdirectory(xbox_driver)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/include/common
    ${CMAKE_CURRENT_SOURCE_DIR}/include/common/Math
    ${CMAKE_CURRENT_SOURCE_DIR}/include/hardware
    ${CMAKE_CURRENT_SOURCE_DIR}/hopper_lcm_types/lcm_types/cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/xbox_driver/include
)

# Legs-only driver (Jetson + Pixhawk architecture):
#   - Leg motors (3x AK60) over SocketCAN (can0)
#   - Gamepad + LCM state I/O
#   - IMU + propeller PWM are NOT handled here anymore; the Pixhawk owns them
#     (px4_bridge consumes hopper_imu_lcmt / motor_pwm_lcmt over LCM/DDS).
add_executable(hopper_driver
    main.cpp
    src/common/PeriodicTask.cpp
    src/common/Utilities_print.cpp
    src/common/utilities.cpp
    src/hardware/utility.cpp
    src/hardware/hopper_hardware.cpp
    src/hardware/ak60_controller.cpp
    src/hardware/kalman_filter.cpp
    src/hardware/linmot_util.cpp
)

target_link_libraries(hopper_driver 
    Threads::Threads
    rt
    lcm
    xbox_controller
    pthread
    udev
)

# ----------------------------------------------------------------------------
# Legacy Velox-F7 flight-controller tools (tri_balance, motor_id_seq) are
# retired in the Pixhawk migration: they talk MSP to the Velox over serial,
# which no longer exists in this hardware stack. The sources
# (main_tri_balance.cpp, main_motor_id_seq.cpp, src/hardware/VeloxF7Wrapper.cpp,
# src/hardware/tri_balance_controller.cpp) are kept in the tree for reference
# but are no longer built. Re-enable below if you still have a Velox F7.
# ----------------------------------------------------------------------------
# add_executable(tri_balance
#     main_tri_balance.cpp
#     src/hardware/VeloxF7Wrapper.cpp
#     src/hardware/tri_balance_controller.cpp
# )
# target_link_libraries(tri_balance Threads::Threads rt xbox_controller pthread udev)
#
# add_executable(motor_id_seq
#     main_motor_id_seq.cpp
#     src/hardware/VeloxF7Wrapper.cpp
# )
# target_link_libraries(motor_id_seq Threads::Threads rt pthread)

message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")
