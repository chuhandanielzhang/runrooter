cmake_minimum_required(VERSION 3.0.2)
project(Lpms_IMU)

# Find required packages
find_package(Threads REQUIRED)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add include directories
include_directories(
    include
)

# Create library
file(GLOB SRC_FILES
    src/*.cpp
    src/*.c
)

add_library(${PROJECT_NAME}
    ${SRC_FILES}
)

# Link libraries
target_link_libraries(${PROJECT_NAME}
    ${CMAKE_THREAD_LIBS_INIT}
    udev
)

# Create example executable
add_executable(imu_example
    ImuWrapperExample.cpp
)

# Link example with library
target_link_libraries(imu_example
    ${PROJECT_NAME}
)
