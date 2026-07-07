cmake_minimum_required(VERSION 3.10)

# Project name and version
project(hopper_driver VERSION 1.0.0)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -w -O2")  # -w disables all warnings
add_subdirectory(Lpms_IMU)
add_subdirectory(xbox_driver)
# Add include directories
include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${PROJECT_SOURCE_DIR}/include/common)
include_directories(${PROJECT_SOURCE_DIR}/include/common/Math)
include_directories(${PROJECT_SOURCE_DIR}/include/hardware)
include_directories(${PROJECT_SOURCE_DIR}/hopper_lcm_types/lcm_types/cpp)
include_directories(${PROJECT_SOURCE_DIR}/Lpms_IMU/include)
include_directories(${PROJECT_SOURCE_DIR}/xbox_driver/include)
# Find required packages
find_package(Threads REQUIRED)

# Collect source files
file(GLOB_RECURSE SOURCES 
    "src/*.cpp"
    "src/common/*.cpp"
    "src/hardware/*.cpp"
)

# Create executables
add_executable(test_motor "test_motor.cpp" ${SOURCES})
add_executable(hopper_driver "main.cpp" ${SOURCES})
# Link libraries
target_link_libraries(test_motor 
    Threads::Threads
    rt  # For real-time features
    lcm
    Lpms_IMU
    xbox_controller
)
target_link_libraries(hopper_driver 
    Threads::Threads
    rt  # For real-time features
    lcm
    Lpms_IMU
    xbox_controller
)
# Set compiler-specific flags
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    target_compile_options(test_motor PRIVATE -std=c++14)
endif()

# Optional: Enable debug symbols in Debug mode
set(CMAKE_BUILD_TYPE Release)
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    target_compile_options(${PROJECT_NAME} PRIVATE -g)
endif()

# Print configuration info
message(STATUS "Project: ${PROJECT_NAME}")
message(STATUS "C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "Compiler: ${CMAKE_CXX_COMPILER_ID}")

# Install target (optional)
install(TARGETS test_motor DESTINATION bin)