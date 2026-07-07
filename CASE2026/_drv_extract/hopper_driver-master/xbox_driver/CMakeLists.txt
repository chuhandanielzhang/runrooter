cmake_minimum_required(VERSION 3.10)
project(xbox_driver)

# Set C++ standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${PROJECT_SOURCE_DIR}/include)

# Add xbox_controller library
add_library(xbox_controller
    src/xbox_controller.cpp
)

# Set include directories for xbox_controller library
target_include_directories(xbox_controller PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# Link libraries for xbox_controller
target_link_libraries(xbox_controller
    rt
    pthread
)

# Add the executable
add_executable(xbox_controller_example
    xbox_controller_example.cpp
)

# Include directories
target_include_directories(xbox_controller_example PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${LCM_INCLUDE_DIRS}
    xbox_controller
)

# Link libraries
target_link_libraries(xbox_controller_example
    rt
    pthread
    xbox_controller
)

# Install targets
install(TARGETS xbox_controller_example
    RUNTIME DESTINATION bin
)
