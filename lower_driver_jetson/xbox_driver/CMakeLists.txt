cmake_minimum_required(VERSION 3.10)
project(xbox_driver)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include_directories(${PROJECT_SOURCE_DIR}/include)

add_library(xbox_controller
    src/xbox_controller.cpp
)

target_include_directories(xbox_controller PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(xbox_controller
    rt
    pthread
)
