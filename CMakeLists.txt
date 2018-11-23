cmake_minimum_required(VERSION 3.12)
project(WindowOptimization)

set(CMAKE_CXX_STANDARD 14)
#set(Eigen3_DIR "/home/libaoyu/Desktop/dist_root/usr/local/share/eigen3/cmake")
#find_package(Eigen3 REQUIRED)
#include_directories(${EIGEN3_INCLUDE_DIR})
#message(STATUS EIGEN3_INCLUDE_DIR:${EIGEN3_INCLUDE_DIR})
include_directories("/home/libaoyu/Desktop/deplib/eigen3")
include_directories("/home/libaoyu/Desktop/deplib/")

add_library(window_optimization
        src/types.h
        src/debug_utils.cpp src/debug_utils.h
        src/window_optimization.cpp src/window_optimization.h
        src/Camera.cpp src/Camera.h
        src/Point.cpp src/Point.h
        src/Residual.cpp src/Residual.h)


add_executable(main main.cpp)
target_link_libraries(main window_optimization)

add_executable(test test.cpp utils.h)