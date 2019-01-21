cmake_minimum_required(VERSION 3.12)
project(WindowOptimization)

set(CMAKE_CXX_STANDARD 14)
#如果采用NRV优化，很多情况下需要调用移动构造函数的代码都会被优化掉，拷贝构造函数暂时没有这个问题
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O0 -fno-elide-constructors")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O0")
SET( CMAKE_VERBOSE_MAKEFILE on )
#set(Eigen3_DIR "/home/libaoyu/Desktop/dist_root/usr/local/share/eigen3/cmake")
#find_package(Eigen3 REQUIRED)
#include_directories(${EIGEN3_INCLUDE_DIR})
#message(STATUS EIGEN3_INCLUDE_DIR:${EIGEN3_INCLUDE_DIR})

add_subdirectory(third_party/yaml-cpp)
include_directories(${YAML_CPP_SOURCE_DIR}/src)
include_directories(${YAML_CPP_SOURCE_DIR}/include)

include_directories("/home/libaoyu/Desktop/deplib/eigen3")
include_directories("/home/libaoyu/Desktop/deplib/")

set(OpenCV_DIR "/home/libaoyu/Desktop/dist_root/usr/local/share/OpenCV")
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
message(STATUS OpenCV_VERSION:${OpenCV_VERSION})
message(STATUS OpenCV_INCLUDE_DIRS:${OpenCV_INCLUDE_DIRS})
message(STATUS OpenCV_LIBS:${OpenCV_LIBS})

add_library(window_optimization
        src/alg_config.h
        src/debug_utils.cpp src/debug_utils.h
        src/backend/window_optimization.cpp src/backend/window_optimization.h
        src/backend/Camera.cpp src/backend/Camera.h
        src/backend/Point.cpp src/backend/Point.h
        src/backend/Residual.cpp src/backend/Residual.h)

add_library(front_end
        src/io/ImageLoader.cpp
        src/io/ImageLoader.h
        src/frontend/DispMatcher.cpp
        src/frontend/DispMatcher.h
        src/alg_utils.cpp
        src/alg_utils.h
        src/frontend/PixelSeletor.cpp
        src/frontend/PixelSeletor.h src/frontend/Frame.cpp src/frontend/Frame.h src/FullSystem.cpp src/FullSystem.h)
target_link_libraries(front_end ${OpenCV_LIBS})

add_library(full_system src/FullSystem.cpp src/FullSystem.h)
target_link_libraries(full_system front_end window_optimization yaml-cpp)

add_executable(main main.cpp OptimizerDebugC.cpp OptimizerDebugC.h)
target_link_libraries(main window_optimization)

add_executable(test test.cpp utils.h)

add_executable(test_img_loader test_img_loader.cpp src/frontend/DispMatcher.h src/frontend/DispMatcher.cpp)
target_link_libraries(test_img_loader full_system)