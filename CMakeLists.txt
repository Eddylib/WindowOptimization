cmake_minimum_required(VERSION 3.12)
project(WindowOptimization)

#一些编译选项
    #如果采用NRV优化，很多情况下需要调用移动构造函数的代码都会被优化掉，拷贝构造函数暂时没有这个问题
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O0 -fno-elide-constructors")
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -O0")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
#set(CMAKE_VERBOSE_MAKEFILE on)

#eigen 必须
set(EIGEN_DIR "/home/libaoyu/Desktop/deplib/eigen3")

#使用ROS的rviz进行绘图？
set(ROS_DROW FALSE)
#前端，未实现，如果此选项打开需要设置OpenCV_DIR变量来告诉cmake opencv的位置，编译采用opencv3.2
set(BUILD_FROUNT FALSE)

if(${ROS_DROW})
    message(STATUS "build with ros rviz draw")
    add_definitions(-D ROS_DRAW)
    set(ROS_WS /home/libaoyu/Data/catkin_ws)
    set(LIB_PUBLISHER ${ROS_WS}/devel/lib/libba_publisher.so)
    set(INC_PUBLISHER ${ROS_WS}/src/showpath/include/showpath)

    if(EXISTS ${LIB_PUBLISHER})
    else()
        message(FATAL_ERROR \${LIB_PUBLISHER}:${LIB_PUBLISHER} \ donot \ exists)
    endif()

    if(EXISTS ${INC_PUBLISHER})
    else()
        message(FATAL_ERROR \${INC_PUBLISHER}:${INC_PUBLISHER} \ donot \ exists)
    endif()
else()
    set(LIB_PUBLISHER "")
    set(INC_PUBLISHER "")
endif()

if(EXISTS ${EIGEN_DIR})
else()
    message(FATAL_ERROR \${EIGEN_DIR}:${EIGEN_DIR} \ donot \ exists)
endif()


add_subdirectory(third_party/yaml-cpp)
include_directories(third_party/Sophus)
include_directories(${YAML_CPP_SOURCE_DIR}/src)
include_directories(${YAML_CPP_SOURCE_DIR}/include)

include_directories(${EIGEN_DIR})

if(${BUILD_FROUNT})
set(OpenCV_DIR "/home/libaoyu/Desktop/dist_root/usr/local/share/OpenCV")
add_definitions(-D BUILD_FROUNT)
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})
message(STATUS OpenCV_VERSION:${OpenCV_VERSION})
message(STATUS OpenCV_INCLUDE_DIRS:${OpenCV_INCLUDE_DIRS})
message(STATUS OpenCV_LIBS:${OpenCV_LIBS})
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
else()
endif()

add_library(window_optimization
        src/alg_config.h
        src/debug_utils.cpp src/debug_utils.h
        src/backend/window_optimization.cpp src/backend/window_optimization.h
        src/backend/Camera.cpp src/backend/Camera.h
        src/backend/Point.cpp src/backend/Point.h
        src/backend/Residual.cpp src/backend/Residual.h)

add_library(utils utils/utils.h utils/utils.cpp)
target_include_directories(utils PUBLIC ${INC_PUBLISHER})

add_executable(main_ba srcBA/main_ba.cpp)
target_include_directories(main_ba PUBLIC ${INC_PUBLISHER})
target_link_libraries(main_ba ${LIB_PUBLISHER})
target_link_libraries(main_ba utils)

#add_library(full_system src/FullSystem.cpp src/FullSystem.h)
#target_link_libraries(full_system front_end window_optimization yaml-cpp)
#add_executable(main main.cpp OptimizerDebugC.cpp OptimizerDebugC.h)
#target_link_libraries(main window_optimization)
#
#add_executable(test test.cpp utils.h)
#
#add_executable(test_img_loader test_img_loader.cpp src/frontend/DispMatcher.h src/frontend/DispMatcher.cpp)
#target_link_libraries(test_img_loader full_system)
