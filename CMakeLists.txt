cmake_minimum_required(VERSION 3.20)
if(POLICY CMP0148)
    cmake_policy(SET CMP0148 OLD)
endif()
project(lerobot_doosan_leaderam LANGUAGES C CXX)

# Standard C++ settings
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 라이브러리
add_subdirectory(doosan_robot_controller)
add_subdirectory(ik_solver)


## 타겟
#add_executable(test_drfl)
#
## 헤더
#target_include_directories(test_drfl
#        PRIVATE
#        ${CMAKE_CURRENT_SOURCE_DIR}/include
#)
#
## 링크
#target_link_libraries(test_drfl
#        PRIVATE
#        drfl
#)