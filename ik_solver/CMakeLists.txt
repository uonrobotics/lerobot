cmake_minimum_required(VERSION 3.25)
project(iksolver)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")

# 라이브러리
find_package(Eigen3 REQUIRED NO_MODULE) # eigen
find_package(PkgConfig REQUIRED)
pkg_check_modules(URDFDOM urdfdom)      # urdf parser
if(NOT URDFDOM_FOUND)
    message(FATAL_ERROR "\n"
            "-------------------------------------------------------\n"
            " [Error] urdfdom library not found!\n"
            " urdfdom 라이브러리를 찾을 수 없습니다.\n"
            " 아래 명령어를 사용하여 설치해 주세요:\n\n"
            "   sudo apt install liburdfdom-dev\n"
            "-------------------------------------------------------"
    )
endif()

# -----------------------------------------------------------------
# Ik Solver 라이브러리 생성
# -----------------------------------------------------------------
add_library(iksolver
        src/kinematics.cpp
        src/ik_solver.cpp
)

# 헤더
target_include_directories(iksolver
        PUBLIC
        include
        ${EIGEN3_INCLUDE_DIR}
        ${URDFDOM_INCLUDE_DIRS}
)

# 링크
target_link_libraries(iksolver
        Eigen3::Eigen
        urdfdom_model
)

# 컴파일 옵션: eigen 고속 연산
target_compile_options(iksolver
        PUBLIC
        -O3                         # 가장 높은 수준의 일반 최적화, 코드 크기보다는 실행 속도를 우선함
        -march=native               # CPU 아키텍처에 맞춰 최적화
        -ffast-math                 # 부동소수점 연산에서 IEEE 754 표준 엄격성을 무시하고 속도를 높임
        -ffinite-math-only          # 부동소수점 결과값에 NaN(Not a Number)이나 Inf(무한대)가 발생하지 않는다고 가정, 이를 체크하는 로직을 제거하여 속도를 높임
        -funroll-loops              # 루프(for, while)를 풀어서 작성한 것과 같은 효과, 루프 제어에 드는 오버헤드를 줄임
        -DNDEBUG                    # assert() 매크로를 비활성화
)


# -----------------------------------------------------------------
# 파이썬 바인딩
# -----------------------------------------------------------------
set(PYTHON_EXECUTABLE "/usr/bin/python3" CACHE FILEPATH "파이썬 실행 파일 경로") # <- 시스템 또는 프로젝트에 사용할 파이썬으로 설정하세요.

# 파이썬 유효성 체크
if (NOT EXISTS ${PYTHON_EXECUTABLE})
    message(AUTHOR_WARNING
            "\n[Warning] 지정된 파이썬 경로를 찾을 수 없습니다: ${PYTHON_EXECUTABLE}\n"
            "명령어 예시: cmake .. -DPYTHON_EXECUTABLE=/your/path/to/python\n"
            "ik_solver_py를 생성할 수 없습니다."
    )
else()
    message(STATUS
            "\n[Info] iK_solver_py를 생성: ${PYTHON_EXECUTABLE}\n"
    )

    include(FetchContent)

    FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG        v2.11.1 # 최신 안정화 버전 사용
    )

    FetchContent_MakeAvailable(pybind11)


    # 파이썬 바인딩
    pybind11_add_module(
            ik_solver_py        # 빌드 이름
            ${CMAKE_CURRENT_LIST_DIR}/src/pybind_module.cpp   # 소스
    )

    # 링크
    target_link_libraries(ik_solver_py
            PRIVATE
            iksolver
    )

    # .so 파일을 이동
    set_target_properties(ik_solver_py
            PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}/python"
    )
endif ()


# -----------------------------------------------------------------
# 예제 빌드
# -----------------------------------------------------------------
if (PROJECT_IS_TOP_LEVEL)
    # imgui
    add_subdirectory(3rdparty/imgui)

    # 예제 실행 파일
    add_executable(example_main
            example/example_main.cpp
            example/draw.cpp
    )

    # 링크
    target_link_libraries(example_main
            iksolver
            imgui
    )

    # #define ROOT_PATH 선언
    target_compile_definitions(example_main
            PUBLIC
            ROOT_PATH="${PROJECT_ROOT}"
    )
endif()