cmake_minimum_required(VERSION 3.25)
project(iksolver)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}")

# 라이브러리
#add_subdirectory(3rdparty/imgui)        # imgui
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


# 라이브러리로 생성
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
#        imgui
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

# 매크로 정의
target_compile_definitions(iksolver
        PUBLIC
        ROOT_PATH="${PROJECT_ROOT}"
)

# 예제 빌드
if (PROJECT_IS_TOP_LEVEL)
    add_executable(example_main
            example/example_main.cpp
            example/draw.cpp
    )

    target_link_libraries(example_main
            iksolver
            imgui
    )
endif()

# 파이썬 바인딩
add_subdirectory(python)

