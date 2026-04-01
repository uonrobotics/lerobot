# -----------------------------------------------------------
# 두산 라이브러리
# -----------------------------------------------------------
add_library(drfl INTERFACE)

target_include_directories(drfl
        INTERFACE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(drfl
        INTERFACE
        ${CMAKE_CURRENT_LIST_DIR}/lib/libDRFL.a
        ${CMAKE_CURRENT_LIST_DIR}/lib/libPocoFoundation.so.80
        ${CMAKE_CURRENT_LIST_DIR}/lib/libPocoNet.so.80
        pthread
)

# -----------------------------------------------------------
# 두산 컨트롤러 (DooSan Robot Controller)
# -----------------------------------------------------------

# 라이브러리 생성
add_library(doosan_robot_controller
        ${CMAKE_CURRENT_LIST_DIR}/src/doosan_robot_controller.cpp
)

# 헤더
target_include_directories(doosan_robot_controller
        PUBLIC
        ${CMAKE_CURRENT_LIST_DIR}/include
)

# 링크
target_link_libraries(doosan_robot_controller
        PUBLIC
        drfl
)


# -----------------------------------------------------------
# 파이썬 바인딩
# -----------------------------------------------------------
#set(PYTHON_EXECUTABLE /home/jusik/miniconda3/envs/dsr/bin/python) # <- 사용하고자 하는 파이썬 버전으로 수정
set(PYTHON_EXECUTABLE "/usr/bin/python3" CACHE FILEPATH "파이썬 실행 파일 경로") # <- 시스템 또는 프로젝트에 사용할 파이썬으로 설정하세요.


# 파이썬 유효성 체크
if(NOT EXISTS ${PYTHON_EXECUTABLE})
    message(AUTHOR_WARNING
            "\n[Warning] 지정된 파이썬 경로를 찾을 수 없습니다: ${PYTHON_EXECUTABLE}\n"
            "conda 환경이 활성화되어 있는지, 또는 경로가 맞는지 확인해 주세요.\n\n"
            "Doosan_robot_controller_py를 생성할 수 없습니다."
    )
else()
    message(STATUS
            "\n[Info ] doosan_robot_controller_py 생성: ${PYTHON_EXECUTABLE}\n"
    )

    # pybind 설치
    include(FetchContent)

    FetchContent_Declare(
            pybind11
            GIT_REPOSITORY https://github.com/pybind/pybind11.git
            GIT_TAG        v2.11.1 # 최신 안정화 버전 사용
    )

    FetchContent_MakeAvailable(pybind11)


    # 파이썬 바인딩 설정
    pybind11_add_module(doosan_robot_controller_py             # 빌드 이름
            ${CMAKE_CURRENT_LIST_DIR}/src/pybind_module.cpp    # 소스
            ${CMAKE_CURRENT_LIST_DIR}/src/doosan_robot_controller.cpp
    )

    # .so 파일 생성 위치 (/lib)
    set_target_properties(doosan_robot_controller_py
            PROPERTIES
            LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/lib
    )

    # 링크
    target_link_libraries(doosan_robot_controller_py
            PRIVATE
            drfl
    )
endif()

