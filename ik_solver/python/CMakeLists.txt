# 콘다의 파이썬 3.12를 사용함
set(PYTHON_EXECUTABLE /home/uon/miniconda3/envs/dsr/bin/python)
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
        pybind_module.cpp   # 소스
)

# 링크
target_link_libraries(ik_solver_py
        PRIVATE
        iksolver
)

# .so 파일을 이동
set_target_properties(ik_solver_py PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
)
