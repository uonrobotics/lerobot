## ------------------------------------------------------------------
## Python Binding
## ------------------------------------------------------------------
set(USER_HOME_DIR $ENV{HOME}) # HOME USER NAME
set(CONDA_ENV_NAME "dsr")  # Conda: dsr

#  Conda 환경의 Python 경로를 우선적으로 찾도록 설정
set(PYTHON_EXECUTABLE "${USER_HOME_DIR}/miniconda3/envs/${CONDA_ENV_NAME}/bin/python")
set(Python_EXECUTABLE ${PYTHON_EXECUTABLE})
set(PYBIND11_PYTHON_EXECUTABLE ${PYTHON_EXECUTABLE})

# Python 3.12 지정
find_package(Python 3.12 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 REQUIRED)

pybind11_add_module(doosan_controller_py
        src/doosan_controller.cpp
        src/doosan_controller_binding.cpp
)

# 헤더
target_include_directories(doosan_controller_py
    PRIVATE
    ${PROJECT_SOURCE_DIR}/include
    ${PROJECT_SOURCE_DIR}/drfl/BINARY_GL013201_20250213/include
)

# 링크
target_link_libraries(doosan_controller_py
    PRIVATE
    drfl
)

# 출력 디렉토리 설정
set(MY_LIB_DEST ${CMAKE_CURRENT_LIST_DIR}/lib)

set_target_properties(doosan_controller_py PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${MY_LIB_DEST}
)


# drfl .so 파일 복사
set(DRFL_LIB_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../drfl/lib")        # 두산 lib가 있는 디렉토리
set(POCO_FOUNDATION_LIB "${DRFL_LIB_DIR}/libPocoFoundation.so.80") # 두산 .so
set(POCO_NET_LIB "${DRFL_LIB_DIR}/libPocoNet.so.80")               # 두산 .so
set(TARGET_DEST "${CMAKE_CURRENT_SOURCE_DIR}/lib")                 # 복사할 디렉토리

add_custom_command(TARGET doosan_controller_py POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E make_directory "${TARGET_DEST}"
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        "${POCO_FOUNDATION_LIB}"
        "${POCO_NET_LIB}"
        "${TARGET_DEST}"
        COMMENT "Copying Poco libraries to wrapper lib directory"
)