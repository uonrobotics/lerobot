add_library(drfl INTERFACE)

target_include_directories(drfl INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

# External libraries that DRFL depends on
set(DRFL_LIB_DIR ${CMAKE_CURRENT_SOURCE_DIR}/lib)

# libDRFL.a is a static library
# libPocoFoundation.so.80 and libPocoNet.so.80 are shared libraries

target_link_libraries(drfl INTERFACE
    ${DRFL_LIB_DIR}/libDRFL.a
    ${DRFL_LIB_DIR}/libPocoFoundation.so.80
    ${DRFL_LIB_DIR}/libPocoNet.so.80
    pthread
)
