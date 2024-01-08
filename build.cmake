cmake_minimum_required(VERSION 3.10)
project(GDTW)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -Wall -std=c++11")

include_directories(/Users/dderiso/anaconda3/include)
include_directories(/Users/dderiso/anaconda3/lib/python3.11/site-packages/numpy/core/include)
include_directories(/Users/dderiso/anaconda3/include/python3.11)

add_library(GDTW SHARED gdtw/gdtw_solver.cpp)

set_target_properties(GDTW PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/build/temp.macosx-11.1-arm64-cpython-311
    PREFIX ""
    SUFFIX ".o"
)
