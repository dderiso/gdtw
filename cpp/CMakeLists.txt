# SPDX-License-Identifier: Apache-2.0
#
# Copyright (C) 2017-2026 
# Dave Deriso <dderiso@alumni.stanford.edu>, Twitter: @davederiso
# Stephen Boyd
#
# GDTW is a Python/C++ library that performs dynamic time warping.
# GDTW improves upon other methods (such as the original DTW, ShapeDTW, and FastDTW) by introducing regularization,
# which obviates the need for pre-processing, and cross-validation for choosing optimal regularization hyper-parameters.
#
# Paper: https://rdcu.be/cT5dD
# Source: https://github.com/dderiso/gdtw
# Docs: https://dderiso.github.io/gdtw

cmake_minimum_required(VERSION 3.20)
project(gdtw_cpp
    VERSION 0.1.0
    DESCRIPTION "General Dynamic Time Warping - all-C++ port mirroring the Python package"
    LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

add_compile_options(-Wall -Wextra -Wno-unused-parameter -Wno-unused-private-field)

add_library(gdtw_third_party INTERFACE)
target_include_directories(gdtw_third_party INTERFACE third_party)

add_library(gdtw_core STATIC
    src/utils.cpp
    src/signal.cpp
    src/solver.cpp
    src/gdtw.cpp
    src/warp.cpp)
target_include_directories(gdtw_core PUBLIC include)
target_link_libraries(gdtw_core PUBLIC gdtw_third_party)
set_target_properties(gdtw_core PROPERTIES POSITION_INDEPENDENT_CODE ON)

enable_testing()
add_executable(gdtw_tests
    tests/test_main.cpp
    tests/test_warp.cpp
    tests/test_loss.cpp
    tests/test_signal.cpp
    tests/test_symmetric.cpp)
target_link_libraries(gdtw_tests PRIVATE gdtw_core)
add_test(NAME gdtw_tests COMMAND gdtw_tests)
