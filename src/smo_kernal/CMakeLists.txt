cmake_minimum_required(VERSION 3.1)
project(CSMO VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# SET(CMAKE_BUILD_TYPE "Debug")

set(PYTHON_ROOT /data/anaconda3/envs/jittor/)
set(PYTHON_VERSION 3.7)


set(PYBIND11_PYTHON_VERSION ${PYTHON_VERSION})
set(PYTHON_EXECUTABLE ${PYTHON_ROOT}/bin/python)
set(pybind11_DIR ${PYTHON_ROOT}/lib/python${PYTHON_VERSION}/site-packages/pybind11/share/cmake/pybind11)
find_package(pybind11 CONFIG REQUIRED)
find_package(Eigen3 REQUIRED)

pybind11_add_module(ccsmo interface.cc smo_solver.cc)
target_include_directories(ccsmo PRIVATE ${EIGEN3_INCLUDE_DIR})