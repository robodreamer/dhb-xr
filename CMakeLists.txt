# Top-level CMake for optional C++ extension (dhb_xr._dhb_xr_cpp)
cmake_minimum_required(VERSION 3.15)
project(dhb_xr_cpp LANGUAGES CXX)

find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)

add_subdirectory(src/dhb_xr/_cpp)
