# Optional C++ extension for dhb_xr (nanobind)
cmake_minimum_required(VERSION 3.15)
project(dhb_xr_cpp LANGUAGES CXX)

find_package(Python 3.10 COMPONENTS Interpreter Development.Module REQUIRED)
find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(_dhb_xr_cpp dhb_cpp.cpp
    INSTALL_RPATH "$ORIGIN"
    INSTALL_RPATH_USE_LINK_PATH TRUE
)
target_compile_features(_dhb_xr_cpp PRIVATE cxx_std_17)

# Output stays in build tree; copy to src/dhb_xr/ via pixi task build-cpp
