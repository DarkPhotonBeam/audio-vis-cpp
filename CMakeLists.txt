cmake_minimum_required(VERSION 3.10)

project(AudioVis)

# Set default build type if not specified
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()

# Compiler flags for different configurations
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")


find_package(SDL3 CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(ALSA REQUIRED)
find_package(Matplot++ CONFIG REQUIRED)

add_executable(AudioVis main.cpp
        wave_io.cpp
        wave_io.h)

target_link_libraries(AudioVis PRIVATE SDL3::SDL3 Eigen3::Eigen ALSA::ALSA Matplot++::cimg Matplot++::matplot)
