cmake_minimum_required(VERSION 3.10)

project(AudioTest)

find_package(SDL3 CONFIG REQUIRED)
find_package(Eigen3 CONFIG REQUIRED)
find_package(ALSA REQUIRED)
find_package(Matplot++ CONFIG REQUIRED)

add_executable(AudioTest main.cpp
        wave_io.cpp
        wave_io.h)

target_link_libraries(AudioTest PRIVATE SDL3::SDL3 Eigen3::Eigen ALSA::ALSA Matplot++::cimg Matplot++::matplot)
