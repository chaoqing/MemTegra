cmake_minimum_required(VERSION 3.14)
project(MemTegra LANGUAGES CXX)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add include directories
include_directories(${CMAKE_SOURCE_DIR}/include)

# Add the source files for the MemTegra library
add_library(MemTegra
    src/MemTegra.cpp
    src/Utils.cpp
)