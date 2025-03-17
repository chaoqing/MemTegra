cmake_minimum_required(VERSION 3.14)

project(Sample LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(../cmake/tools.cmake)
include(../cmake/CPM.cmake)

CPMAddPackage(NAME MemTegra SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/..)

# Add the sample executable
file(GLOB sources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
add_executable(${PROJECT_NAME} ${sources})

set_target_properties(${PROJECT_NAME} PROPERTIES CXX_STANDARD 17)

# Link the sample executable with the MemTegra library
target_link_libraries(${PROJECT_NAME} MemTegra)
