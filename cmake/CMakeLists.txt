cmake_minimum_required(VERSION 3.14...3.22)

project(BuildAll LANGUAGES CXX)

include(tools.cmake)

# generate compile_commands.json
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# needed to generate test target
enable_testing()

add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../sample ${CMAKE_BINARY_DIR}/sample)
add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../tests ${CMAKE_BINARY_DIR}/tests)

if(CMAKE_EXPORT_COMPILE_COMMANDS)
  file(WRITE "${PROJECT_BINARY_DIR}/compile_files.list")
  collect_files("MemTegra" "${PROJECT_BINARY_DIR}")
  collect_files("Sample" "${PROJECT_BINARY_DIR}")
  collect_files("MemTegraTest" "${PROJECT_BINARY_DIR}")
endif()
