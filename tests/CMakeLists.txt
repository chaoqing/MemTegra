cmake_minimum_required(VERSION 3.14)

project(MemTegraTest LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(../cmake/tools.cmake)
include(../cmake/CPM.cmake)

CPMAddPackage(NAME MemTegra SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/..)

# Find the Google Test package
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Add the test executable
file(GLOB sources CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/*.cpp)
add_executable(${PROJECT_NAME} ${sources})

# Link the test executable with the MemTegra library and Google Test
target_link_libraries(${PROJECT_NAME} MemTegra ${GTEST_LIBRARIES} pthread)

# Enable testing
enable_testing()

# Add a test
add_test(NAME ${PROJECT_NAME} COMMAND ${PROJECT_NAME} --gtest_output=xml:${CMAKE_CURRENT_BINARY_DIR}/Testing/${PROJECT_NAME}_results.xml)
