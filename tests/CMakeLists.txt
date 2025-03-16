set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Find the Google Test package
find_package(GTest REQUIRED)
include_directories(${GTEST_INCLUDE_DIRS})

# Add the test executable
add_executable(test_memtegra test_memtegra.cpp)

# Link the test executable with the MemTegra library and Google Test
target_link_libraries(test_memtegra MemTegra ${GTEST_LIBRARIES} pthread)

# Enable testing
enable_testing()

# Add a test
add_test(NAME MemTegraTest COMMAND test_memtegra)