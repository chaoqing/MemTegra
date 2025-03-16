set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add the sample executable
add_executable(sample_app main.cpp)

# Link the sample executable with the MemTegra library
target_link_libraries(sample_app MemTegra)
