cmake_minimum_required(VERSION 3.14)
project(
  MemTegra
  VERSION 0.1
  LANGUAGES CXX
)

if(PROJECT_SOURCE_DIR STREQUAL PROJECT_BINARY_DIR)
  message(
    FATAL_ERROR
      "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there."
  )
endif()

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Check for CUDA support
find_program(CUDA_COMPILER nvcc)
if(CUDA_COMPILER)
  set(ENABLE_CUDA_DEFAULT 1)
else()
  set(ENABLE_CUDA_DEFAULT 0)
endif()

option(ENABLE_CUDA "Enable CUDA support" ${ENABLE_CUDA_DEFAULT})

if(ENABLE_CUDA)
  set(DEFAULT_LIBRARY_TYPE SHARED) # Use dynamic library when CUDA is enabled
else()
  set(DEFAULT_LIBRARY_TYPE STATIC) # Use static library otherwise
endif()

# Add the source files for the MemTegra library
add_library(${PROJECT_NAME} ${DEFAULT_LIBRARY_TYPE}
    src/MemTegra.cpp
    src/Utils.cpp
    src/device_memory.cpp
    include/MemTegra/MemTegra.h
    include/MemTegra/strong_pointer.hpp
    include/MemTegra/Utils.h
    include/MemTegra/allocator.hpp
    include/MemTegra/memory_tags.h
    include/MemTegra/device_memory.h
)

target_include_directories(
  ${PROJECT_NAME} PUBLIC $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
)

# being a cross-platform target, we enforce standards conformance on MSVC
target_compile_options(${PROJECT_NAME} PUBLIC "$<$<COMPILE_LANG_AND_ID:CXX,MSVC>:/permissive->")

# Add CUDA-specific sources and properties if enabled
if(ENABLE_CUDA)
  enable_language(CUDA)

  # Set CUDA architectures
  if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75) # Example: Set to 7.5 for modern GPUs
  endif()

  set_property(TARGET ${PROJECT_NAME} PROPERTY CUDA_ARCHITECTURES ${CMAKE_CUDA_ARCHITECTURES})

  target_sources(${PROJECT_NAME} PRIVATE
	  #src/device_memory.cu
  )
  set(CU_SOURCE_FILES src/device_memory.cpp) 
  set_source_files_properties(${CU_SOURCE_FILES} PROPERTIES LANGUAGE CUDA)

  target_compile_definitions(${PROJECT_NAME} PUBLIC ENABLE_CUDA=1)
  message(STATUS "CUDA support enabled with architectures: ${CMAKE_CUDA_ARCHITECTURES}")
else()
  message(STATUS "CUDA support disabled.")
  get_target_property(LIB_SOURCES ${PROJECT_NAME} SOURCES)
  set(CU_SOURCE_FILES ${LIB_SOURCES}) 
  list(FILTER CU_SOURCE_FILES INCLUDE REGEX "\\.cu$")
  if(CU_SOURCE_FILES)
	  set_source_files_properties(${CU_SOURCE_FILES} PROPERTIES LANGUAGE CXX)
	  # TODO: need to fix the c++ compiler error on cu files
  endif()
endif()
