cmake_minimum_required(VERSION 3.16)
project(llm_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Options
option(LLM_CUDA "Enable CUDA GPU support" OFF)
option(LLM_OPENMP "Enable OpenMP parallelization" ON)

# Source files
set(LLM_SOURCES src/main.cpp)

# Compiler flags
if(CMAKE_CXX_COMPILER_ID MATCHES "Clang|GNU")
    add_compile_options(-O2 -Wall -Wextra -Wpedantic)
endif()

# OpenMP
if(LLM_OPENMP)
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        message(STATUS "OpenMP found: ${OpenMP_CXX_VERSION}")
    else()
        message(STATUS "OpenMP not found, disabling")
        set(LLM_OPENMP OFF)
    endif()
endif()

# CUDA
if(LLM_CUDA)
    enable_language(CUDA)
    find_package(CUDAToolkit REQUIRED)
    add_compile_definitions(LLM_USE_CUDA)
    list(APPEND LLM_SOURCES src/cuda_kernels.cu)
    message(STATUS "CUDA enabled: ${CUDAToolkit_VERSION}")
endif()

# Main executable
add_executable(llm ${LLM_SOURCES})
target_include_directories(llm PRIVATE src)

if(LLM_OPENMP AND OpenMP_CXX_FOUND)
    target_link_libraries(llm PRIVATE OpenMP::OpenMP_CXX)
    target_compile_definitions(llm PRIVATE LLM_USE_OPENMP)
endif()

if(LLM_CUDA)
    target_link_libraries(llm PRIVATE CUDA::cudart CUDA::cublas)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        set_target_properties(llm PROPERTIES CUDA_ARCHITECTURES "70;75;80;86;89;90")
    endif()
endif()

# Math library on Linux
if(UNIX AND NOT APPLE)
    target_link_libraries(llm PRIVATE m)
endif()

# Tests
enable_testing()
add_executable(test_gguf tests/test_gguf.cpp)
target_include_directories(test_gguf PRIVATE src)
if(UNIX AND NOT APPLE)
    target_link_libraries(test_gguf PRIVATE m)
endif()
add_test(NAME test_gguf COMMAND test_gguf)
