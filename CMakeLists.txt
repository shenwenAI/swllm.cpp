cmake_minimum_required(VERSION 3.16)

# Help Windows users who run "cmake .." from a regular terminal (e.g.
# PowerShell) without a Visual Studio environment.  CMake may default to the
# "NMake Makefiles" generator, which fails if nmake is not on PATH.  Detect
# this early and print actionable guidance instead of a cryptic error.
if(CMAKE_HOST_SYSTEM_NAME STREQUAL "Windows" AND CMAKE_GENERATOR STREQUAL "NMake Makefiles")
    find_program(_NMAKE_FOUND nmake)
    if(NOT _NMAKE_FOUND)
        message(FATAL_ERROR
            "CMake selected the 'NMake Makefiles' generator but nmake was not found.\n"
            "Please try one of the following:\n"
            "  1. Open 'Developer Command Prompt for Visual Studio' and re-run cmake.\n"
            "  2. Specify a generator with the -G flag, for example:\n"
            "       cmake .. -G \"Visual Studio 17 2022\"\n"
            "       cmake .. -G \"Visual Studio 16 2019\"\n"
            "       cmake .. -G \"MinGW Makefiles\"\n"
            "       cmake .. -G \"Ninja\"\n"
            "  3. Use a CMake preset:\n"
            "       cmake --preset windows-vs2022\n"
            "       cmake --preset windows-vs2019\n"
        )
    endif()
endif()

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
elseif(MSVC)
    add_compile_options(/W3 /O2)
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
    target_compile_definitions(llm PRIVATE LLM_USE_CUDA)
    target_link_libraries(llm PRIVATE CUDA::cudart CUDA::cublas)
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        # Auto-detect local GPU when CMake supports it; otherwise pick
        # architectures that the installed CUDA toolkit actually supports.
        if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
            set_target_properties(llm PROPERTIES CUDA_ARCHITECTURES "native")
        elseif(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13.0")
            set_target_properties(llm PROPERTIES CUDA_ARCHITECTURES "75;80;86;89;90")
        elseif(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.0")
            set_target_properties(llm PROPERTIES CUDA_ARCHITECTURES "70;75;80;86;89;90")
        else()
            set_target_properties(llm PROPERTIES CUDA_ARCHITECTURES "70;75;80")
        endif()
    endif()
endif()

# Math library on Linux
if(UNIX AND NOT APPLE)
    target_link_libraries(llm PRIVATE m)
endif()

# Winsock library on Windows (for HTTP server)
if(WIN32)
    target_link_libraries(llm PRIVATE ws2_32)
endif()

# Tests
enable_testing()
add_executable(test_gguf tests/test_gguf.cpp)
target_include_directories(test_gguf PRIVATE src)
if(UNIX AND NOT APPLE)
    target_link_libraries(test_gguf PRIVATE m)
endif()
add_test(NAME test_gguf COMMAND test_gguf)

if(LLM_CUDA)
    add_executable(test_gpu tests/test_gpu.cu src/cuda_kernels.cu)
    target_include_directories(test_gpu PRIVATE src)
    target_compile_definitions(test_gpu PRIVATE LLM_USE_CUDA)
    target_link_libraries(test_gpu PRIVATE CUDA::cudart CUDA::cublas)
    if(UNIX AND NOT APPLE)
        target_link_libraries(test_gpu PRIVATE m)
    endif()
    if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
        if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24")
            set_target_properties(test_gpu PROPERTIES CUDA_ARCHITECTURES "native")
        elseif(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "13.0")
            set_target_properties(test_gpu PROPERTIES CUDA_ARCHITECTURES "75;80;86;89;90")
        elseif(CUDAToolkit_VERSION VERSION_GREATER_EQUAL "12.0")
            set_target_properties(test_gpu PROPERTIES CUDA_ARCHITECTURES "70;75;80;86;89;90")
        else()
            set_target_properties(test_gpu PROPERTIES CUDA_ARCHITECTURES "70;75;80")
        endif()
    endif()
    add_test(NAME test_gpu COMMAND test_gpu)
endif()
