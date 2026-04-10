cmake_minimum_required(VERSION 3.18)

# Set CUDA compiler path before project declaration
set(CMAKE_CUDA_COMPILER /usr/local/cuda-13.2/bin/nvcc)

project(RadioFFT LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set CUDA architecture (RTX 3050 Ti is Ampere = 8.6)
set(CMAKE_CUDA_ARCHITECTURES 86)

# Find CUDA Toolkit
find_package(CUDAToolkit 13.2 REQUIRED)

# Create executable
add_executable(radio_fft main.cu)

# Link cuFFT library
target_link_libraries(radio_fft PRIVATE CUDA::cufft)
