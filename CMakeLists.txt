cmake_minimum_required(VERSION 3.15)
project(LSLDemo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# Add OpenMP
find_package(OpenMP REQUIRED)
if(OpenMP_FOUND)
    message(STATUS "Found OpenMP, enabling parallelization.")
endif()

# Add liblsl as submodule (assumed at external/liblsl)
add_subdirectory(external/liblsl)
add_library(LSL::lsl ALIAS lsl)

# Add your executable
add_executable(lsl_demo
    src/main.cpp
    src/LSLAdapter.cpp
    src/LSLOutput.cpp 
    src/FeatureExtractor.cpp
    thirdparty/fft/fft.cpp 
)

# Include directory for LSLAdapter.hpp
target_include_directories(lsl_demo PRIVATE 
    include
    thirdparty/fft
)

# Link to liblsl and OpenMP
target_link_libraries(lsl_demo PRIVATE 
    LSL::lsl
    OpenMP::OpenMP_CXX 
)
