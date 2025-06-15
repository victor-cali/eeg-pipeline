cmake_minimum_required(VERSION 3.15)
project(LSLDemo LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)

# Add liblsl as submodule (assumed at external/liblsl)
add_subdirectory(external/liblsl)
add_library(LSL::lsl ALIAS lsl)

# Add your executable
add_executable(lsl_demo
    src/main.cpp
    src/LSLAdapter.cpp
    src/RingBuffer.cpp
)

add_executable(ringbuffer_test
    src/ringbuffer_test.cpp    # <--- new test file
    src/RingBuffer.cpp
)

target_include_directories(ringbuffer_test PRIVATE include)


# Include directory for LSLAdapter.hpp
target_include_directories(lsl_demo PRIVATE include)

# Link to liblsl target
target_link_libraries(lsl_demo PRIVATE LSL::lsl)
