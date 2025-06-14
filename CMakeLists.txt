cmake_minimum_required(VERSION 3.15)
project(EEGPipeline LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
find_package(OpenMP REQUIRED)
enable_language(CUDA)

# liblsl as submodule
add_subdirectory(external/liblsl)

add_executable(eeg_app
  src/main.cpp
  src/LSLAdapter.cpp
  src/RingBuffer.cpp
  src/FilterProcessor.cpp
  # ... add other source files here
)
target_include_directories(eeg_app PRIVATE
  external/liblsl/include
  include
)
target_link_libraries(eeg_app PRIVATE
  lsl
  OpenMP::OpenMP_CXX
  CUDA::cudart
)
