cmake_minimum_required(VERSION 3.8)
project(osnet_inference)

# Default to Release build for better performance
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Compiler options
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic -std=c++14)
endif()

# Find OpenCV
find_package(OpenCV REQUIRED COMPONENTS core highgui imgproc imgcodecs)

# Set DNN paths
set(DNN_PATH "/usr/include/dnn")
set(DNN_LIB_PATH "/usr/lib/")

include_directories(${DNN_PATH})
link_directories(${DNN_LIB_PATH})

# Build OSNet library
add_library(osnet_lib SHARED
  OSNet.cpp
)

target_include_directories(osnet_lib
  PRIVATE
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(osnet_lib
  ${OpenCV_LIBS}
  dnn
  pthread
  rt
  dl
)

# Main test executable
add_executable(osnet_test
  main.cc
)

target_include_directories(osnet_test
  PRIVATE
    ${OpenCV_INCLUDE_DIRS}
    ${CMAKE_CURRENT_SOURCE_DIR}
)

target_link_libraries(osnet_test
  osnet_lib
  ${OpenCV_LIBS}
  dnn
  pthread
  rt
  dl
)

# Install targets
install(TARGETS osnet_test osnet_lib
  DESTINATION bin
)

# Copy model and test images
configure_file(
  ${CMAKE_CURRENT_SOURCE_DIR}/osnet_64x128_nv12.bin
  ${CMAKE_CURRENT_BINARY_DIR}/osnet_64x128_nv12.bin
  COPYONLY
)

file(COPY test_image DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
