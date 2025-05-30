﻿cmake_minimum_required(VERSION 3.18)
project(yolov12-tensorrt LANGUAGES CXX CUDA)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
add_definitions(-DAPI_EXPORTS)

# Add source files
set(SOURCES
    main.cpp
    src/preprocess.cu
)

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/src/YOLOv12.cpp")
    list(APPEND SOURCES src/YOLOv12.cpp)
else()
    message(WARNING "src/yolov12.cpp not found. Please check your sources.")
endif()

# Add headers (optional for IDE support)
set(HEADERS
    src/YOLOv12.h
    src/macros.h
    src/logging.h
    src/cuda_utils.h
    src/preprocess.h
    src/common.h
)

# Set paths for OpenCV and TensorRT
set(OpenCV_DIR "your OpenCV build directory path")
set(TENSORRT_DIR "your tensorrt path")

# Find OpenCV
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

# Include TensorRT
include_directories(${TENSORRT_DIR}/include)
link_directories(${TENSORRT_DIR}/lib)
set(TENSORRT_LIBS nvinfer nvinfer_plugin nvparsers nvonnxparser)

# Add include directories
include_directories(src/)

# Create executable
add_executable(${PROJECT_NAME} ${SOURCES} ${HEADERS})

# Link libraries
target_link_libraries(${PROJECT_NAME}
    ${OpenCV_LIBS}
    ${TENSORRT_LIBS}
)