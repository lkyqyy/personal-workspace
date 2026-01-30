cmake_minimum_required(VERSION 3.18)
project(PolypSegmentationTRT LANGUAGES CXX CUDA)

# -----------------------------------------------------------------------------
# 语言与标准
# -----------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 11)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# -----------------------------------------------------------------------------
# 构建类型与输出目录
# -----------------------------------------------------------------------------
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()
set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS Release Debug RelWithDebInfo MinSizeRel)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/bin)

# Debug：带符号、低优化；Release：优化（仅 CXX，避免与 CUDA 混用导致 -O 报错）
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
    set(CMAKE_CXX_FLAGS_RELEASE "-O3")
endif()

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# -----------------------------------------------------------------------------
# 依赖：OpenCV、TensorRT、CUDA
# -----------------------------------------------------------------------------
find_package(OpenCV REQUIRED)
find_package(CUDAToolkit REQUIRED)

set(TENSORRT_ROOT "/home/lk/TensorRT-8.6.1.6" CACHE PATH "TensorRT 根目录")
set(TENSORRT_INCLUDE_DIR "${TENSORRT_ROOT}/include")
set(TENSORRT_LIB_DIR     "${TENSORRT_ROOT}/lib")

find_library(TENSORRT_nvinfer NAMES nvinfer
    PATHS ${TENSORRT_LIB_DIR} ${TENSORRT_ROOT}/lib64 NO_DEFAULT_PATH)
find_library(TENSORRT_nvonnxparser NAMES nvonnxparser
    PATHS ${TENSORRT_LIB_DIR} ${TENSORRT_ROOT}/lib64 NO_DEFAULT_PATH)
if(NOT TENSORRT_nvinfer)
    message(FATAL_ERROR "TensorRT nvinfer not found under ${TENSORRT_ROOT}")
endif()

# -----------------------------------------------------------------------------
# 源文件与可执行目标
# -----------------------------------------------------------------------------
set(APP_SOURCES
    src/main.cpp
    src/InferenceEngine.cpp
    src/Preprocessor.cpp
    src/Postprocessor.cpp
    src/Metrics.cpp
)

add_executable(${PROJECT_NAME} ${APP_SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${OpenCV_INCLUDE_DIRS}
    ${TENSORRT_INCLUDE_DIR}
    ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    ${OpenCV_LIBS}
    ${TENSORRT_nvinfer}
    CUDA::cudart
)
if(TENSORRT_nvonnxparser)
    target_link_libraries(${PROJECT_NAME} PRIVATE ${TENSORRT_nvonnxparser})
endif()

# -----------------------------------------------------------------------------
# 可选：CUDA 内核（存在 src/*.cu 时再链接）
# -----------------------------------------------------------------------------
file(GLOB CUDA_SOURCES "src/*.cu")
if(CUDA_SOURCES)
    add_library(cuda_kernels STATIC ${CUDA_SOURCES})
    target_include_directories(cuda_kernels PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${TENSORRT_INCLUDE_DIR}
        ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES}
    )
    target_link_libraries(cuda_kernels PUBLIC CUDA::cudart)
    target_link_libraries(${PROJECT_NAME} PRIVATE cuda_kernels)
endif()

# -----------------------------------------------------------------------------
# 功能模块单元测试（googletest，不依赖 TensorRT）
# 使用 FetchContent 拉取 googletest，避免误用系统/conda 下其他平台的 GTest。
# -----------------------------------------------------------------------------
option(BUILD_TESTS "Build unit tests with Google Test" ON)
if(BUILD_TESTS)
    include(FetchContent)
    set(FETCHCONTENT_QUIET FALSE)
    FetchContent_Declare(
        googletest
        GIT_REPOSITORY https://github.com/google/googletest.git
        GIT_TAG        v1.14.0
    )
    set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
    FetchContent_MakeAvailable(googletest)
    if(TARGET GTest::gtest_main)
        enable_testing()
        add_executable(PolypSegmentationTRT_tests
            tests/test_preprocessor.cpp
            tests/test_postprocessor.cpp
            tests/test_metrics.cpp
            src/Preprocessor.cpp
            src/Postprocessor.cpp
            src/Metrics.cpp
        )
        target_include_directories(PolypSegmentationTRT_tests PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/include
            ${OpenCV_INCLUDE_DIRS}
        )
        target_link_libraries(PolypSegmentationTRT_tests PRIVATE
            GTest::gtest_main
            ${OpenCV_LIBS}
        )
        include(GoogleTest)
        gtest_discover_tests(PolypSegmentationTRT_tests)
        message(STATUS "Unit tests enabled: PolypSegmentationTRT_tests (googletest via FetchContent)")
    endif()
endif()
