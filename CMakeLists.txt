cmake_minimum_required(VERSION 3.24)
project(canticle LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(CMAKE_CUDA_ARCHITECTURES 89)
set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG")
set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math -Xptxas -O3 -Xcompiler=/O2")

set(CAN_ROOT ${CMAKE_CURRENT_SOURCE_DIR})

file(GLOB_RECURSE CAN_SRC_CU  ${CAN_ROOT}/src/*.cu)
file(GLOB_RECURSE CAN_SRC_CPP ${CAN_ROOT}/src/*.cpp)

add_executable(canticle ${CAN_SRC_CPP} ${CAN_SRC_CU})
target_include_directories(canticle PRIVATE ${CAN_ROOT}/include)
target_compile_definitions(canticle PRIVATE CAN_ROOT="${CAN_ROOT}")

find_package(CUDAToolkit REQUIRED)
target_link_libraries(canticle PRIVATE CUDA::cudart)

message(STATUS "canticle: CUDA ${CUDAToolkit_VERSION}, arch ${CMAKE_CUDA_ARCHITECTURES}")
