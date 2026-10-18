cmake_minimum_required(VERSION 3.24)

# ===========================================================================
#  CUDA IS OPTIONAL.
#
#  The project's goal is a model that runs on any low-performance device. The
#  build used to require CUDA unconditionally, which made that goal false in the
#  most direct way possible: the code would not compile on a machine without it.
#
#  Now: if a CUDA toolkit is found, build the GPU path (0.22 ms/query).
#  If not, build the CPU path (am_kernels_cpu.cpp, slower but correct) and say so
#  loudly, because a silent fallback would let someone believe they measured the
#  fast path when they did not.
#
#  Force either with:  cmake -DCANTICLE_FORCE_CPU=ON
# ===========================================================================

option(CANTICLE_FORCE_CPU "Build without CUDA even if a toolkit is present" OFF)

set(CAN_USE_CUDA OFF)
if(NOT CANTICLE_FORCE_CPU)
  include(CheckLanguage)
  check_language(CUDA)
  if(CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    find_package(CUDAToolkit QUIET)
    if(CUDAToolkit_FOUND)
      set(CAN_USE_CUDA ON)
    endif()
  endif()
endif()

if(CAN_USE_CUDA)
  project(canticle LANGUAGES CXX CUDA)
else()
  project(canticle LANGUAGES CXX)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "/O2 /DNDEBUG")

set(CAN_ROOT ${CMAKE_CURRENT_SOURCE_DIR})
file(GLOB_RECURSE CAN_SRC_CPP ${CAN_ROOT}/src/*.cpp)

if(CAN_USE_CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)
  set(CMAKE_CUDA_ARCHITECTURES 89)
  set(CMAKE_CUDA_FLAGS_RELEASE "-O3 --use_fast_math -Xptxas -O3 -Xcompiler=/O2")
  file(GLOB_RECURSE CAN_SRC_CU ${CAN_ROOT}/src/*.cu)
  list(REMOVE_ITEM CAN_SRC_CPP ${CAN_ROOT}/src/am_kernels_cpu.cpp)
  add_executable(canticle ${CAN_SRC_CPP} ${CAN_SRC_CU})
  target_link_libraries(canticle PRIVATE CUDA::cudart)
  message(STATUS "canticle: CUDA ${CUDAToolkit_VERSION}, arch ${CMAKE_CUDA_ARCHITECTURES}")
else()
  # The CPU build carries ROUTE B ONLY (the dialogue / tool-routing line).
  #
  # Route A (ternary GEMM benchmarks, STE training, the tokenizer model) is
  # GPU-specific by nature: it exists to measure bandwidth-bound kernels. A CPU
  # implementation of a bandwidth benchmark measures the host memory bus, which
  # is not the question it was built to answer.
  #
  # So those sources are excluded rather than stubbed. The commands they back
  # report "not built" instead of silently doing nothing -- see main.cpp.
  #
  # Everything route B needs is portable: am.cpp, chat.cpp and all 29 am-* suites
  # contain zero CUDA references. That is why this list is short.
  # ONLY the sources that genuinely need a GPU. The first version of this list
  # also excluded hybrid.cpp, data.cpp, tokenizer.cpp, lex.cpp and model.cpp --
  # grouped under "route A" by directory -- and linking then failed with
  # seventeen unresolved symbols. Those five are plain CPU code that route B
  # depends on (LexModel, BPETokenizer, synth_tool_data, hybrid_key).
  #
  # The rule is: remove a file only after the linker complains about it, not
  # because of where it lives in the tree.
  set(CAN_NEEDS_GPU
    src/sweep.cpp src/train_main.cpp src/ftrain_main.cpp src/smoke_model.cpp
    src/head_train.cpp src/align.cpp src/diag.cpp src/diag2.cpp
    src/dump.cpp src/tokdiag.cpp src/ckpt.cpp src/model.cpp src/chat_main.cpp
  )
  foreach(f ${CAN_NEEDS_GPU})
    list(REMOVE_ITEM CAN_SRC_CPP ${CAN_ROOT}/${f})
  endforeach()
  list(REMOVE_ITEM CAN_SRC_CPP ${CAN_ROOT}/src/am_kernels_cpu.cpp)
  add_executable(canticle ${CAN_SRC_CPP} ${CAN_ROOT}/src/am_kernels_cpu.cpp)
  target_compile_definitions(canticle PRIVATE CANTICLE_CPU)
  message(WARNING "canticle: CPU build -- route B (dialogue/tool routing) only. "
                  "Route A benchmarks and training are not included; retrieval "
                  "will be slower than the CUDA path. Run 'canticle am-bench' "
                  "to measure what you actually got instead of assuming.")
endif()

target_include_directories(canticle PRIVATE ${CAN_ROOT}/include)
target_compile_definitions(canticle PRIVATE CAN_ROOT="${CAN_ROOT}")
