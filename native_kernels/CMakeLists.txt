cmake_minimum_required(VERSION 3.1)
project(native_kernels)

find_package(Legion REQUIRED)

list(APPEND NATIVE_KERNELS_SRC
  lifeline_mapper.cc       lifeline_mapper.h
  phaser_tasks.cc phaser_tasks.h
)

if(Legion_USE_CUDA)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -Wno-deprecated-gpu-targets)
  list(APPEND NATIVE_KERNELS_SRC
    phaser_tasks_gpu.cu
  )
endif()

if(Legion_USE_CUDA)
  cuda_add_library(native_kernels SHARED ${NATIVE_KERNELS_SRC})
  cuda_add_cufft_to_target(native_kernels)
else()
  add_library(native_kernels SHARED ${NATIVE_KERNELS_SRC})
endif()
target_link_libraries(native_kernels Legion::Legion)

set_property(TARGET native_kernels PROPERTY CXX_STANDARD 11)
set_property(TARGET native_kernels PROPERTY CXX_STANDARD_REQUIRED ON)
