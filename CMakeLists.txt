cmake_minimum_required(VERSION 3.18)
project(originality LANGUAGES CXX CUDA)

# Find CUDA package
add_library(cuda_lcs STATIC
  cuda_code/lcs.cu
)

target_compile_features(cuda_lcs PUBLIC cxx_std_11)

set_target_properties(cuda_lcs PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

add_executable(cuda_lcs_test lcs.cu)

set_property(TARGET cuda_lcs_test PROPERTY CUDA_SEPARABLE_COMPILATION ON)

target_link_libraries(cuda_lcs_test PRIVATE cuda_code)

endif()