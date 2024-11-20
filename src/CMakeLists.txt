# cmake_minimum_required(VERSION 3.22)

# set(CMAKE_CUDA_STANDARD 17)
# set(CMAKE_CUDA_STANDARD_REQUIRED ON)
# set(CMAKE_CUDA_ARCHITECTURES 80)
# project(EnlargingTensors LANGUAGES CUDA CXX)

# add_library(main STATIC main.cu)
# set_source_files_properties(main.cu PROPERTIES LANGUAGE CUDA)
# set_target_properties(main PROPERTIES
#         CUDA_SEPARABLE_COMPILATION ON
# #        CUDA_RUNTIME_LIBRARY Shared
# #        POSITION_INDEPENDENT_CODE ON
# )

# add_executable(off_vector off_vector.cu)
# add_dependencies(off_vector main)
# target_link_libraries(off_vector PUBLIC main)
# set_source_files_properties(off_vector.cu PROPERTIES LANGUAGE CUDA)
# set_target_properties(off_vector PROPERTIES
#         CUDA_SEPARABLE_COMPILATION ON
# #        CUDA_RUNTIME_LIBRARY Shared
# #        POSITION_INDEPENDENT_CODE ON
# )

add_executable(program main.cu OffLayer.cu)

set_target_properties(
    program PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)