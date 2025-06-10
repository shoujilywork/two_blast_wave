cmake_minimum_required(VERSION 3.18)
project(TWO_BLAST_WAVE LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
enable_language(CUDA)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES "75")  # T4 GPU
# find Eigen
find_package(Eigen3 REQUIRED)
find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)

# 
add_executable(main 
    src/main_omp.cpp
    src/dx_p.cpp
    src/dx_n.cpp
    src/dy_p.cpp
    src/dy_n.cpp
    src/nov_5.cpp
    src/boundary.cpp
    src/initial.cpp
    src/split_lf.cu
)


set_target_properties(main PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_include_directories(main PRIVATE
    ${EIGEN3_INCLUDE_DIR}
    include/
)

target_link_libraries(main PRIVATE
    OpenMP::OpenMP_CXX
)
