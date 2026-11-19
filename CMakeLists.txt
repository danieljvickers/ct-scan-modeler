cmake_minimum_required(VERSION 3.24)
project(ctm LANGUAGES CXX CUDA)

# case optimization compiles the case parameters in as constants, the toolchain writes the header
option(CTM_CASE_OPTIMIZATION "compile case parameters in as constants" OFF)
set(CTM_CASE_HEADER_DIR "" CACHE PATH "directory holding the generated case_parameters.hpp")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
if(NOT CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif()

find_package(MPI REQUIRED COMPONENTS CXX)
find_package(CUDAToolkit REQUIRED)
find_package(HDF5 REQUIRED COMPONENTS C)
if(NOT HDF5_IS_PARALLEL)
    message(FATAL_ERROR "HDF5 at ${HDF5_ROOT} is not built with MPI")
endif()

add_executable(simulation
    src/simulation/main.cu
    src/simulation/global_parameters.cu
    src/simulation/grid.cu
    src/simulation/mpi_proxy.cu
    src/simulation/boundary_conditions.cu
    src/simulation/geometry.cu
    src/simulation/ibm.cu
    src/simulation/linear_solvers.cu
    src/simulation/right_hand_side.cu
    src/simulation/out.cu
)
target_link_libraries(simulation PRIVATE MPI::MPI_CXX CUDA::cufft CUDA::cublas ${HDF5_C_LIBRARIES})
target_include_directories(simulation PRIVATE ${HDF5_C_INCLUDE_DIRS})
set_target_properties(simulation PROPERTIES INSTALL_RPATH "${HDF5_ROOT}/lib" BUILD_WITH_INSTALL_RPATH ON)

if(CTM_CASE_OPTIMIZATION)
    if(NOT EXISTS "${CTM_CASE_HEADER_DIR}/case_parameters.hpp")
        message(FATAL_ERROR "CTM_CASE_OPTIMIZATION is on but ${CTM_CASE_HEADER_DIR}/case_parameters.hpp does not exist")
    endif()
    target_compile_definitions(simulation PRIVATE CTM_CASE_OPTIMIZATION)
    target_include_directories(simulation PRIVATE "${CTM_CASE_HEADER_DIR}")
endif()

install(TARGETS simulation RUNTIME DESTINATION bin)
