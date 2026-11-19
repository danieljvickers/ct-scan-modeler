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

add_executable(simulation src/simulation/main.cu)
target_link_libraries(simulation PRIVATE MPI::MPI_CXX)

if(CTM_CASE_OPTIMIZATION)
    if(NOT EXISTS "${CTM_CASE_HEADER_DIR}/case_parameters.hpp")
        message(FATAL_ERROR "CTM_CASE_OPTIMIZATION is on but ${CTM_CASE_HEADER_DIR}/case_parameters.hpp does not exist")
    endif()
    target_compile_definitions(simulation PRIVATE CTM_CASE_OPTIMIZATION)
    target_include_directories(simulation PRIVATE "${CTM_CASE_HEADER_DIR}")
endif()

install(TARGETS simulation RUNTIME DESTINATION bin)
