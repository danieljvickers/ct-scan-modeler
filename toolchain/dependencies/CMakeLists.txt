# third party libraries, built once per computer into build/dependencies/<computer>/install
cmake_minimum_required(VERSION 3.24)
project(ctm_dependencies LANGUAGES C)
include(ExternalProject)

# parallel HDF5 for the field output, compiled with the MPI C wrapper
ExternalProject_Add(hdf5
    URL https://github.com/HDFGroup/hdf5/releases/download/hdf5_1.14.6/hdf5-1.14.6.tar.gz
    URL_HASH SHA256=e4defbac30f50d64e1556374aa49e574417c9e72c6b1de7a4ff88c4b1bea6e9b
    DOWNLOAD_EXTRACT_TIMESTAMP ON
    CMAKE_ARGS
        -DCMAKE_INSTALL_PREFIX=${CMAKE_INSTALL_PREFIX}
        -DCMAKE_BUILD_TYPE=Release
        -DCMAKE_C_COMPILER=${CMAKE_C_COMPILER}
        -DHDF5_ENABLE_PARALLEL=ON
        -DBUILD_SHARED_LIBS=ON
        -DBUILD_STATIC_LIBS=OFF
        -DBUILD_TESTING=OFF
        -DHDF5_BUILD_EXAMPLES=OFF
        -DHDF5_BUILD_FORTRAN=OFF
        -DHDF5_BUILD_CPP_LIB=OFF
        -DHDF5_BUILD_HL_LIB=OFF
        -DHDF5_BUILD_TOOLS=ON
        -DHDF5_ENABLE_Z_LIB_SUPPORT=OFF
        -DHDF5_ENABLE_SZIP_SUPPORT=OFF
)
