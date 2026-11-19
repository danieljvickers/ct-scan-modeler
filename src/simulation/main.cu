// placeholder simulation: one MPI rank per GPU, each prints from a kernel to check the build and launch path
#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>
#include <mpi.h>

#ifdef CTM_CASE_OPTIMIZATION
#include "case_parameters.hpp"
#endif

#define CUDA_CHECK(call)                                                                                   \
    do {                                                                                                   \
        cudaError_t error = (call);                                                                        \
        if (error != cudaSuccess) {                                                                        \
            std::fprintf(stderr, "ERROR :: %s at %s:%d\n", cudaGetErrorString(error), __FILE__, __LINE__); \
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                                                       \
        }                                                                                                  \
    } while (0)

__global__ void helloKernel(int rank, int device) {
    std::printf("INFO :: hello from thread %d of block %d on rank %d, GPU %d\n", threadIdx.x, blockIdx.x, rank, device);
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    // ranks on the same node share its GPUs round robin
    MPI_Comm node_comm;
    MPI_Comm_split_type(MPI_COMM_WORLD, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &node_comm);
    int local_rank;
    MPI_Comm_rank(node_comm, &local_rank);
    int num_devices;
    CUDA_CHECK(cudaGetDeviceCount(&num_devices));
    int device = local_rank % num_devices;
    CUDA_CHECK(cudaSetDevice(device));

    // report the build and input on rank 0
    if (rank == 0) {
        std::printf("INFO :: input file %s\n", argc > 1 ? argv[1] : "(none)");
#ifdef CTM_CASE_OPTIMIZATION
        std::printf("INFO :: case optimized build, %s\n", case_parameters::summary);
#else
        std::printf("INFO :: generic build, case parameters are read at run time\n");
#endif
    }
    MPI_Barrier(MPI_COMM_WORLD);

    // say hello from every rank's GPU
    char hostname[MPI_MAX_PROCESSOR_NAME];
    int hostname_length;
    MPI_Get_processor_name(hostname, &hostname_length);
    cudaDeviceProp properties;
    CUDA_CHECK(cudaGetDeviceProperties(&properties, device));
    std::printf("INFO :: rank %d of %d on %s using GPU %d (%s)\n", rank, num_ranks, hostname, device, properties.name);
    helloKernel<<<1, 2>>>(rank, device);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    MPI_Comm_free(&node_comm);
    MPI_Finalize();
    return EXIT_SUCCESS;
}
