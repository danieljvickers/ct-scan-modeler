// shared types, error checks, and kernel launch helpers
#ifndef CTM_COMMON_HPP
#define CTM_COMMON_HPP

#include <cstdio>
#include <cstdlib>

#include <cuda_runtime.h>
#include <mpi.h>

using real = double;  // precision of every field and geometry value
const MPI_Datatype mpi_real = MPI_DOUBLE;  // not constexpr, OpenMPI handles are addresses

constexpr int max_dims = 3;
constexpr int block_size = 256;  // threads per block for 1D kernel launches
constexpr real pi = 3.141592653589793238462643383279502884;

#define CUDA_CHECK(call)                                                                                   \
    do {                                                                                                   \
        cudaError_t error = (call);                                                                        \
        if (error != cudaSuccess) {                                                                        \
            std::fprintf(stderr, "ERROR :: %s at %s:%d\n", cudaGetErrorString(error), __FILE__, __LINE__); \
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                                                       \
        }                                                                                                  \
    } while (0)

#define CTM_REQUIRE(condition, ...)                                                    \
    do {                                                                               \
        if (!(condition)) {                                                            \
            std::fprintf(stderr, "ERROR :: ");                                         \
            std::fprintf(stderr, __VA_ARGS__);                                         \
            std::fprintf(stderr, " (%s:%d)\n", __FILE__, __LINE__);                    \
            MPI_Abort(MPI_COMM_WORLD, EXIT_FAILURE);                                   \
        }                                                                              \
    } while (0)

inline int numBlocks(long num_items) {
    return static_cast<int>((num_items + block_size - 1)/block_size);
}

inline int mpiRank() {
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    return rank;
}

inline int mpiSize() {
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    return size;
}

// device array that frees itself, copyable only by move
template <typename T>
class DeviceArray {
public:
    DeviceArray() = default;
    explicit DeviceArray(long size) { this->resize(size); }
    ~DeviceArray() { if (this->pointer) cudaFree(this->pointer); }
    DeviceArray(const DeviceArray&) = delete;
    DeviceArray& operator=(const DeviceArray&) = delete;
    DeviceArray(DeviceArray&& other) noexcept : pointer(other.pointer), count(other.count) { other.pointer = nullptr; other.count = 0; }
    DeviceArray& operator=(DeviceArray&& other) noexcept {
        if (this != &other) {
            if (this->pointer) cudaFree(this->pointer);
            this->pointer = other.pointer;
            this->count = other.count;
            other.pointer = nullptr;
            other.count = 0;
        }
        return *this;
    }

    void resize(long size) {
        if (this->pointer) CUDA_CHECK(cudaFree(this->pointer));
        this->pointer = nullptr;
        this->count = size;
        if (size > 0) CUDA_CHECK(cudaMalloc(&this->pointer, size*sizeof(T)));
        if (size > 0) CUDA_CHECK(cudaMemset(this->pointer, 0, size*sizeof(T)));
    }
    void copyFromHost(const T* host, long size) {
        if (size != this->count) this->resize(size);
        if (size > 0) CUDA_CHECK(cudaMemcpy(this->pointer, host, size*sizeof(T), cudaMemcpyHostToDevice));
    }
    void copyToHost(T* host) const {
        if (this->count > 0) CUDA_CHECK(cudaMemcpy(host, this->pointer, this->count*sizeof(T), cudaMemcpyDeviceToHost));
    }
    void fillZero() {
        if (this->count > 0) CUDA_CHECK(cudaMemset(this->pointer, 0, this->count*sizeof(T)));
    }

    T* data() { return this->pointer; }
    const T* data() const { return this->pointer; }
    long size() const { return this->count; }

private:
    T* pointer = nullptr;
    long count = 0;
};

#endif
