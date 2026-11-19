#include "mpi_proxy.hpp"

#include <algorithm>

real allreduceSum(real value) {
    real total;
    MPI_Allreduce(&value, &total, 1, mpi_real, MPI_SUM, MPI_COMM_WORLD);
    return total;
}

real allreduceMax(real value) {
    real maximum;
    MPI_Allreduce(&value, &maximum, 1, mpi_real, MPI_MAX, MPI_COMM_WORLD);
    return maximum;
}

void allreduceSum(real* values, int count) {
    MPI_Allreduce(MPI_IN_PLACE, values, count, mpi_real, MPI_SUM, MPI_COMM_WORLD);
}

HaloExchange::HaloExchange(const GridView& grid) : grid(grid) {
    this->plane_size = grid.stride(grid.decomposed_axis);
    const size_t bytes = halo_width*this->plane_size*sizeof(real);
    CUDA_CHECK(cudaMallocHost(&this->send_lower, bytes));
    CUDA_CHECK(cudaMallocHost(&this->send_upper, bytes));
    CUDA_CHECK(cudaMallocHost(&this->receive_lower, bytes));
    CUDA_CHECK(cudaMallocHost(&this->receive_upper, bytes));
}

HaloExchange::~HaloExchange() {
    cudaFreeHost(this->send_lower);
    cudaFreeHost(this->send_upper);
    cudaFreeHost(this->receive_lower);
    cudaFreeHost(this->receive_upper);
}

void HaloExchange::exchange(real* field) {
    if (mpiSize() == 1) return;
    const int axis = this->grid.decomposed_axis;
    const int lower_rank = this->grid.lower_boundary[axis] ? MPI_PROC_NULL : mpiRank() - 1;
    const int upper_rank = this->grid.upper_boundary[axis] ? MPI_PROC_NULL : mpiRank() + 1;
    const long count = halo_width*this->plane_size;
    const size_t bytes = count*sizeof(real);
    const int cells = this->grid.cells[axis];

    // owned planes next to each neighbor go out, halo planes come in; storage plane = local plane + halo_width
    real* first_owned = field + halo_width*this->plane_size;
    real* last_owned = field + cells*this->plane_size;
    real* lower_halo = field;
    real* upper_halo = field + (cells + halo_width)*this->plane_size;
    if (upper_rank != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpy(this->send_upper, last_owned, bytes, cudaMemcpyDeviceToHost));
    if (lower_rank != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpy(this->send_lower, first_owned, bytes, cudaMemcpyDeviceToHost));
    MPI_Sendrecv(this->send_upper, count, mpi_real, upper_rank, 0, this->receive_lower, count, mpi_real, lower_rank, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    MPI_Sendrecv(this->send_lower, count, mpi_real, lower_rank, 1, this->receive_upper, count, mpi_real, upper_rank, 1,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    if (lower_rank != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpy(lower_halo, this->receive_lower, bytes, cudaMemcpyHostToDevice));
    if (upper_rank != MPI_PROC_NULL) CUDA_CHECK(cudaMemcpy(upper_halo, this->receive_upper, bytes, cudaMemcpyHostToDevice));
}

namespace {

// compact [slab index][row] to send order [row][slab index], rows are grouped by destination rank
__global__ void packSlabsKernel(const real* compact, real* send, long num_rows, int slab_cells) {
    const long element = blockIdx.x*static_cast<long>(blockDim.x) + threadIdx.x;
    if (element >= num_rows*slab_cells) return;
    const long row = element % num_rows;
    const int slab_index = static_cast<int>(element/num_rows);
    send[row*slab_cells + slab_index] = compact[element];
}

// blocks from each source rank, [local row][source slab index], into pencils [local row][full axis]
__global__ void unpackPencilsKernel(const real* receive, real* pencils, int local_rows, int axis_cells, int num_ranks,
                                    const int* slab_offsets, const int* slab_counts, const int* receive_displacements) {
    const long element = blockIdx.x*static_cast<long>(blockDim.x) + threadIdx.x;
    if (element >= static_cast<long>(local_rows)*axis_cells) return;
    int source = 0;
    while (source + 1 < num_ranks && element >= receive_displacements[source + 1]) ++source;
    const long within = element - receive_displacements[source];
    const long local_row = within/slab_counts[source];
    const int axis_index = slab_offsets[source] + static_cast<int>(within % slab_counts[source]);
    pencils[local_row*axis_cells + axis_index] = receive[element];
}

// pencils back to blocks per destination rank, the exact inverse of unpackPencilsKernel
__global__ void packPencilsKernel(const real* pencils, real* send, int local_rows, int axis_cells, int num_ranks,
                                  const int* slab_offsets, const int* slab_counts, const int* receive_displacements) {
    const long element = blockIdx.x*static_cast<long>(blockDim.x) + threadIdx.x;
    if (element >= static_cast<long>(local_rows)*axis_cells) return;
    const long local_row = element/axis_cells;
    const int axis_index = static_cast<int>(element % axis_cells);
    const int destination = partOwner(axis_cells, num_ranks, axis_index);
    send[receive_displacements[destination] + local_row*slab_counts[destination] + axis_index - slab_offsets[destination]] = pencils[element];
}

// send order [row][slab index] back to compact [slab index][row]
__global__ void unpackSlabsKernel(const real* receive, real* compact, long num_rows, int slab_cells) {
    const long element = blockIdx.x*static_cast<long>(blockDim.x) + threadIdx.x;
    if (element >= num_rows*slab_cells) return;
    const long row = element/slab_cells;
    const int slab_index = static_cast<int>(element % slab_cells);
    compact[slab_index*num_rows + row] = receive[element];
}

}  // namespace

SlabTranspose::SlabTranspose(const GridView& grid) {
    const int axis = grid.decomposed_axis;
    this->num_ranks = mpiSize();
    this->rank = mpiRank();
    this->axis_cells = grid.global_cells[axis];
    this->slab_cells = grid.cells[axis];
    this->slab_offset = grid.offset[axis];
    this->num_rows = static_cast<int>(grid.numOwned()/grid.cells[axis]);
    CTM_REQUIRE(this->num_rows >= this->num_ranks, "too many ranks for the transform transpose");
    this->local_rows = partCount(this->num_rows, this->num_ranks, this->rank);
    this->row_offset = partOffset(this->num_rows, this->num_ranks, this->rank);

    // message sizes follow from the row split and the slab split
    std::vector<int> slab_offsets(this->num_ranks), slab_counts(this->num_ranks);
    this->send_counts.resize(this->num_ranks);
    this->send_displacements.resize(this->num_ranks);
    this->receive_counts.resize(this->num_ranks);
    this->receive_displacements.resize(this->num_ranks);
    for (int other = 0; other < this->num_ranks; ++other) {
        slab_offsets[other] = partOffset(this->axis_cells, this->num_ranks, other);
        slab_counts[other] = partCount(this->axis_cells, this->num_ranks, other);
        this->send_counts[other] = partCount(this->num_rows, this->num_ranks, other)*this->slab_cells;
        this->send_displacements[other] = partOffset(this->num_rows, this->num_ranks, other)*this->slab_cells;
        this->receive_counts[other] = this->local_rows*slab_counts[other];
        this->receive_displacements[other] = this->local_rows*slab_offsets[other];
    }
    this->device_slab_offsets.copyFromHost(slab_offsets.data(), this->num_ranks);
    this->device_slab_counts.copyFromHost(slab_counts.data(), this->num_ranks);
    this->device_receive_displacements.copyFromHost(this->receive_displacements.data(), this->num_ranks);

    const long slab_elements = static_cast<long>(this->num_rows)*this->slab_cells;
    const long pencil_elements = static_cast<long>(this->local_rows)*this->axis_cells;
    this->device_send.resize(std::max(slab_elements, pencil_elements));
    this->device_receive.resize(std::max(slab_elements, pencil_elements));
    CUDA_CHECK(cudaMallocHost(&this->host_send, std::max(slab_elements, pencil_elements)*sizeof(real)));
    CUDA_CHECK(cudaMallocHost(&this->host_receive, std::max(slab_elements, pencil_elements)*sizeof(real)));
}

SlabTranspose::~SlabTranspose() {
    cudaFreeHost(this->host_send);
    cudaFreeHost(this->host_receive);
}

void SlabTranspose::toPencils(const real* compact, real* pencils) {
    const long slab_elements = static_cast<long>(this->num_rows)*this->slab_cells;
    const long pencil_elements = static_cast<long>(this->local_rows)*this->axis_cells;
    packSlabsKernel<<<numBlocks(slab_elements), block_size>>>(compact, this->device_send.data(), this->num_rows, this->slab_cells);
    CUDA_CHECK(cudaMemcpy(this->host_send, this->device_send.data(), slab_elements*sizeof(real), cudaMemcpyDeviceToHost));
    MPI_Alltoallv(this->host_send, this->send_counts.data(), this->send_displacements.data(), mpi_real, this->host_receive,
                  this->receive_counts.data(), this->receive_displacements.data(), mpi_real, MPI_COMM_WORLD);
    CUDA_CHECK(cudaMemcpy(this->device_receive.data(), this->host_receive, pencil_elements*sizeof(real), cudaMemcpyHostToDevice));
    unpackPencilsKernel<<<numBlocks(pencil_elements), block_size>>>(this->device_receive.data(), pencils, this->local_rows, this->axis_cells,
                                                                    this->num_ranks, this->device_slab_offsets.data(),
                                                                    this->device_slab_counts.data(), this->device_receive_displacements.data());
    CUDA_CHECK(cudaGetLastError());
}

void SlabTranspose::toSlabs(const real* pencils, real* compact) {
    const long slab_elements = static_cast<long>(this->num_rows)*this->slab_cells;
    const long pencil_elements = static_cast<long>(this->local_rows)*this->axis_cells;
    packPencilsKernel<<<numBlocks(pencil_elements), block_size>>>(pencils, this->device_send.data(), this->local_rows, this->axis_cells,
                                                                  this->num_ranks, this->device_slab_offsets.data(),
                                                                  this->device_slab_counts.data(), this->device_receive_displacements.data());
    CUDA_CHECK(cudaMemcpy(this->host_send, this->device_send.data(), pencil_elements*sizeof(real), cudaMemcpyDeviceToHost));
    MPI_Alltoallv(this->host_send, this->receive_counts.data(), this->receive_displacements.data(), mpi_real, this->host_receive,
                  this->send_counts.data(), this->send_displacements.data(), mpi_real, MPI_COMM_WORLD);
    CUDA_CHECK(cudaMemcpy(this->device_receive.data(), this->host_receive, slab_elements*sizeof(real), cudaMemcpyHostToDevice));
    unpackSlabsKernel<<<numBlocks(slab_elements), block_size>>>(this->device_receive.data(), compact, this->num_rows, this->slab_cells);
    CUDA_CHECK(cudaGetLastError());
}
