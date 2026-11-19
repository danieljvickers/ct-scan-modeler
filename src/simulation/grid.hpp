// uniform grid and slab decomposition along the slowest axis, following MFC's m_grid
#ifndef CTM_GRID_HPP
#define CTM_GRID_HPP

#include "global_parameters.hpp"

constexpr int halo_width = 3;  // decomposition halos reach image point stencils on the neighbor rank

// plain view of the local grid, passed to kernels by value
struct GridView {
    int num_dims;
    int decomposed_axis;              // slowest axis: y in 2D, z in 3D
    int cells[max_dims];              // owned cells, 1 in an inactive direction
    int halo[max_dims];               // halo width, 0 in an inactive direction
    int storage[max_dims];            // cells + 2*halo
    int offset[max_dims];             // global index of the first owned cell
    int global_cells[max_dims];
    real origin[max_dims];            // domain begin
    real spacing[max_dims];           // cell width
    bool periodic[max_dims];          // the direction wraps around
    bool lower_boundary[max_dims];    // this rank touches the physical begin, never true for a periodic direction
    bool upper_boundary[max_dims];    // this rank touches the physical end, never true for a periodic direction

    // index of local cell (i, j, k) in a field with halos, negative indices reach into the halo
    __host__ __device__ long storageIndex(int i, int j, int k) const {
        return (static_cast<long>(k + this->halo[2])*this->storage[1] + (j + this->halo[1]))*this->storage[0] + (i + this->halo[0]);
    }
    // index of owned cell (i, j, k) in a compact vector without halos
    __host__ __device__ long ownedIndex(int i, int j, int k) const {
        return (static_cast<long>(k)*this->cells[1] + j)*this->cells[0] + i;
    }
    __host__ __device__ long numOwned() const { return static_cast<long>(this->cells[0])*this->cells[1]*this->cells[2]; }
    __host__ __device__ long numStorage() const { return static_cast<long>(this->storage[0])*this->storage[1]*this->storage[2]; }
    __host__ __device__ long stride(int direction) const {
        return direction == 0 ? 1 : direction == 1 ? this->storage[0] : static_cast<long>(this->storage[0])*this->storage[1];
    }
    __host__ __device__ real center(int direction, int local_index) const {
        return this->origin[direction] + (this->offset[direction] + local_index + 0.5)*this->spacing[direction];
    }
    __host__ __device__ real cellVolume() const { return this->spacing[0]*this->spacing[1]*this->spacing[2]; }
    __host__ __device__ void ownedCell(long owned, int& i, int& j, int& k) const {
        i = static_cast<int>(owned % this->cells[0]);
        j = static_cast<int>((owned/this->cells[0]) % this->cells[1]);
        k = static_cast<int>(owned/(static_cast<long>(this->cells[0])*this->cells[1]));
    }
};

// number of dimensions, a compile time constant in case optimized builds
__host__ __device__ inline int numDims(const GridView& grid) {
#ifdef CTM_CASE_OPTIMIZATION
    (void)grid;
    return case_parameters::p > 0 ? 3 : 2;
#else
    return grid.num_dims;
#endif
}

// even split of count items over num_parts, first remainder parts get one extra
__host__ __device__ inline int partCount(int count, int num_parts, int part) {
    return count/num_parts + (part < count % num_parts ? 1 : 0);
}
__host__ __device__ inline int partOffset(int count, int num_parts, int part) {
    return part*(count/num_parts) + (part < count % num_parts ? part : count % num_parts);
}
__host__ __device__ inline int partOwner(int count, int num_parts, int index) {
    const int base = count/num_parts, remainder = count % num_parts;
    return index < remainder*(base + 1) ? index/(base + 1) : remainder + (index - remainder*(base + 1))/base;
}

GridView makeGrid(const Parameters& parameters);

#endif
