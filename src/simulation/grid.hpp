// uniform grid and slab decomposition along the slowest axis, following MFC's m_grid
#ifndef CTM_GRID_HPP
#define CTM_GRID_HPP

#include <memory>
#include <vector>

#include "global_parameters.hpp"

constexpr int halo_width = 3;  // decomposition halos reach image point stencils on the neighbor rank

// plain view of the local grid, passed to kernels by value
//
// dense storage holds every cell of the box with halos, [z][y][x]. Tiled storage splits that box into tiles of 2^tile_shift
// cells per axis and keeps only the active tiles, one after another; the slot after them stays zero and stands in for every
// cell that is not stored. Owned vectors are compact [z][y][x] when dense and share the storage layout when tiled.
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

    bool tiled = false;
    int tile_shift[max_dims] = {0, 0, 0};  // log2 of the tile width, 0 in an inactive direction
    int tiles[max_dims] = {1, 1, 1};       // tiles per axis over the storage box
    long num_tiles = 0;                    // stored tiles
    const int* tile_slots = nullptr;       // device, flattened tile -> slot, -1 when not stored
    const int* slot_tiles = nullptr;       // device, slot -> flattened tile
    const int* host_tile_slots = nullptr;  // host copies for lookups outside kernels
    const int* host_slot_tiles = nullptr;

    __host__ __device__ long cellsPerTile() const { return 1L << (this->tile_shift[0] + this->tile_shift[1] + this->tile_shift[2]); }
    __host__ __device__ long emptyCell() const { return this->num_tiles*this->cellsPerTile(); }

    // index of local cell (i, j, k) in a field with halos, negative indices reach into the halo
    __host__ __device__ long storageIndex(int i, int j, int k) const {
        if (!this->tiled)
            return (static_cast<long>(k + this->halo[2])*this->storage[1] + (j + this->halo[1]))*this->storage[0] + (i + this->halo[0]);
        const int s[max_dims] = {i + this->halo[0], j + this->halo[1], k + this->halo[2]};
        for (int d = 0; d < max_dims; ++d)
            if (s[d] < 0 || s[d] >= this->storage[d]) return this->emptyCell();
        const long tile = (static_cast<long>(s[2] >> this->tile_shift[2])*this->tiles[1] + (s[1] >> this->tile_shift[1]))*this->tiles[0]
                          + (s[0] >> this->tile_shift[0]);
#ifdef __CUDA_ARCH__
        const int slot = this->tile_slots[tile];
#else
        const int slot = this->host_tile_slots[tile];
#endif
        if (slot < 0) return this->emptyCell();
        long local = s[2] & ((1 << this->tile_shift[2]) - 1);
        local = (local << this->tile_shift[1]) | (s[1] & ((1 << this->tile_shift[1]) - 1));
        local = (local << this->tile_shift[0]) | (s[0] & ((1 << this->tile_shift[0]) - 1));
        return slot*this->cellsPerTile() + local;
    }
    // local cell of a storage index, false for the empty slot and for tile cells past the storage box
    __host__ __device__ bool storageCell(long cell, int& i, int& j, int& k) const {
        if (!this->tiled) {
            if (cell < 0 || cell >= this->numStorage()) return false;
            i = static_cast<int>(cell % this->storage[0]) - this->halo[0];
            j = static_cast<int>((cell/this->storage[0]) % this->storage[1]) - this->halo[1];
            k = static_cast<int>(cell/(static_cast<long>(this->storage[0])*this->storage[1])) - this->halo[2];
            return true;
        }
        const int shift_sum = this->tile_shift[0] + this->tile_shift[1] + this->tile_shift[2];
        const long slot = cell >> shift_sum;
        if (cell < 0 || slot >= this->num_tiles) return false;
        const int local = static_cast<int>(cell & (this->cellsPerTile() - 1));
#ifdef __CUDA_ARCH__
        const long tile = this->slot_tiles[slot];
#else
        const long tile = this->host_slot_tiles[slot];
#endif
        const int tile_x = static_cast<int>(tile % this->tiles[0]), tile_y = static_cast<int>((tile/this->tiles[0]) % this->tiles[1]);
        const int tile_z = static_cast<int>(tile/(static_cast<long>(this->tiles[0])*this->tiles[1]));
        const int s0 = (tile_x << this->tile_shift[0]) | (local & ((1 << this->tile_shift[0]) - 1));
        const int s1 = (tile_y << this->tile_shift[1]) | ((local >> this->tile_shift[0]) & ((1 << this->tile_shift[1]) - 1));
        const int s2 = (tile_z << this->tile_shift[2]) | (local >> (this->tile_shift[0] + this->tile_shift[1]));
        if (s0 >= this->storage[0] || s1 >= this->storage[1] || s2 >= this->storage[2]) return false;
        i = s0 - this->halo[0];
        j = s1 - this->halo[1];
        k = s2 - this->halo[2];
        return true;
    }
    // the cell one or more steps along a direction; dense storage is a plain offset, tiled storage stays inside the tile when it can
    __host__ __device__ long shifted(long cell, int i, int j, int k, int direction, int steps) const {
        if (!this->tiled) return cell + steps*this->stride(direction);
        int index[max_dims] = {i, j, k};
        const int width = 1 << this->tile_shift[direction];
        const int local = (index[direction] + this->halo[direction]) & (width - 1);
        if (cell < this->emptyCell() && local + steps >= 0 && local + steps < width) {
            int lower_shifts = 0;
            for (int d = 0; d < direction; ++d) lower_shifts += this->tile_shift[d];
            return cell + steps*(1L << lower_shifts);
        }
        index[direction] += steps;
        return this->storageIndex(index[0], index[1], index[2]);
    }
    __host__ __device__ bool isOwned(int i, int j, int k) const {
        return i >= 0 && j >= 0 && k >= 0 && i < this->cells[0] && j < this->cells[1] && k < this->cells[2];
    }

    // index of owned cell (i, j, k) in an owned vector
    __host__ __device__ long ownedIndex(int i, int j, int k) const {
        if (this->tiled) return this->storageIndex(i, j, k);
        return (static_cast<long>(k)*this->cells[1] + j)*this->cells[0] + i;
    }
    // owned cells, dense count
    __host__ __device__ long numOwned() const { return static_cast<long>(this->cells[0])*this->cells[1]*this->cells[2]; }
    // entries of an owned vector
    __host__ __device__ long numOwnedVector() const { return this->tiled ? this->numStorage() : this->numOwned(); }
    __host__ __device__ long numStorage() const {
        if (this->tiled) return (this->num_tiles + 1)*this->cellsPerTile();
        return static_cast<long>(this->storage[0])*this->storage[1]*this->storage[2];
    }
    // threads that cover every owned cell, and every stored cell
    __host__ __device__ long numOwnedThreads() const { return this->tiled ? this->emptyCell() : this->numOwned(); }
    __host__ __device__ long numStorageThreads() const { return this->tiled ? this->emptyCell() : this->numStorage(); }
    // owned cell of a thread, with its owned vector entry and storage index; false for threads with no owned cell
    __host__ __device__ bool ownedThread(long thread, int& i, int& j, int& k, long& entry, long& cell) const {
        if (!this->tiled) {
            if (thread >= this->numOwned()) return false;
            this->ownedCell(thread, i, j, k);
            entry = thread;
            cell = this->storageIndex(i, j, k);
            return true;
        }
        if (thread >= this->emptyCell() || !this->storageCell(thread, i, j, k) || !this->isOwned(i, j, k)) return false;
        entry = cell = thread;
        return true;
    }
    // owned cell of an owned vector entry
    __host__ __device__ void entryCell(long entry, int& i, int& j, int& k) const {
        if (this->tiled) {
            this->storageCell(entry, i, j, k);
            return;
        }
        this->ownedCell(entry, i, j, k);
    }

    __host__ __device__ long stride(int direction) const {
        return direction == 0 ? 1 : direction == 1 ? this->storage[0] : static_cast<long>(this->storage[0])*this->storage[1];
    }
    __host__ __device__ real center(int direction, int local_index) const {
        return this->origin[direction] + (this->offset[direction] + local_index + 0.5)*this->spacing[direction];
    }
    __host__ __device__ real cellVolume() const { return this->spacing[0]*this->spacing[1]*this->spacing[2]; }
    // owned cell of a dense compact index, the [z][y][x] order of the output files
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

// the tile maps of a tiled grid, which must outlive every view made from it
class TileSet {
public:
    // tiles of 2^shift cells over the storage box of a grid, keeping the flagged ones, [tile z][y][x]
    GridView build(const GridView& box, const int shift[max_dims], const std::vector<char>& active_tiles);
    long numActive() const { return static_cast<long>(this->host_slot_tiles.size()); }

private:
    std::vector<int> host_tile_slots, host_slot_tiles;
    DeviceArray<int> tile_slots, slot_tiles;
};

// tiles per axis over the storage box for a tile shift
void tileCounts(const GridView& box, const int shift[max_dims], int tiles[max_dims]);

#endif
