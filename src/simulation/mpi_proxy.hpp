// halo exchange, reductions, and slab to pencil transposes, following MFC's m_mpi_proxy
#ifndef CTM_MPI_PROXY_HPP
#define CTM_MPI_PROXY_HPP

#include <vector>

#include "grid.hpp"

real allreduceSum(real value);
real allreduceMax(real value);
void allreduceSum(real* values, int count);

// swaps the halo planes along the decomposed axis with the neighbor ranks, staged through pinned host memory
class HaloExchange {
public:
    explicit HaloExchange(const GridView& grid);
    ~HaloExchange();
    void exchange(real* field);

private:
    GridView grid;
    long plane_size;  // elements in one storage plane normal to the decomposed axis
    real* send_lower = nullptr;
    real* send_upper = nullptr;
    real* receive_lower = nullptr;
    real* receive_upper = nullptr;
};

// moves a compact owned vector, laid out [slab index][row], to pencils laid out [row][full axis] and back,
// so the decomposed axis can be transformed locally; rows are the flattened non-decomposed indices
class SlabTranspose {
public:
    explicit SlabTranspose(const GridView& grid);
    ~SlabTranspose();
    void toPencils(const real* compact, real* pencils);
    void toSlabs(const real* pencils, real* compact);

    int axis_cells;     // full length of the decomposed axis
    int num_rows;       // rows over all ranks
    int local_rows;     // rows this rank holds as pencils
    int row_offset;     // first row this rank holds

private:
    int num_ranks, rank, slab_cells, slab_offset;
    std::vector<int> send_counts, send_displacements, receive_counts, receive_displacements;
    DeviceArray<real> device_send, device_receive;
    DeviceArray<int> device_slab_offsets, device_slab_counts, device_receive_displacements;
    real* host_send = nullptr;
    real* host_receive = nullptr;
};

#endif
