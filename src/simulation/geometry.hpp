// STL immersed boundaries: reading, solid markers, ghost points, levelsets, and image point weights
#ifndef CTM_GEOMETRY_HPP
#define CTM_GEOMETRY_HPP

#include "grid.hpp"

constexpr int max_stencil = 8;  // 2x2x2 image point stencil, 2x2 in 2D with the rest weighted zero

// cell types on owned cells
enum CellType : signed char { cell_fluid = 0, cell_ghost = 1, cell_interior = 2 };

// struct of arrays over this rank's ghost points
struct GhostPoints {
    int count = 0;
    DeviceArray<long> storage_cells;  // ghost cell in fields with halos
    DeviceArray<long> owned_cells;    // ghost cell in compact owned vectors
    DeviceArray<int> patch_ids;       // immersed boundary patch, from 1
    DeviceArray<long> stencil_cells;  // [ghost][max_stencil] storage cells interpolated to the image point
    DeviceArray<real> weights;        // [ghost][max_stencil] interpolation weights, zero on solid cells
};

class Geometry {
public:
    Geometry(const Parameters& parameters, const GridView& grid);

    int num_patches = 0;
    DeviceArray<int> markers;              // storage sized, 0 in fluid and the patch id inside a body
    DeviceArray<signed char> cell_types;   // owned compact, CellType
    GhostPoints ghost_points;
    long global_ghost_points = 0;
    long global_bilinear = 0;              // ghost points whose stencil was all fluid
};

#endif
