// geometric multigrid V-cycles on the fluid cells of a tiled grid, the preconditioner of the implicit systems when the grid is tiled;
// cell centered with averaged restriction, constant prolongation, rediscretized coarse operators and red-black Gauss-Seidel,
// Trottenberg, Oosterlee & Schuller (2001); ghost and solid rows are eliminated after the cycle
#ifndef CTM_MULTIGRID_HPP
#define CTM_MULTIGRID_HPP

#include <memory>
#include <vector>

#include "boundary_conditions.hpp"
#include "geometry.hpp"
#include "linear_solvers.hpp"

// the grids of every level: level 0 is the solver grid, each coarser level merges 2 cells per active axis;
// a coarse cell is an unknown when any of its fine cells is, its other neighbors act as walls
class MultigridHierarchy {
public:
    MultigridHierarchy(const GridView& grid, const signed char* cell_types);
    ~MultigridHierarchy();
    MultigridHierarchy(const MultigridHierarchy&) = delete;
    MultigridHierarchy& operator=(const MultigridHierarchy&) = delete;

    struct Level {
        GridView grid;
        std::unique_ptr<TileSet> tiles;    // null on level 0, which uses the solver's tiles
        DeviceArray<signed char> unknown;  // storage layout, 1 on the cells the cycle solves for
        DeviceArray<real> solution, rhs, residual;
        long num_unknowns = 0;
    };
    std::vector<Level> levels;
    std::vector<long> coarse_cells;  // storage index of every unknown of the coarsest level, solved directly
    DeviceArray<long> device_coarse_cells;
    cublasHandle_t handle;
};

// shift + scale*laplacian on the unknowns of one level, with mirrored values past the domain faces and the walls
struct LevelOperator {
    real shift, scale;
    real inverse_spacing_squared[max_dims];
    real boundary[max_dims][2];  // halo coefficient at the domain faces, +1 zero gradient, -1 zero value
    real wall;                   // the same at faces to cells that are not unknowns
};

class MultigridSolver : public Preconditioner {
public:
    // ghost_sign is the ghost row sign of the operator it preconditions: -1 mirrors the pressure, +1 the no-slip velocity
    MultigridSolver(MultigridHierarchy& hierarchy, const GhostPoints& ghosts, const signed char* cell_types, const HaloRule& rule,
                    real shift, real scale, real ghost_sign);
    void solve(const real* rhs, real* solution) override;  // owned vectors of the solver grid

private:
    void cycle(int level);
    LevelOperator levelOperator(int level) const;

    MultigridHierarchy& hierarchy;
    const GhostPoints& ghosts;
    const signed char* cell_types;
    HaloRule rule;
    real shift, scale, ghost_sign;
    DeviceArray<real> coarse_inverse;  // dense inverse of the coarsest operator, column major
    DeviceArray<real> coarse_rhs, coarse_solution;
};

#endif
