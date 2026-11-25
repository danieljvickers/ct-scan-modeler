#include "multigrid.hpp"

#include <algorithm>
#include <cmath>
#include <unordered_map>

#define CUBLAS_CHECK(call) CTM_REQUIRE((call) == CUBLAS_STATUS_SUCCESS, "cuBLAS call %s failed", #call)

namespace {

constexpr int pre_sweeps = 2;                 // red-black Gauss-Seidel sweeps before and after each coarse correction
constexpr int post_sweeps = 2;
constexpr long max_coarse_unknowns = 512;     // the coarsest level is solved with a dense inverse once it is this small
constexpr int max_levels = 20;

__device__ inline long levelThread() { return blockIdx.x*static_cast<long>(blockDim.x) + threadIdx.x; }

// an owned unknown of a level and its index
__device__ inline bool unknownCell(const GridView& grid, const signed char* unknown, long cell, int index[max_dims]) {
    return cell < grid.numStorageThreads() && grid.storageCell(cell, index[0], index[1], index[2]) && grid.isOwned(index[0], index[1], index[2]) &&
           unknown[cell] != 0;
}

// one unknown's row: its diagonal and the weighted sum of its unknown neighbors
__device__ inline void rowTerms(const GridView& grid, const LevelOperator& op, const signed char* unknown, const real* x, const int index[max_dims],
                                real& diagonal, real& neighbors) {
    diagonal = op.shift;
    neighbors = 0.0;
    for (int d = 0; d < numDims(grid); ++d)
        for (int side = 0; side < 2; ++side) {
            int neighbor[max_dims] = {index[0], index[1], index[2]};
            neighbor[d] += side == 0 ? -1 : 1;
            const real weight = op.scale*op.inverse_spacing_squared[d];
            if (neighbor[d] < 0 || neighbor[d] >= grid.cells[d]) {
                if (!grid.periodic[d]) {
                    diagonal += weight*(op.boundary[d][side] - 1.0);
                    continue;
                }
                neighbor[d] = (neighbor[d] + grid.cells[d]) % grid.cells[d];
            }
            const long cell = grid.storageIndex(neighbor[0], neighbor[1], neighbor[2]);
            if (unknown[cell] != 0) {
                neighbors += weight*x[cell];
                diagonal -= weight;
            } else {
                diagonal += weight*(op.wall - 1.0);
            }
        }
}

__global__ void smoothKernel(GridView grid, LevelOperator op, const signed char* unknown, const real* rhs, real* x, int color) {
    const long cell = levelThread();
    int index[max_dims];
    if (!unknownCell(grid, unknown, cell, index) || ((index[0] + index[1] + index[2]) & 1) != color) return;
    real diagonal, neighbors;
    rowTerms(grid, op, unknown, x, index, diagonal, neighbors);
    x[cell] = (rhs[cell] - neighbors)/diagonal;
}

__global__ void residualKernel(GridView grid, LevelOperator op, const signed char* unknown, const real* rhs, const real* x, real* residual) {
    const long cell = levelThread();
    if (cell >= grid.numStorageThreads()) return;
    int index[max_dims];
    if (!unknownCell(grid, unknown, cell, index)) {
        residual[cell] = 0.0;
        return;
    }
    real diagonal, neighbors;
    rowTerms(grid, op, unknown, x, index, diagonal, neighbors);
    residual[cell] = rhs[cell] - diagonal*x[cell] - neighbors;
}

// coarse right hand side: the average residual of the fine cells it covers, and a zero first guess
__global__ void restrictKernel(GridView fine, GridView coarse, const real* fine_residual, const signed char* coarse_unknown, real* coarse_rhs,
                               real* coarse_solution, real weight) {
    const long cell = levelThread();
    if (cell >= coarse.numStorageThreads()) return;
    coarse_solution[cell] = 0.0;
    int index[max_dims];
    if (!unknownCell(coarse, coarse_unknown, cell, index)) {
        coarse_rhs[cell] = 0.0;
        return;
    }
    real sum = 0.0;
    const int children_z = numDims(fine) == 3 ? 2 : 1;
    for (int dz = 0; dz < children_z; ++dz)
        for (int dy = 0; dy < 2; ++dy)
            for (int dx = 0; dx < 2; ++dx) {
                const int child[max_dims] = {2*index[0] + dx, 2*index[1] + dy, numDims(fine) == 3 ? 2*index[2] + dz : index[2]};
                if (fine.isOwned(child[0], child[1], child[2])) sum += fine_residual[fine.storageIndex(child[0], child[1], child[2])];
            }
    coarse_rhs[cell] = weight*sum;
}

// fine unknowns take the correction of the coarse cell they sit in
__global__ void prolongKernel(GridView fine, GridView coarse, const signed char* fine_unknown, const real* coarse_solution, real* fine_solution) {
    const long cell = levelThread();
    int index[max_dims];
    if (!unknownCell(fine, fine_unknown, cell, index)) return;
    const int parent_z = numDims(fine) == 3 ? index[2] >> 1 : index[2];
    fine_solution[cell] += coarse_solution[coarse.storageIndex(index[0] >> 1, index[1] >> 1, parent_z)];
}

__global__ void gatherCoarseKernel(long count, const long* cells, const real* field, real* compact) {
    const long entry = levelThread();
    if (entry < count) compact[entry] = field[cells[entry]];
}

__global__ void scatterCoarseKernel(long count, const long* cells, const real* compact, real* field) {
    const long entry = levelThread();
    if (entry < count) field[cells[entry]] = compact[entry];
}

// level 0 right hand side on the fluid cells, from a solver owned vector, which shares the storage layout of a tiled grid
__global__ void prepareKernel(GridView grid, const signed char* unknown, const real* rhs, real* level_rhs, real* level_solution) {
    const long cell = levelThread();
    if (cell >= grid.numStorageThreads()) return;
    int index[max_dims];
    level_rhs[cell] = unknownCell(grid, unknown, cell, index) ? rhs[cell] : 0.0;
    level_solution[cell] = 0.0;
}

// diagonal of shift + scale*laplacian on the solver grid, with the halo coefficient replacing a missing neighbor at a domain face
__device__ inline real helmholtzDiagonal(const GridView& grid, const LevelOperator& op, const int index[max_dims]) {
    real laplacian_diagonal = 0.0;
    for (int d = 0; d < numDims(grid); ++d) {
        real diagonal = -2.0;
        if (!grid.periodic[d] && index[d] == 0) diagonal += op.boundary[d][0];
        if (!grid.periodic[d] && index[d] == grid.cells[d] - 1) diagonal += op.boundary[d][1];
        laplacian_diagonal += diagonal*op.inverse_spacing_squared[d];
    }
    return op.shift + op.scale*laplacian_diagonal;
}

// fluid cells take the cycle's answer, solid interior rows are the identity on a tiled grid, everything outside the owned cells zero
__global__ void finishKernel(GridView grid, LevelOperator op, const signed char* unknown, const signed char* cell_types, const real* rhs,
                             const real* level_solution, real* solution) {
    const long cell = levelThread();
    if (cell >= grid.numStorageThreads()) return;
    int index[max_dims];
    if (!grid.storageCell(cell, index[0], index[1], index[2]) || !grid.isOwned(index[0], index[1], index[2])) {
        solution[cell] = 0.0;
        return;
    }
    if (unknown[cell] != 0) {
        solution[cell] = level_solution[cell];
        return;
    }
    solution[cell] = cell_types[cell] == cell_interior ? rhs[cell] : 0.0;
}

// a ghost row diagonal*(x_ghost + ghost_sign*x_image) = rhs solved for the ghost value with the image interpolated from the fluid answer
__global__ void ghostKernel(GridView grid, LevelOperator op, int num_ghost_points, const long* owned_cells, const long* stencil_cells,
                            const real* weights, const real* rhs, real ghost_sign, real* solution) {
    const int ghost = blockIdx.x*blockDim.x + threadIdx.x;
    if (ghost >= num_ghost_points) return;
    const long entry = owned_cells[ghost];
    int index[max_dims];
    grid.entryCell(entry, index[0], index[1], index[2]);
    real image_value = 0.0;
    for (int corner = 0; corner < max_stencil; ++corner)
        image_value += weights[ghost*max_stencil + corner]*solution[stencil_cells[ghost*max_stencil + corner]];
    solution[entry] = rhs[entry]/helmholtzDiagonal(grid, op, index) - ghost_sign*image_value;
}

__global__ void unknownFromCellTypesKernel(GridView grid, const signed char* cell_types, signed char* unknown, int* count) {
    int i, j, k;
    long entry, cell;
    if (!grid.ownedThread(levelThread(), i, j, k, entry, cell)) return;
    unknown[cell] = cell_types[entry] == cell_fluid ? 1 : 0;
    if (unknown[cell]) atomicAdd(count, 1);
}

// flag the coarse parent of every fine unknown in a dense coarse mask [z][y][x]
__global__ void markParentsKernel(GridView fine, const signed char* fine_unknown, int coarse_x, int coarse_y, signed char* mask) {
    const long cell = levelThread();
    int index[max_dims];
    if (!unknownCell(fine, fine_unknown, cell, index)) return;
    const int parent_z = numDims(fine) == 3 ? index[2] >> 1 : index[2];
    mask[(static_cast<long>(parent_z)*coarse_y + (index[1] >> 1))*coarse_x + (index[0] >> 1)] = 1;
}

__global__ void unknownFromMaskKernel(GridView grid, const signed char* mask, signed char* unknown) {
    const long cell = levelThread();
    int index[max_dims];
    if (cell >= grid.numStorageThreads() || !grid.storageCell(cell, index[0], index[1], index[2]) || !grid.isOwned(index[0], index[1], index[2]))
        return;
    unknown[cell] = mask[(static_cast<long>(index[2])*grid.cells[1] + index[1])*grid.cells[0] + index[0]];
}

void allocateLevel(MultigridHierarchy::Level& level) {
    const long storage = level.grid.numStorage();
    level.solution.resize(storage);
    level.rhs.resize(storage);
    level.residual.resize(storage);
}

// Gauss-Jordan with partial pivoting, in place on a row major matrix
void invertDense(std::vector<real>& matrix, int n) {
    std::vector<real> inverse(static_cast<size_t>(n)*n, 0.0);
    for (int i = 0; i < n; ++i) inverse[static_cast<size_t>(i)*n + i] = 1.0;
    real largest = 0.0;
    for (real value : matrix) largest = std::max(largest, std::abs(value));
    for (int column = 0; column < n; ++column) {
        int pivot = column;
        for (int row = column + 1; row < n; ++row)
            if (std::abs(matrix[static_cast<size_t>(row)*n + column]) > std::abs(matrix[static_cast<size_t>(pivot)*n + column])) pivot = row;
        CTM_REQUIRE(std::abs(matrix[static_cast<size_t>(pivot)*n + column]) > 1e-13*largest, "the coarsest multigrid operator is singular");
        if (pivot != column)
            for (int c = 0; c < n; ++c) {
                std::swap(matrix[static_cast<size_t>(pivot)*n + c], matrix[static_cast<size_t>(column)*n + c]);
                std::swap(inverse[static_cast<size_t>(pivot)*n + c], inverse[static_cast<size_t>(column)*n + c]);
            }
        const real scale = 1.0/matrix[static_cast<size_t>(column)*n + column];
        for (int c = 0; c < n; ++c) {
            matrix[static_cast<size_t>(column)*n + c] *= scale;
            inverse[static_cast<size_t>(column)*n + c] *= scale;
        }
        for (int row = 0; row < n; ++row) {
            const real factor = matrix[static_cast<size_t>(row)*n + column];
            if (row == column || factor == 0.0) continue;
            for (int c = 0; c < n; ++c) {
                matrix[static_cast<size_t>(row)*n + c] -= factor*matrix[static_cast<size_t>(column)*n + c];
                inverse[static_cast<size_t>(row)*n + c] -= factor*inverse[static_cast<size_t>(column)*n + c];
            }
        }
    }
    matrix.swap(inverse);
}

}  // namespace

MultigridHierarchy::MultigridHierarchy(const GridView& grid, const signed char* cell_types) {
    CTM_REQUIRE(grid.tiled, "multigrid needs a tiled grid");
    CUBLAS_CHECK(cublasCreate(&this->handle));
    this->levels.emplace_back();
    Level& first = this->levels.back();
    first.grid = grid;
    first.unknown.resize(grid.numStorage());
    DeviceArray<int> count(1);
    unknownFromCellTypesKernel<<<numBlocks(grid.numOwnedThreads()), block_size>>>(grid, cell_types, first.unknown.data(), count.data());
    CUDA_CHECK(cudaGetLastError());
    int host_count;
    count.copyToHost(&host_count);
    first.num_unknowns = host_count;
    allocateLevel(first);

    while (static_cast<int>(this->levels.size()) < max_levels && this->levels.back().num_unknowns > max_coarse_unknowns) {
        const Level& fine = this->levels.back();
        GridView box = fine.grid;
        box.tiled = false;
        bool coarsens = false;
        for (int d = 0; d < max_dims; ++d) {
            if (d < fine.grid.num_dims) {
                box.cells[d] = (fine.grid.cells[d] + 1)/2;
                box.spacing[d] = 2.0*fine.grid.spacing[d];
                coarsens |= box.cells[d] < fine.grid.cells[d];
            }
            box.halo[d] = 0;
            box.offset[d] = 0;
            box.global_cells[d] = box.cells[d];
            box.storage[d] = box.cells[d];
        }
        if (!coarsens) break;

        // dense mask of the coarse unknowns, then the tiles that hold any
        const long dense_cells = static_cast<long>(box.cells[0])*box.cells[1]*box.cells[2];
        DeviceArray<signed char> mask(dense_cells);
        markParentsKernel<<<numBlocks(fine.grid.numStorageThreads()), block_size>>>(fine.grid, fine.unknown.data(), box.cells[0], box.cells[1],
                                                                                     mask.data());
        CUDA_CHECK(cudaGetLastError());
        std::vector<signed char> host_mask(dense_cells);
        mask.copyToHost(host_mask.data());
        int tiles[max_dims];
        tileCounts(box, fine.grid.tile_shift, tiles);
        std::vector<char> active(static_cast<size_t>(tiles[0])*tiles[1]*tiles[2], 0);
        long num_unknowns = 0;
        for (long cell = 0; cell < dense_cells; ++cell) {
            if (!host_mask[cell]) continue;
            ++num_unknowns;
            const long x = cell % box.cells[0], y = (cell/box.cells[0]) % box.cells[1], z = cell/(static_cast<long>(box.cells[0])*box.cells[1]);
            active[((z >> fine.grid.tile_shift[2])*tiles[1] + (y >> fine.grid.tile_shift[1]))*tiles[0] + (x >> fine.grid.tile_shift[0])] = 1;
        }
        if (num_unknowns >= fine.num_unknowns) break;

        this->levels.emplace_back();
        Level& coarse = this->levels.back();
        coarse.tiles = std::make_unique<TileSet>();
        coarse.grid = coarse.tiles->build(box, this->levels[this->levels.size() - 2].grid.tile_shift, active);
        coarse.unknown.resize(coarse.grid.numStorage());
        unknownFromMaskKernel<<<numBlocks(coarse.grid.numStorageThreads()), block_size>>>(coarse.grid, mask.data(), coarse.unknown.data());
        CUDA_CHECK(cudaGetLastError());
        coarse.num_unknowns = num_unknowns;
        allocateLevel(coarse);
    }

    // the coarsest unknowns, in storage order
    const Level& last = this->levels.back();
    std::vector<signed char> host_unknown(last.grid.numStorage());
    last.unknown.copyToHost(host_unknown.data());
    for (long cell = 0; cell < last.grid.numStorageThreads(); ++cell) {
        int i, j, k;
        if (host_unknown[cell] && last.grid.storageCell(cell, i, j, k) && last.grid.isOwned(i, j, k)) this->coarse_cells.push_back(cell);
    }
    CTM_REQUIRE(static_cast<long>(this->coarse_cells.size()) <= 8*max_coarse_unknowns,
                "multigrid stopped coarsening at %zu unknowns, too many for the dense coarse solve", this->coarse_cells.size());
    this->device_coarse_cells.copyFromHost(this->coarse_cells.data(), static_cast<long>(this->coarse_cells.size()));
    if (mpiRank() == 0) {
        std::printf("INFO :: multigrid levels, unknowns:");
        for (const Level& level : this->levels) std::printf(" %ld", level.num_unknowns);
        std::printf("\n");
    }
}

MultigridHierarchy::~MultigridHierarchy() {
    cublasDestroy(this->handle);
}

LevelOperator MultigridSolver::levelOperator(int level) const {
    const GridView& grid = this->hierarchy.levels[level].grid;
    LevelOperator op;
    op.shift = this->shift;
    op.scale = this->scale;
    for (int d = 0; d < max_dims; ++d) {
        op.inverse_spacing_squared[d] = d < grid.num_dims ? 1.0/(grid.spacing[d]*grid.spacing[d]) : 0.0;
        op.boundary[d][0] = this->rule.coefficient[d][0];
        op.boundary[d][1] = this->rule.coefficient[d][1];
    }
    op.wall = -this->ghost_sign;
    return op;
}

MultigridSolver::MultigridSolver(MultigridHierarchy& hierarchy, const GhostPoints& ghosts, const signed char* cell_types, const HaloRule& rule,
                                 real shift, real scale, real ghost_sign)
    : hierarchy(hierarchy), ghosts(ghosts), cell_types(cell_types), rule(rule), shift(shift), scale(scale), ghost_sign(ghost_sign) {
    // dense operator of the coarsest level, rows in the order of coarse_cells
    const MultigridHierarchy::Level& last = hierarchy.levels.back();
    const GridView& grid = last.grid;
    const LevelOperator op = this->levelOperator(static_cast<int>(hierarchy.levels.size()) - 1);
    const int n = static_cast<int>(hierarchy.coarse_cells.size());
    std::unordered_map<long, int> row_of;
    for (int row = 0; row < n; ++row) row_of[hierarchy.coarse_cells[row]] = row;
    std::vector<real> matrix(static_cast<size_t>(n)*n, 0.0);
    bool has_value_boundary = op.shift != 0.0 || op.wall < 0.0;
    for (int row = 0; row < n; ++row) {
        int index[max_dims];
        grid.storageCell(hierarchy.coarse_cells[row], index[0], index[1], index[2]);
        real diagonal = op.shift;
        for (int d = 0; d < grid.num_dims; ++d)
            for (int side = 0; side < 2; ++side) {
                int neighbor[max_dims] = {index[0], index[1], index[2]};
                neighbor[d] += side == 0 ? -1 : 1;
                const real weight = op.scale*op.inverse_spacing_squared[d];
                if (neighbor[d] < 0 || neighbor[d] >= grid.cells[d]) {
                    if (!grid.periodic[d]) {
                        diagonal += weight*(op.boundary[d][side] - 1.0);
                        has_value_boundary |= op.boundary[d][side] < 0.0;
                        continue;
                    }
                    neighbor[d] = (neighbor[d] + grid.cells[d]) % grid.cells[d];
                }
                const auto found = row_of.find(grid.storageIndex(neighbor[0], neighbor[1], neighbor[2]));
                if (found != row_of.end()) {
                    matrix[static_cast<size_t>(row)*n + found->second] += weight;
                    diagonal -= weight;
                } else {
                    diagonal += weight*(op.wall - 1.0);
                }
            }
        matrix[static_cast<size_t>(row)*n + row] += diagonal;
    }
    // a pure Neumann problem fixes its constant through a rank one term, which leaves zero mean answers alone
    if (!has_value_boundary) {
        real mean_diagonal = 0.0;
        for (int row = 0; row < n; ++row) mean_diagonal += std::abs(matrix[static_cast<size_t>(row)*n + row])/n;
        for (real& value : matrix) value += mean_diagonal/n;
    }
    invertDense(matrix, n);
    std::vector<real> column_major(static_cast<size_t>(n)*n);
    for (int row = 0; row < n; ++row)
        for (int column = 0; column < n; ++column) column_major[static_cast<size_t>(column)*n + row] = matrix[static_cast<size_t>(row)*n + column];
    this->coarse_inverse.copyFromHost(column_major.data(), static_cast<long>(column_major.size()));
    this->coarse_rhs.resize(n);
    this->coarse_solution.resize(n);
}

void MultigridSolver::cycle(int level) {
    MultigridHierarchy::Level& current = this->hierarchy.levels[level];
    const GridView& grid = current.grid;
    const LevelOperator op = this->levelOperator(level);
    const int blocks = numBlocks(grid.numStorageThreads());
    const bool is_coarsest = level == static_cast<int>(this->hierarchy.levels.size()) - 1;
    if (is_coarsest) {
        const long n = static_cast<long>(this->hierarchy.coarse_cells.size());
        const real one = 1.0, zero = 0.0;
        gatherCoarseKernel<<<numBlocks(n), block_size>>>(n, this->hierarchy.device_coarse_cells.data(), current.rhs.data(), this->coarse_rhs.data());
        CUBLAS_CHECK(cublasDgemv(this->hierarchy.handle, CUBLAS_OP_N, static_cast<int>(n), static_cast<int>(n), &one, this->coarse_inverse.data(),
                                 static_cast<int>(n), this->coarse_rhs.data(), 1, &zero, this->coarse_solution.data(), 1));
        scatterCoarseKernel<<<numBlocks(n), block_size>>>(n, this->hierarchy.device_coarse_cells.data(), this->coarse_solution.data(),
                                                          current.solution.data());
        CUDA_CHECK(cudaGetLastError());
        return;
    }
    for (int sweep = 0; sweep < pre_sweeps; ++sweep)
        for (int color = 0; color < 2; ++color)
            smoothKernel<<<blocks, block_size>>>(grid, op, current.unknown.data(), current.rhs.data(), current.solution.data(), color);
    residualKernel<<<blocks, block_size>>>(grid, op, current.unknown.data(), current.rhs.data(), current.solution.data(), current.residual.data());
    MultigridHierarchy::Level& coarse = this->hierarchy.levels[level + 1];
    const real weight = 1.0/(grid.num_dims == 3 ? 8.0 : 4.0);
    restrictKernel<<<numBlocks(coarse.grid.numStorageThreads()), block_size>>>(grid, coarse.grid, current.residual.data(), coarse.unknown.data(),
                                                                               coarse.rhs.data(), coarse.solution.data(), weight);
    CUDA_CHECK(cudaGetLastError());
    this->cycle(level + 1);
    prolongKernel<<<blocks, block_size>>>(grid, coarse.grid, current.unknown.data(), coarse.solution.data(), current.solution.data());
    for (int sweep = 0; sweep < post_sweeps; ++sweep)
        for (int color = 1; color >= 0; --color)
            smoothKernel<<<blocks, block_size>>>(grid, op, current.unknown.data(), current.rhs.data(), current.solution.data(), color);
    CUDA_CHECK(cudaGetLastError());
}

void MultigridSolver::solve(const real* rhs, real* solution) {
    MultigridHierarchy::Level& first = this->hierarchy.levels[0];
    const GridView& grid = first.grid;
    const LevelOperator op = this->levelOperator(0);
    const int blocks = numBlocks(grid.numStorageThreads());
    prepareKernel<<<blocks, block_size>>>(grid, first.unknown.data(), rhs, first.rhs.data(), first.solution.data());
    CUDA_CHECK(cudaGetLastError());
    this->cycle(0);
    finishKernel<<<blocks, block_size>>>(grid, op, first.unknown.data(), this->cell_types, rhs, first.solution.data(), solution);
    if (this->ghosts.count > 0)
        ghostKernel<<<numBlocks(this->ghosts.count), block_size>>>(grid, op, this->ghosts.count, this->ghosts.owned_cells.data(),
                                                                    this->ghosts.stencil_cells.data(), this->ghosts.weights.data(), rhs,
                                                                    this->ghost_sign, solution);
    CUDA_CHECK(cudaGetLastError());
}
