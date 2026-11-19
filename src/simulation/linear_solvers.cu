#include "linear_solvers.hpp"

#include <cmath>
#include <vector>

#include "mpi_proxy.hpp"

#define CUFFT_CHECK(call) CTM_REQUIRE((call) == CUFFT_SUCCESS, "cuFFT call %s failed", #call)
#define CUBLAS_CHECK(call) CTM_REQUIRE((call) == CUBLAS_STATUS_SUCCESS, "cuBLAS call %s failed", #call)

namespace {

// halo coefficient pair -> transform waves: angle pi*(k + wave_shift)*(n + sample_shift)/N
void transformShifts(real coefficient_begin, real coefficient_end, real& wave_shift, real& sample_shift, bool& sine) {
    const bool flat_begin = coefficient_begin > 0, flat_end = coefficient_end > 0;
    sample_shift = 0.5;
    sine = !flat_begin;                                           // a zero value at the begin face needs sines
    wave_shift = flat_begin == flat_end ? (flat_begin ? 0.0 : 1.0) : 0.5;  // DCT-II, DST-II, or the quarter wave type IV
}

// load lines with the pre-twiddle exp(-i pi a n/N) and zero pad to 2N
__global__ void gatherLinesKernel(const real* data, cufftDoubleComplex* work, int length, long inner, long batch, real wave_shift) {
    const long element = blockIdx.x*static_cast<long>(blockDim.x) + threadIdx.x;
    const int padded = 2*length;
    if (element >= batch*padded) return;
    const long line = element/padded;
    const int n = static_cast<int>(element % padded);
    if (n >= length) {
        work[element] = {0.0, 0.0};
        return;
    }
    const real value = data[(line/inner)*length*inner + line % inner + n*inner];
    real sine_part, cosine_part;
    sincospi(-wave_shift*n/length, &sine_part, &cosine_part);
    work[element] = {value*cosine_part, value*sine_part};
}

// post-twiddle exp(-i pi (k + a) b/N), keep the real part for cosines and minus the imaginary part for sines
__global__ void scatterLinesKernel(const cufftDoubleComplex* work, real* data, int length, long inner, long batch, real wave_shift,
                                   real sample_shift, bool sine) {
    const long element = blockIdx.x*static_cast<long>(blockDim.x) + threadIdx.x;
    if (element >= batch*length) return;
    const long line = element/length;
    const int k = static_cast<int>(element % length);
    const cufftDoubleComplex spectrum = work[line*2*length + k];
    real sine_part, cosine_part;
    sincospi(-(k + wave_shift)*sample_shift/length, &sine_part, &cosine_part);
    const real real_part = spectrum.x*cosine_part - spectrum.y*sine_part;
    const real imaginary_part = spectrum.x*sine_part + spectrum.y*cosine_part;
    data[(line/inner)*length*inner + line % inner + static_cast<long>(k)*inner] = sine ? -imaginary_part : real_part;
}

__global__ void gatherHartleyKernel(const real* data, cufftDoubleComplex* work, int length, long inner, long batch) {
    const long element = blockIdx.x*static_cast<long>(blockDim.x) + threadIdx.x;
    if (element >= batch*length) return;
    const long line = element/length;
    const int n = static_cast<int>(element % length);
    work[element] = {data[(line/inner)*length*inner + line % inner + static_cast<long>(n)*inner], 0.0};
}

// Re(F) - Im(F) of the forward FFT is the sum with cos + sin
__global__ void scatterHartleyKernel(const cufftDoubleComplex* work, real* data, int length, long inner, long batch) {
    const long element = blockIdx.x*static_cast<long>(blockDim.x) + threadIdx.x;
    if (element >= batch*length) return;
    const long line = element/length;
    const int k = static_cast<int>(element % length);
    data[(line/inner)*length*inner + line % inner + static_cast<long>(k)*inner] = work[element].x - work[element].y;
}

// pencils [row][axis]: divide by the eigenvalue of shift + scale*laplacian and fold in the basis norms
__global__ void divideEigenvaluesKernel(GridView grid, real* pencils, int local_rows, int row_offset, int axis_cells, real shift, real scale,
                                        const real* eigenvalues_x, const real* eigenvalues_y, const real* eigenvalues_z,
                                        const real* norms_x, const real* norms_y, const real* norms_z) {
    const long element = blockIdx.x*static_cast<long>(blockDim.x) + threadIdx.x;
    if (element >= static_cast<long>(local_rows)*axis_cells) return;
    const long row = row_offset + element/axis_cells;
    const int axis_index = static_cast<int>(element % axis_cells);
    const int x = static_cast<int>(row % grid.global_cells[0]);
    real eigenvalue, norm;
    if (numDims(grid) == 2) {
        eigenvalue = eigenvalues_x[x] + eigenvalues_y[axis_index];
        norm = norms_x[x]*norms_y[axis_index];
    } else {
        const int y = static_cast<int>(row/grid.global_cells[0]);
        eigenvalue = eigenvalues_x[x] + eigenvalues_y[y] + eigenvalues_z[axis_index];
        norm = norms_x[x]*norms_y[y]*norms_z[axis_index];
    }
    const real operator_eigenvalue = shift + scale*eigenvalue;
    pencils[element] *= operator_eigenvalue == 0.0 ? 0.0 : norm/operator_eigenvalue;  // a fully periodic Poisson problem drops its constant mode
}

}  // namespace

LineTransforms::~LineTransforms() {
    for (auto& entry : this->plans) cufftDestroy(entry.second);
}

cufftHandle LineTransforms::plan(int padded_length, long batch) {
    const auto key = std::make_pair(padded_length, batch);
    const auto found = this->plans.find(key);
    if (found != this->plans.end()) return found->second;
    cufftHandle handle;
    int lengths[1] = {padded_length};
    CUFFT_CHECK(cufftPlanMany(&handle, 1, lengths, nullptr, 1, padded_length, nullptr, 1, padded_length, CUFFT_Z2Z, static_cast<int>(batch)));
    this->plans[key] = handle;
    return handle;
}

void LineTransforms::apply(real* data, int length, long outer, long inner, real wave_shift, real sample_shift, bool sine) {
    const long batch = outer*inner;
    if (this->work.size() < batch*2*length) this->work.resize(batch*2*length);
    gatherLinesKernel<<<numBlocks(batch*2*length), block_size>>>(data, this->work.data(), length, inner, batch, wave_shift);
    CUFFT_CHECK(cufftExecZ2Z(this->plan(2*length, batch), this->work.data(), this->work.data(), CUFFT_FORWARD));
    scatterLinesKernel<<<numBlocks(batch*length), block_size>>>(this->work.data(), data, length, inner, batch, wave_shift, sample_shift, sine);
    CUDA_CHECK(cudaGetLastError());
}

void LineTransforms::applyHartley(real* data, int length, long outer, long inner) {
    const long batch = outer*inner;
    if (this->work.size() < batch*length) this->work.resize(batch*length);
    gatherHartleyKernel<<<numBlocks(batch*length), block_size>>>(data, this->work.data(), length, inner, batch);
    CUFFT_CHECK(cufftExecZ2Z(this->plan(length, batch), this->work.data(), this->work.data(), CUFFT_FORWARD));
    scatterHartleyKernel<<<numBlocks(batch*length), block_size>>>(this->work.data(), data, length, inner, batch);
    CUDA_CHECK(cudaGetLastError());
}

void SpectralSolver::transformAxis(real* data, int length, long outer, long inner, int axis, bool inverse) {
    if (this->hartley[axis]) {
        this->transforms.applyHartley(data, length, outer, inner);  // its own inverse up to the norm in the divide
        return;
    }
    const real wave_shift = inverse ? this->sample_shift[axis] : this->wave_shift[axis];
    const real sample_shift = inverse ? this->wave_shift[axis] : this->sample_shift[axis];
    this->transforms.apply(data, length, outer, inner, wave_shift, sample_shift, this->sine[axis]);
}

SpectralSolver::SpectralSolver(const GridView& grid, SlabTranspose& transpose, LineTransforms& transforms, const HaloRule& rule,
                               real shift, real scale)
    : grid(grid), transpose(transpose), transforms(transforms), shift(shift), scale(scale) {
    for (int d = 0; d < max_dims; ++d) {
        const int length = grid.global_cells[d];
        std::vector<real> eigenvalues(length, 0.0), norms(length, 1.0);
        transformShifts(rule.coefficient[d][0], rule.coefficient[d][1], this->wave_shift[d], this->sample_shift[d], this->sine[d]);
        this->hartley[d] = grid.periodic[d];
        if (this->hartley[d]) {
            for (int k = 0; k < length; ++k) {
                const real angle = pi*k/length;
                eigenvalues[k] = -(4.0/(grid.spacing[d]*grid.spacing[d]))*std::sin(angle)*std::sin(angle);
                norms[k] = 1.0/length;
            }
        } else if (d < grid.num_dims) {
            for (int k = 0; k < length; ++k) {
                const real angle = pi*(k + this->wave_shift[d])/(2.0*length);
                eigenvalues[k] = -(4.0/(grid.spacing[d]*grid.spacing[d]))*std::sin(angle)*std::sin(angle);
                const bool is_half_norm = (this->wave_shift[d] == 0.0 && k == 0) || (this->wave_shift[d] == 1.0 && k == length - 1);
                norms[k] = (is_half_norm ? 1.0 : 2.0)/length;
            }
        }
        this->eigenvalues[d].copyFromHost(eigenvalues.data(), length);
        this->norms[d].copyFromHost(norms.data(), length);
    }
    this->compact.resize(grid.numOwned());
    this->pencils.resize(static_cast<long>(transpose.local_rows)*transpose.axis_cells);
}

void SpectralSolver::solve(const real* rhs, real* solution) {
    const GridView& g = this->grid;
    const int axis = g.decomposed_axis;
    CUDA_CHECK(cudaMemcpy(this->compact.data(), rhs, g.numOwned()*sizeof(real), cudaMemcpyDeviceToDevice));

    // forward along the local axes of the compact layout [z][y][x], then along the decomposed axis as pencils
    for (int d = 0; d < axis; ++d) {
        const long inner = d == 0 ? 1 : g.cells[0];
        this->transformAxis(this->compact.data(), g.cells[d], g.numOwned()/(g.cells[d]*inner), inner, d, false);
    }
    this->transpose.toPencils(this->compact.data(), this->pencils.data());
    const int axis_cells = this->transpose.axis_cells;
    this->transformAxis(this->pencils.data(), axis_cells, this->transpose.local_rows, 1, axis, false);

    divideEigenvaluesKernel<<<numBlocks(static_cast<long>(this->transpose.local_rows)*axis_cells), block_size>>>(
        g, this->pencils.data(), this->transpose.local_rows, this->transpose.row_offset, axis_cells, this->shift, this->scale,
        this->eigenvalues[0].data(), this->eigenvalues[1].data(), this->eigenvalues[2].data(), this->norms[0].data(), this->norms[1].data(),
        this->norms[2].data());
    CUDA_CHECK(cudaGetLastError());

    // the inverse swaps the wave and sample shifts, then retraces the same axes in reverse
    this->transformAxis(this->pencils.data(), axis_cells, this->transpose.local_rows, 1, axis, true);
    this->transpose.toSlabs(this->pencils.data(), this->compact.data());
    for (int d = axis - 1; d >= 0; --d) {
        const long inner = d == 0 ? 1 : g.cells[0];
        this->transformAxis(this->compact.data(), g.cells[d], g.numOwned()/(g.cells[d]*inner), inner, d, true);
    }
    CUDA_CHECK(cudaMemcpy(solution, this->compact.data(), g.numOwned()*sizeof(real), cudaMemcpyDeviceToDevice));
}

Gmres::Gmres(long size, int restart, real relative_tolerance, int max_iterations)
    : size(size), restart(restart), max_iterations(max_iterations), relative_tolerance(relative_tolerance) {
    CUBLAS_CHECK(cublasCreate(&this->handle));
    this->basis.resize(size*(restart + 1));
    this->vector.resize(size);
    this->preconditioned.resize(size);
}

Gmres::~Gmres() {
    cublasDestroy(this->handle);
}

real Gmres::globalDot(const real* a, const real* b) {
    real local;
    CUBLAS_CHECK(cublasDdot(this->handle, static_cast<int>(this->size), a, 1, b, 1, &local));
    return allreduceSum(local);
}

int Gmres::solve(const LinearOperator& apply_matrix, const LinearOperator& apply_preconditioner, const real* rhs, real* solution) {
    const int n = static_cast<int>(this->size);
    const real rhs_norm = std::sqrt(this->globalDot(rhs, rhs));
    if (rhs_norm == 0.0) {
        CUDA_CHECK(cudaMemset(solution, 0, this->size*sizeof(real)));
        return 0;
    }
    const real tolerance = this->relative_tolerance*rhs_norm;
    const real one = 1.0, minus_one = -1.0, zero = 0.0;
    std::vector<real> hessenberg((this->restart + 1)*this->restart), projections(this->restart + 1), cosines(this->restart),
        sines(this->restart), residuals(this->restart + 1);
    auto hessian = [&](int row, int column) -> real& { return hessenberg[column*(this->restart + 1) + row]; };

    int iterations = 0;
    while (true) {
        // true residual of the current solution
        apply_matrix(solution, this->vector.data());
        CUDA_CHECK(cudaMemcpy(this->basis.data(), rhs, this->size*sizeof(real), cudaMemcpyDeviceToDevice));
        CUBLAS_CHECK(cublasDaxpy(this->handle, n, &minus_one, this->vector.data(), 1, this->basis.data(), 1));
        const real residual_norm = std::sqrt(this->globalDot(this->basis.data(), this->basis.data()));
        if (residual_norm <= tolerance) return iterations;
        CTM_REQUIRE(iterations < this->max_iterations, "GMRES stalled at relative residual %.3e after %d iterations",
                    residual_norm/rhs_norm, iterations);
        const real inverse_norm = 1.0/residual_norm;
        CUBLAS_CHECK(cublasDscal(this->handle, n, &inverse_norm, this->basis.data(), 1));
        std::fill(hessenberg.begin(), hessenberg.end(), 0.0);
        std::fill(residuals.begin(), residuals.end(), 0.0);
        residuals[0] = residual_norm;

        int columns = 0;
        for (int column = 0; column < this->restart; ++column) {
            real* current = this->basis.data() + column*this->size;
            apply_preconditioner(current, this->preconditioned.data());
            apply_matrix(this->preconditioned.data(), this->vector.data());

            // classical Gram-Schmidt applied twice, the projections are global dot products
            for (int pass = 0; pass < 2; ++pass) {
                CUBLAS_CHECK(cublasDgemv(this->handle, CUBLAS_OP_T, n, column + 1, &one, this->basis.data(), n, this->vector.data(), 1, &zero,
                                         this->preconditioned.data(), 1));
                CUDA_CHECK(cudaMemcpy(projections.data(), this->preconditioned.data(), (column + 1)*sizeof(real), cudaMemcpyDeviceToHost));
                allreduceSum(projections.data(), column + 1);
                CUDA_CHECK(cudaMemcpy(this->preconditioned.data(), projections.data(), (column + 1)*sizeof(real), cudaMemcpyHostToDevice));
                CUBLAS_CHECK(cublasDgemv(this->handle, CUBLAS_OP_N, n, column + 1, &minus_one, this->basis.data(), n, this->preconditioned.data(), 1,
                                         &one, this->vector.data(), 1));
                for (int row = 0; row <= column; ++row) hessian(row, column) += projections[row];
            }
            const real next_norm = std::sqrt(this->globalDot(this->vector.data(), this->vector.data()));
            hessian(column + 1, column) = next_norm;

            // Givens rotations keep the Hessenberg matrix triangular and give the residual norm
            for (int row = 0; row < column; ++row) {
                const real upper = hessian(row, column), lower = hessian(row + 1, column);
                hessian(row, column) = cosines[row]*upper + sines[row]*lower;
                hessian(row + 1, column) = -sines[row]*upper + cosines[row]*lower;
            }
            const real radius = std::hypot(hessian(column, column), hessian(column + 1, column));
            cosines[column] = hessian(column, column)/radius;
            sines[column] = hessian(column + 1, column)/radius;
            hessian(column, column) = radius;
            hessian(column + 1, column) = 0.0;
            residuals[column + 1] = -sines[column]*residuals[column];
            residuals[column] = cosines[column]*residuals[column];

            ++iterations;
            columns = column + 1;
            const bool breakdown = next_norm <= 1e-14*residual_norm;
            if (std::abs(residuals[column + 1]) <= tolerance || breakdown || iterations >= this->max_iterations) break;
            const real inverse_next = 1.0/next_norm;
            CUDA_CHECK(cudaMemcpy(this->basis.data() + (column + 1)*this->size, this->vector.data(), this->size*sizeof(real),
                                  cudaMemcpyDeviceToDevice));
            CUBLAS_CHECK(cublasDscal(this->handle, n, &inverse_next, this->basis.data() + (column + 1)*this->size, 1));
        }

        // back substitution, then solution += M^-1 (V y)
        std::vector<real> coefficients(columns);
        for (int row = columns - 1; row >= 0; --row) {
            real value = residuals[row];
            for (int column = row + 1; column < columns; ++column) value -= hessian(row, column)*coefficients[column];
            coefficients[row] = value/hessian(row, row);
        }
        CUDA_CHECK(cudaMemcpy(this->preconditioned.data(), coefficients.data(), columns*sizeof(real), cudaMemcpyHostToDevice));
        CUBLAS_CHECK(cublasDgemv(this->handle, CUBLAS_OP_N, n, columns, &one, this->basis.data(), n, this->preconditioned.data(), 1, &zero,
                                 this->vector.data(), 1));
        apply_preconditioner(this->vector.data(), this->preconditioned.data());
        CUBLAS_CHECK(cublasDaxpy(this->handle, n, &one, this->preconditioned.data(), 1, solution, 1));
    }
}
