// spectral preconditioner and right preconditioned restarted GMRES, Saad & Schultz (1986), for the implicit systems
#ifndef CTM_LINEAR_SOLVERS_HPP
#define CTM_LINEAR_SOLVERS_HPP

#include <functional>
#include <map>
#include <memory>

#include <cublas_v2.h>
#include <cufft.h>

#include "boundary_conditions.hpp"

// batched 1D sine or cosine transforms along one axis through a zero padded complex FFT of twice the length
class LineTransforms {
public:
    ~LineTransforms();
    // data[outer][length][inner] <- sum_n data[outer][n][inner]*trig(pi*(k + wave_shift)*(n + sample_shift)/length)
    void apply(real* data, int length, long outer, long inner, real wave_shift, real sample_shift, bool sine);

private:
    cufftHandle plan(int padded_length, long batch);
    std::map<std::pair<int, long>, cufftHandle> plans;
    DeviceArray<cufftDoubleComplex> work;
};

// exact solve of (shift + scale*laplacian) x = b on the uniform grid with domain halos only, the no-body operator
class SpectralSolver {
public:
    SpectralSolver(const GridView& grid, SlabTranspose& transpose, LineTransforms& transforms, const HaloRule& rule, real shift, real scale);
    void solve(const real* rhs, real* solution);

private:
    GridView grid;
    SlabTranspose& transpose;
    LineTransforms& transforms;
    real shift, scale;
    real wave_shift[max_dims], sample_shift[max_dims];
    bool sine[max_dims];
    DeviceArray<real> eigenvalues[max_dims];  // per axis laplacian eigenvalues
    DeviceArray<real> norms[max_dims];        // per axis 1/|basis|^2, folded into the divide
    DeviceArray<real> compact, pencils;
};

using LinearOperator = std::function<void(const real* input, real* output)>;

class Gmres {
public:
    Gmres(long size, int restart, real relative_tolerance, int max_iterations);
    ~Gmres();
    // solution holds the initial guess on entry, returns the iteration count
    int solve(const LinearOperator& apply_matrix, const LinearOperator& apply_preconditioner, const real* rhs, real* solution);

private:
    real globalDot(const real* a, const real* b);

    long size;
    int restart, max_iterations;
    real relative_tolerance;
    cublasHandle_t handle;
    DeviceArray<real> basis, vector, preconditioned;
};

#endif
