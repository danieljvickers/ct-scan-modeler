// spectral preconditioner and right preconditioned restarted GMRES, Saad & Schultz (1986), for the implicit systems
#ifndef CTM_LINEAR_SOLVERS_HPP
#define CTM_LINEAR_SOLVERS_HPP

#include <functional>
#include <map>
#include <memory>
#include <tuple>

#include <cublas_v2.h>
#include <cufft.h>

#include "boundary_conditions.hpp"

// batched 1D sine and cosine transforms along one axis of data[outer][length][inner], unnormalized
class LineTransforms {
public:
    ~LineTransforms();
    // general form through a zero padded complex FFT of twice the length, the fallback for odd type IV lengths:
    // data <- sum_n data[n]*trig(pi*(k + wave_shift)*(n + sample_shift)/length)
    void apply(real* data, int length, long outer, long inner, real wave_shift, real sample_shift, bool sine);
    // DCT-II or DST-II, and their transposes (DCT-III, DST-III) when inverse, through one real FFT of the same length, Makhoul (1980)
    void applyType2(real* data, int length, long outer, long inner, bool sine, bool inverse);
    // DCT-IV or DST-IV, their own transposes, through a complex FFT of half the length; the length must be even
    void applyType4(real* data, int length, long outer, long inner, bool sine);
    // discrete Hartley transform, sum_n data[n]*(cos + sin)(2 pi k n/length), real and diagonalizing on a periodic axis
    void applyHartley(real* data, int length, long outer, long inner);

private:
    cufftHandle plan(cufftType type, int length, long batch);
    std::map<std::tuple<int, int, long>, cufftHandle> plans;
    DeviceArray<cufftDoubleComplex> work;
    DeviceArray<real> real_work;
};

// approximate inverse applied to owned vectors, the right preconditioner of GMRES
class Preconditioner {
public:
    virtual ~Preconditioner() = default;
    virtual void solve(const real* rhs, real* solution) = 0;
};

// exact solve of (shift + scale*laplacian) x = b on the uniform grid with domain halos only, the no-body operator
class SpectralSolver : public Preconditioner {
public:
    SpectralSolver(const GridView& grid, SlabTranspose& transpose, LineTransforms& transforms, const HaloRule& rule, real shift, real scale);
    void solve(const real* rhs, real* solution) override;

private:
    void transformAxis(real* data, int length, long outer, long inner, int axis, bool inverse);
    bool is_single_rank;  // every axis is local, so the decomposed axis transforms in place without the transpose

    GridView grid;
    SlabTranspose& transpose;
    LineTransforms& transforms;
    real shift, scale;
    real wave_shift[max_dims], sample_shift[max_dims];
    bool sine[max_dims];
    bool hartley[max_dims];  // periodic axes
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
