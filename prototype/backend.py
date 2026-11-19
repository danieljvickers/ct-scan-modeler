"""Array backend: cupy on the GPU when MFC_CUPY is set, numpy otherwise, matching mfpython."""

import os

import numpy
import scipy.sparse

USING_GPU = False
import numpy as xp
import scipy.fft as fft
import scipy.sparse as sparse
import scipy.sparse.linalg as sparse_linalg

if os.environ.get("MFC_CUPY", "").strip().lower() in ("1", "true", "yes", "on"):
    try:
        import cupy
        import cupyx.scipy.fft
        import cupyx.scipy.sparse
        import cupyx.scipy.sparse.linalg
        cupy.cuda.runtime.getDeviceCount()  # raises without a usable GPU
        xp, fft, sparse, sparse_linalg = cupy, cupyx.scipy.fft, cupyx.scipy.sparse, cupyx.scipy.sparse.linalg
        USING_GPU = True
    except Exception as error:
        print(f"WARN :: MFC_CUPY is set but cupy is unavailable ({type(error).__name__}: {error}); falling back to numpy")


def to_device(host_array):
    return xp.asarray(host_array)


def to_host(array):
    """Copy a backend array to a numpy array; numpy input passes through."""
    return array.get() if USING_GPU and isinstance(array, xp.ndarray) else numpy.asarray(array)


def sparse_to_device(host_matrix):
    """Move a SciPy sparse matrix to the backend as CSR."""
    host_matrix = scipy.sparse.csr_matrix(host_matrix)
    return sparse.csr_matrix(host_matrix) if USING_GPU else host_matrix
