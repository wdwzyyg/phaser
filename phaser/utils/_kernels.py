
import typing as t

import cupy
import numpy

from .num import split_array

# grid
#  block
#    thread


# implementation 1, just bad
# blockIdx: object row
# threadIdx: object col
# loops over cutout #s
"""
get_cutouts_float32_kernel = cupy.RawKernel(r\"""
extern "C" __global__
void get_cutouts_float32(const float *obj, float *cutouts, const int *start_i, const int *start_j,
                         int n_cutouts, int cutout_shape_y, int cutout_shape_x) {
    size_t i = blockIdx.x;
    size_t j = threadIdx.x;
    size_t obj_idx = blockDim.x * blockIdx.x + threadIdx.x;

    for (size_t cutout_i = 0; cutout_i < n_cutouts; cutout_i++) {
        ptrdiff_t i_rel = i - start_i[cutout_i];
        ptrdiff_t j_rel = j - start_j[cutout_i];
        if (i_rel >= 0 && i_rel < cutout_shape_y && j_rel >= 0 && j_rel < cutout_shape_x) {
            int cutout_idx = cutout_shape_y * cutout_shape_x * cutout_i + cutout_shape_y * j_rel + i_rel;
            cutouts[cutout_idx] = obj[obj_idx];
        }
    }
}
\""", 'get_cutouts_float32')

def get_cutouts(obj: cupy.ndarray, start_idxs: cupy.ndarray, cutout_shape: t.Tuple[int, int]) -> cupy.ndarray:
    assert obj.ndim == 2  # for now
    start_i, start_j = split_array(start_idxs, axis=-1)
    cutouts = cupy.empty((*start_i.shape, *cutout_shape), dtype=obj.dtype)
    n_cutouts = numpy.prod(start_i.shape)

    grid = (obj.shape[0],)
    block = (obj.shape[1],)
    get_cutouts_float32_kernel(grid, block, (obj, cutouts, start_i, start_j, n_cutouts, cutout_shape[0], cutout_shape[1]))
    return cutouts
"""

get_cutouts_float32_kernel = cupy.RawKernel(r"""
extern "C" __global__
void get_cutouts_float32(const float *obj, float *cutouts, const int *start_idxs, int n_cutouts,
                         int obj_shape_y, int obj_shape_x, int cutout_shape_y, int cutout_shape_x) {
    size_t cutout_num = threadIdx.x + blockIdx.x * blockDim.x;
    size_t          i = threadIdx.y + blockIdx.y * blockDim.y;
    size_t          j = threadIdx.z + blockIdx.z * blockDim.z;

    if (cutout_num >= n_cutouts || i >= cutout_shape_y || j >= cutout_shape_x) {
        return;
    }

    ptrdiff_t obj_i = i + start_idxs[2*cutout_num];
    ptrdiff_t obj_j = j + start_idxs[2*cutout_num + 1];

    if (obj_i < 0 || obj_i >= obj_shape_y || obj_j < 0 || obj_j >= obj_shape_x) {
        return;
    }

    int cutout_idx = cutout_shape_y * cutout_shape_x * cutout_num + cutout_shape_x * i + j;
    int obj_idx = obj_shape_x * obj_i + obj_j;
    cutouts[cutout_idx] = obj[obj_idx];
}
""", 'get_cutouts_float32')

def get_cutouts(obj: cupy.ndarray, start_idxs: cupy.ndarray, cutout_shape: t.Tuple[int, int]) -> cupy.ndarray:
    assert obj.ndim == 2  # for now
    cutouts = cupy.empty((*start_idxs.shape[:-1], *cutout_shape), dtype=obj.dtype)
    n_cutouts = numpy.prod(start_idxs.shape[:-1])

    grid = (4, 16, 16)
    shape = (n_cutouts, *cutouts.shape[:-2])
    block = tuple((s + g - 1) // g for (s, g) in zip(shape, grid))
    get_cutouts_float32_kernel(grid, block, (obj, cutouts, start_idxs, n_cutouts, obj.shape[-2], obj.shape[-1], cutout_shape[-2], cutout_shape[-1]))
    return cutouts


set_cutouts_float32_kernel = cupy.RawKernel(r"""
extern "C" __global__
void set_cutouts_float32(const float *obj, float *cutouts, const int *start_idxs, int n_cutouts,
                         int obj_shape_y, int obj_shape_x, int cutout_shape_y, int cutout_shape_x) {
    size_t cutout_num = blockIdx.x * blockDim.x + threadIdx.x;
    size_t          i = blockIdx.y * blockDim.y + threadIdx.y;
    size_t          j = blockIdx.z * blockDim.z + threadIdx.z;

    if (cutout_num >= n_cutouts || i >= cutout_shape_y || j >= cutout_shape_x) {
        return;
    }

    ptrdiff_t obj_i = i + start_idxs[2*cutout_num];
    ptrdiff_t obj_j = j + start_idxs[2*cutout_num + 1];

    if (obj_i < 0 || obj_i >= obj_shape_y || obj_j < 0 || obj_j >= obj_shape_x) {
        return;
    }

    int cutout_idx = cutout_shape_y * cutout_shape_x * cutout_num + cutout_shape_x * i + j;
    int obj_idx = obj_shape_x * obj_i + obj_j;
    obj[obj_idx] = cutouts[cutout_idx];
}
""", 'set_cutouts_float32')

def set_cutouts(obj: cupy.ndarray, cutouts: cupy.ndarray, start_idxs: cupy.ndarray):
    assert obj.ndim == 2  # for now
    assert obj.dtype == cutouts.dtype
    assert start_idxs.shape == (*cutouts.shape[:-2], 2)
    n_cutouts = numpy.prod(cutouts.shape[:-2])

    grid = (4, 16, 16)
    shape = (n_cutouts, *cutouts.shape[:-2])
    block = tuple((s + g - 1) // g for (s, g) in zip(shape, grid))
    get_cutouts_float32_kernel(grid, block, (obj, cutouts, start_idxs, n_cutouts, *obj.shape[-2:], *cutouts.shape[-2:]))
    return cutouts