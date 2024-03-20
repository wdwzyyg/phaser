
import functools
import typing as t
from types import NotImplementedType

import cupy
from cupy.cuda.runtime import CUDARuntimeError
import numpy

# grid
#  block
#    thread
# blocks contain threads, a grid contains blocks
# max 1024 threads/block, maximum per axis (1024, 1024, 64)
# max grid size per axis ~2^32
# x is the fast axis (warp axis)

# TODO:
#   much more thorough testing
#   support >2d objects (just add wrapper around?)
#   optimize block size

def set_cutouts(obj: cupy.ndarray, cutouts: cupy.ndarray, start_idxs: cupy.ndarray):
    assert obj.ndim >= 2
    if obj.ndim > 2:
        raise NotImplementedError()  # for now

    kernel = _get_cutout_kernel(obj.dtype, 'set')
    if kernel is NotImplemented:
        raise NotImplementedError()

    # output array must be the correct dtype
    assert obj.dtype == cutouts.dtype
    # last two dimensions of output array must be contiguous
    assert obj[tuple(0 for _ in obj.shape[:-2])].flags.c_contiguous

    # cutouts must be the correct shape
    assert cutouts.shape[:-2] == (*obj.shape[:-2], *start_idxs.shape[:-1])
    assert obj.shape[:-2] == cutouts.shape[:obj.ndim-2]
    assert start_idxs.shape[-1] == 2

    n_cutouts = numpy.prod(cutouts.shape[obj.ndim-2:-2])

    block = (128, 8, 1)
    shape = (cutouts.shape[-1], cutouts.shape[-2], n_cutouts)
    grid = tuple((s + b - 1) // b for (s, b) in zip(shape, block))

    start_idxs = cupy.ascontiguousarray(start_idxs.astype(cupy.uint64))

    args = (obj, cupy.ascontiguousarray(cutouts), start_idxs, n_cutouts, *obj.shape[-2:], *cutouts.shape[-2:])
    kernel(grid, block, args)
    return cutouts


def get_cutouts(obj: cupy.ndarray, start_idxs: cupy.ndarray, cutout_shape: t.Tuple[int, int]) -> cupy.ndarray:
    assert obj.ndim >= 2
    if obj.ndim > 2:
        raise NotImplementedError()  # for now

    kernel = _get_cutout_kernel(obj.dtype, 'get')
    if kernel is NotImplemented:
        raise NotImplementedError()

    # check start_idxs shape
    assert start_idxs.shape[-1] == 2
    cutouts = cupy.empty((*obj.shape[:-2], *start_idxs.shape[:-1], *cutout_shape), dtype=obj.dtype)
    # output array must be contiguous
    assert cutouts.flags.c_contiguous
    n_cutouts = numpy.prod(start_idxs.shape[:-1])

    block = (128, 8, 1)
    shape = (cutouts.shape[-1], cutouts.shape[-2], n_cutouts)
    grid = tuple((s + b - 1) // b for (s, b) in zip(shape, block))

    start_idxs = cupy.ascontiguousarray(start_idxs.astype(cupy.uint64))

    args = (cupy.ascontiguousarray(obj), cutouts, start_idxs, n_cutouts, *obj.shape[-2:], *cutouts.shape[-2:])
    kernel(grid, block, args)
    return cutouts


def add_cutouts(obj: cupy.ndarray, cutouts: cupy.ndarray, start_idxs: cupy.ndarray):
    assert obj.ndim >= 2  # for now
    if obj.ndim > 2:
        raise NotImplementedError()  # for now

    kernel = _get_cutout_kernel(obj.dtype, 'add')
    if kernel is NotImplemented:
        raise NotImplementedError()

    # output array must be the correct dtype
    assert obj.dtype == cutouts.dtype
    # last two dimensions of output array must be contiguous
    assert obj[tuple(0 for _ in obj.shape[:-2])].flags.c_contiguous

    # cutouts must be the correct shape
    assert cutouts.shape[:-2] == (*obj.shape[:-2], *start_idxs.shape[:-1])
    assert obj.shape[:-2] == cutouts.shape[:obj.ndim-2]
    assert start_idxs.shape[-1] == 2

    n_cutouts = numpy.prod(cutouts.shape[obj.ndim-2:-2])

    block = (128, 8, 1)
    shape = (cutouts.shape[-1], cutouts.shape[-2], n_cutouts)
    grid = tuple((s + b - 1) // b for (s, b) in zip(shape, block))

    start_idxs = cupy.ascontiguousarray(start_idxs.astype(cupy.uint64))

    args = (obj, cupy.ascontiguousarray(cutouts), start_idxs, n_cutouts, *obj.shape[-2:], *cutouts.shape[-2:])
    kernel(grid, block, args)
    return cutouts


_CUTOUT_OPERATIONS = {
    'get': "cutouts[cutout_idx] = obj[obj_idx];",
    'add': "atomicAdd(&obj[obj_idx], cutouts[cutout_idx]);",
    # TODO does this need to be atomic?
    'set': "obj[obj_idx] = cutouts[cutout_idx];",
}

_CUTOUT_CONST = {
    'get': (True, False),
    'set': (False, True),
    'add': (False, True),
}

# cupy/_core/_scalar.pyx
_DTYPE_TO_KERNEL_TYPE: t.Dict[t.Type[numpy.generic], str] = {
    numpy.bool_: "bool",
    numpy.int8: "signed char",
    numpy.uint8: "unsigned char",
    numpy.int16: "signed short",
    numpy.uint16: "unsigned short",
    numpy.int32: "signed int",
    numpy.uint32: "unsigned int",
    numpy.int64: "signed long long",
    numpy.uint64: "unsigned long long",
    numpy.float16: "float16",
    numpy.float32: "float",
    numpy.float64: "double",
    numpy.complex64: "complex<float>",
    numpy.complex128: "complex<double>",
}

@functools.cache
def _get_cutout_kernel(dtype: numpy.dtype, operation: str) -> t.Union[cupy.RawKernel, NotImplementedType]:
    try:
        ty = _DTYPE_TO_KERNEL_TYPE[dtype.type]
        op = _CUTOUT_OPERATIONS[operation]
        const_s = tuple("const " if c else "" for c in _CUTOUT_CONST[operation])
    except KeyError:
        return NotImplemented

    if operation == 'add':
        if ty not in {
                "float16", "float", "double", "complex<float>", "complex<double>",
                "signed int", "unsigned int", "unsigned long long",
            }:
            return NotImplemented
        # double atomicAdd requires compute capability >=6.x
        device = cupy.cuda.Device()
        if ty in ('complex<double>', 'double') and int(device.compute_capability[0]) < 6:
            return NotImplemented

    kernel_name = f"{operation}_cutouts_{dtype.type.__name__}"

    kernel = cupy.RawKernel(rf"""
#include <cupy/complex.cuh>

#define T {ty}

// support atomicAdd for complex datatypes
// we can just add real and complex parts separately
template<typename U>
__device__ inline void atomicAdd(complex<U> *address, complex<U> val) {{
    atomicAdd(&reinterpret_cast<U*>(address)[0], val.real());
    atomicAdd(&reinterpret_cast<U*>(address)[1], val.imag());
}}

extern "C" __global__
void {kernel_name}({const_s[0]}T *obj, {const_s[1]}T *cutouts, const long long *start_idxs, long long n_cutouts,
                         long long obj_shape_y, long long obj_shape_x, long long cutout_shape_y, long long cutout_shape_x) {{
    size_t          j = blockIdx.x * blockDim.x + threadIdx.x;
    size_t          i = blockIdx.y * blockDim.y + threadIdx.y;
    size_t cutout_num = blockIdx.z * blockDim.z + threadIdx.z;

    // check if our thread is outside the requested cutout
    if (cutout_num >= n_cutouts || i >= cutout_shape_y || j >= cutout_shape_x) {{
        return;
    }}

    // offset to get object location, check that it's in bounds
    long long obj_i = i + start_idxs[2*cutout_num];
    long long obj_j = j + start_idxs[2*cutout_num + 1];
    assert(obj_i >= 0 && obj_i < obj_shape_y && obj_j >= 0 && obj_j < obj_shape_x);
    if (!(obj_i >= 0 && obj_i < obj_shape_y && obj_j >= 0 && obj_j < obj_shape_x)) {{
        return;
    }}

    // compute cutout and object indices
    size_t cutout_idx = cutout_shape_y * cutout_shape_x * cutout_num + cutout_shape_x * i + j;
    size_t obj_idx = obj_shape_x * obj_i + obj_j;
    // and perform our operation
    {op}
}}
""", kernel_name)
    kernel.compile()
    return kernel
