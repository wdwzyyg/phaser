"""
Utilities for image processing & filtering
"""

import typing as t

import numpy
from numpy.typing import ArrayLike, NDArray

from .num import get_array_module


NumT = t.TypeVar('NumT', bound=numpy.number)


@t.overload
def remove_linear_ramp(data: NDArray[NumT]) -> NDArray[NumT]:
    ...

@t.overload
def remove_linear_ramp(data: ArrayLike) -> NDArray[numpy.float_]:
    ...

def remove_linear_ramp(data: ArrayLike) -> NDArray[numpy.number]:
    """
    Removes a linear 'ramp' from an image or stack of images.
    """

    from scipy.linalg import lstsq

    xp = get_array_module(data)
    output = xp.empty_like(data)

    data = xp.array(data)

    (yy, xx) = (arr.flatten() for arr in xp.indices(data.shape[-2:], dtype=float))
    pts = xp.stack((xp.ones_like(xx), xx, yy), axis=-1)

    # TODO fix on gpu
    for idx in numpy.ndindex(data.shape[:-2]):
        layer = data[*idx].astype(numpy.float_)
        p, residues, rank, singular = lstsq(pts, layer.flatten())
        output[*idx] = (layer - (p @ pts.T).reshape(layer.shape)).astype(output.dtype)

    return output