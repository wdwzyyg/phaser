"""
Utilities for image processing & filtering
"""

import typing as t

import numpy
from numpy.typing import ArrayLike, NDArray

from .num import get_array_module


NumT = t.TypeVar('NumT', bound=numpy.number)


@t.overload
def remove_linear_ramp(data: NDArray[NumT], mask: t.Optional[NDArray[numpy.bool_]] = None) -> NDArray[NumT]:
    ...

@t.overload
def remove_linear_ramp(data: ArrayLike, mask: t.Optional[NDArray[numpy.bool_]] = None) -> NDArray[numpy.float64]:
    ...

def remove_linear_ramp(data: ArrayLike, mask: t.Optional[NDArray[numpy.bool_]] = None) -> NDArray[numpy.number]:
    """
    Removes a linear 'ramp' from an image or stack of images.
    """

    xp = get_array_module(data)
    output = xp.empty_like(data)

    data = xp.array(data)

    (yy, xx) = (arr.flatten() for arr in xp.indices(data.shape[-2:], dtype=float))
    pts = xp.stack((xp.ones_like(xx), xx, yy), axis=-1)

    if mask is None:
        mask = xp.ones(len(yy), dtype=numpy.bool_)
    else:
        mask = mask.flatten()

    # TODO fix on gpu
    for idx in numpy.ndindex(data.shape[:-2]):
        layer = data[*idx].astype(numpy.float64)
        p, residues, rank, singular = xp.linalg.lstsq(pts[mask], layer.flatten()[mask], rcond=None)
        output[*idx] = (layer - (p @ pts.T).reshape(layer.shape)).astype(output.dtype)

    return output