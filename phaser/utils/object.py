"""
Object & object cutout utilities
"""

from dataclasses import dataclass
import typing as t

import numpy
from numpy.typing import ArrayLike, DTypeLike, NDArray

from .num import get_array_module, to_real_dtype, NumT, ComplexT


@t.overload
def random_phase_object(shape: t.Tuple[int, int], mag: float = 1e-6, *, seed: t.Optional[t.Any] = None,
                        dtype: t.Optional[ComplexT] = None, xp: t.Any = None) -> NDArray[ComplexT]:
    ...

@t.overload
def random_phase_object(shape: t.Tuple[int, int], mag: float = 1e-6, *, seed: t.Optional[t.Any] = None,
                        dtype: t.Optional[DTypeLike] = None, xp: t.Any = None) -> NDArray[numpy.complexfloating]:
    ...

def random_phase_object(shape: t.Tuple[int, int], mag: float = 1e-6, *, seed: t.Optional[t.Any] = None,
                        dtype: t.Optional[DTypeLike] = None, xp: t.Any = None) -> NDArray[numpy.complexfloating]:
    if xp is None or t.TYPE_CHECKING:
        xp2 = numpy
    else:
        xp2 = xp

    if isinstance(seed, numpy.random.RandomState):
        rng = seed
    else:
        rng = numpy.random.RandomState(seed=seed)

    real_dtype = to_real_dtype(dtype) if dtype is not None else numpy.float_
    obj_angle = xp2.array(rng.normal(0., mag, shape, dtype=real_dtype), dtype=real_dtype)
    return xp2.cos(obj_angle) + xp2.sin(obj_angle) * 1.j


@dataclass(frozen=True, init=False)
class ObjectSampling:
    shape: NDArray[numpy.int_]
    """Sampling shape `(n_y, n_x)`"""
    sampling: NDArray[numpy.float_]
    """Sample spacing `(s_y, s_x)`"""
    corner: NDArray[numpy.float_]
    """Corner of sampling `(y_min, x_min)`"""

    region_min: t.Optional[NDArray[numpy.float_]]
    region_max: t.Optional[NDArray[numpy.float_]]

    @property
    def min(self) -> NDArray[numpy.float_]:
        return self.corner

    @property
    def max(self) -> NDArray[numpy.float_]:
        return self.corner + self.shape * self.sampling

    def __init__(self, shape: t.Tuple[int, int], sampling: ArrayLike, corner: t.Optional[ArrayLike] = None,
                 region_min: t.Optional[ArrayLike] = None, region_max: t.Optional[ArrayLike] = None):
        object.__setattr__(self, 'shape', numpy.broadcast_to(numpy.array(shape, dtype=numpy.int_), (2,)))
        object.__setattr__(self, 'sampling', numpy.broadcast_to(numpy.array(sampling, dtype=numpy.float_), (2,)))
        object.__setattr__(self, 'region_min', numpy.broadcast_to(numpy.array(region_min, dtype=numpy.int_), (2,)) if region_min is not None else None)
        object.__setattr__(self, 'region_max', numpy.broadcast_to(numpy.array(region_max, dtype=numpy.int_), (2,)) if region_max is not None else None)

        if corner is None:
            extent = self.shape * self.sampling
            # TODO is this strictly correct or off by a half pixel?
            corner = -extent / 2.
        else:
            corner = numpy.broadcast_to(numpy.array(corner, dtype=numpy.float_), (2,))

        object.__setattr__(self, 'corner', corner)

    @classmethod
    def from_scan(cls: t.Type[t.Self], scan_positions: NDArray[numpy.floating], sampling: ArrayLike, pad: ArrayLike = 0.) -> t.Self:
        """Create an ObjectSampling around the given scan positions, padded by a radius `pad` in y and x."""
        sampling = numpy.array(sampling, dtype=numpy.float_)
        pad = numpy.broadcast_to(pad, (2,))
        y_min, y_max = numpy.nanmin(scan_positions[..., 0]), numpy.nanmax(scan_positions[:, 0])
        x_min, x_max = numpy.nanmin(scan_positions[..., 1]), numpy.nanmax(scan_positions[:, 1])

        n_y = numpy.ceil((2.*pad[0] + y_max - y_min) / sampling[0]).astype(numpy.int_) + 1
        n_x = numpy.ceil((2.*pad[1] + x_max - x_min) / sampling[1]).astype(numpy.int_) + 1

        return cls((n_y, n_x), sampling, (y_min - pad[0], x_min - pad[1]), (y_min, x_min), (y_max, x_max))

    def _pos_to_object_idx(self, pos: NDArray[numpy.float_], cutout_shape: t.Tuple[int, ...]) -> NDArray[numpy.float_]:
        """Return starting index for the cutout closest to centered around `pos` (`(y, x)`)"""
        return (numpy.array(pos) - self.corner) / self.sampling - numpy.array(cutout_shape[-2:]) / 2.

    def slice_at_pos(self, pos: ArrayLike, cutout_shape: t.Tuple[int, ...]) -> t.Tuple[slice, slice]:
        """
        Return slices to cutout a region of shape `cutout_shape` around the object position `pos`.

        # Parameters

         - `pos`: Object position to return a cutout around `(y, x)`
         - `cutout_shape`: Shape of cutout to return.

        Returns slices which can be used to index into an object. E.g. `obj[slice_at_pos(pos, (32, 32))]`
        will return an array of shape `(32, 32)`.
        """
        pos = numpy.array(pos)
        (start_i, start_j) = numpy.round(self._pos_to_object_idx(pos, cutout_shape)).astype(numpy.int_)
        return (
            slice(start_i, start_i + cutout_shape[-2]),
            slice(start_j, start_j + cutout_shape[-1]),
        )

    def get_subpx_shifts(self, pos: ArrayLike, cutout_shape: t.Tuple[int, ...]) -> NDArray[numpy.float_]:
        """
        Get the subpixel shifts between `pos` and the cutout region around `pos`.

        Returns the shift from the rounded position to the actual position.
        """
        pos = self._pos_to_object_idx(numpy.array(pos), cutout_shape)
        return pos - numpy.round(pos)

    def get_view_at_pos(self, arr: NDArray[NumT], pos: ArrayLike, shape: t.Tuple[int, ...]) -> NDArray[NumT]:
        """
        Get cutout views of `arr` of shape `shape` around positions `pos`
        """
        pos = numpy.array(pos)
        if pos.ndim == 1:
            return arr[self.slice_at_pos(pos, shape)]
        
        xp = get_array_module(arr)
        out = xp.empty((*pos.shape[:-1], *shape[-2:]), dtype=arr.dtype)
        for idx in numpy.ndindex(pos.shape[:-1]):
            out[*idx] = arr[self.slice_at_pos(pos[idx], shape)]

        return out

    def set_view_at_pos(self, arr: numpy.ndarray, pos: ArrayLike, view: numpy.ndarray):
        """
        Set cutout views of `arr` of shape `shape` around positions `pos`
        """
        pos = numpy.array(pos)
        if pos.ndim == 1:
            arr[self.slice_at_pos(pos, view.shape)] = view

        for idx in numpy.ndindex(pos.shape[:-1]):
            arr[self.slice_at_pos(pos[idx], view.shape)] = view[idx]

    def add_view_at_pos(self, arr: numpy.ndarray, pos: ArrayLike, view: numpy.ndarray):
        """
        Add to cutout views of `arr` of shape `shape` around positions `pos`
        """
        pos = numpy.array(pos)
        if pos.ndim == 1:
            arr[self.slice_at_pos(pos, view.shape)] += view

        for idx in numpy.ndindex(pos.shape[:-1]):
            arr[self.slice_at_pos(pos[idx], view.shape)] += view[idx]

    def get_region_crop(self) -> t.Tuple[slice, slice]:
        if self.region_min is None:
            min_i, min_j = 0, 0
        else:
            min_i, min_j = numpy.ceil(self._pos_to_object_idx(self.region_min, (0, 0))).astype(numpy.int_)
        if self.region_max is None:
            max_i, max_j = None, None
        else:
            max_i, max_j = numpy.floor(self._pos_to_object_idx(self.region_max, (0, 0))).astype(numpy.int_) + 1

        return (slice(min_i, max_i), slice(min_j, max_j))

    @t.overload
    def grid(self, *, dtype: t.Type[NumT], xp: t.Any = None) -> t.Tuple[NDArray[NumT], NDArray[NumT]]:
        ...

    @t.overload
    def grid(self, *, dtype: t.Optional[DTypeLike] = None, xp: t.Any = None) -> t.Tuple[NDArray[numpy.floating], NDArray[numpy.floating]]:
        ...

    def grid(self, *, dtype: t.Any = None, xp: t.Any = None) -> t.Tuple[NDArray[numpy.number], NDArray[numpy.number]]:
        """Return the sampling grid `(yy, xx)` for this object"""
        if xp is None:
            xp2 = get_array_module(self.sampling, self.corner)
        elif not t.TYPE_CHECKING:
            xp2 = xp
        else:
            xp2 = numpy

        if dtype is None:
            dtype = numpy.common_type(self.sampling, self.corner)

        ys = xp2.linspace(self.min[0], self.max[1], self.shape[0], endpoint=False, dtype=dtype)
        xs = xp2.linspace(self.min[0], self.max[1], self.shape[1], endpoint=False, dtype=dtype)

        return tuple(xp2.meshgrid(ys, xs, indexing='ij'))  # type: ignore