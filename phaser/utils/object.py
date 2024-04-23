"""
Object & object cutout utilities
"""
from __future__ import annotations

from dataclasses import dataclass, field
import typing as t

import numpy
from numpy.typing import ArrayLike, DTypeLike, NDArray

from .num import get_array_module, to_real_dtype, is_cupy, is_jax
from .num import NumT, ComplexT, DTypeT
from .misc import create_rng


@t.overload
def random_phase_object(shape: t.Iterable[int], sigma: float = 1e-6, *, seed: t.Optional[object] = None,
                        dtype: t.Optional[ComplexT] = None, xp: t.Any = None) -> NDArray[ComplexT]:
    ...

@t.overload
def random_phase_object(shape: t.Iterable[int], sigma: float = 1e-6, *, seed: t.Optional[object] = None,
                        dtype: t.Optional[DTypeLike] = None, xp: t.Any = None) -> NDArray[numpy.complexfloating]:
    ...

def random_phase_object(shape: t.Iterable[int], sigma: float = 1e-6, *, seed: t.Optional[object] = None,
                        dtype: t.Optional[DTypeLike] = None, xp: t.Any = None) -> NDArray[numpy.complexfloating]:
    """
    Construct a random phase object of shape `shape`.

    # Parameters

      - `shape`: Shape of random phase object to generate
      - `sigma`: Standard deviation of phase variation to create
      - `seed`: Random seed or existing random number generator to use. See `create_rng` for more details.
      - `dtype`: Output datatype of object. Must be complex. Defaults to `numpy.float_`
      - `xp`: Array module to create object on.
    """
    if xp is None or t.TYPE_CHECKING:
        xp2 = numpy
    else:
        xp2 = xp

    rng = create_rng(seed, 'random_phase_object')

    real_dtype = to_real_dtype(dtype) if dtype is not None else numpy.float_
    obj_angle = xp2.array(rng.normal(0., sigma, tuple(shape)), dtype=real_dtype)
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
        """Minimum object pixel position (y, x). Alias for `corner`."""
        return self.corner

    @property
    def max(self) -> NDArray[numpy.float_]:
        """Maximum pixel position (y, x)."""
        return self.corner + (self.shape - 1) * self.sampling

    @property
    def extent(self) -> NDArray[numpy.float_]:
        return self.shape * self.sampling

    def __init__(self, shape: t.Tuple[int, int], sampling: ArrayLike, corner: t.Optional[ArrayLike] = None,
                 region_min: t.Optional[ArrayLike] = None, region_max: t.Optional[ArrayLike] = None):
        object.__setattr__(self, 'shape', numpy.broadcast_to(numpy.array(shape, dtype=numpy.int_), (2,)))
        object.__setattr__(self, 'sampling', numpy.broadcast_to(numpy.array(sampling, dtype=numpy.float_), (2,)))
        object.__setattr__(self, 'region_min', numpy.broadcast_to(numpy.array(region_min, dtype=numpy.float_), (2,)) if region_min is not None else None)
        object.__setattr__(self, 'region_max', numpy.broadcast_to(numpy.array(region_max, dtype=numpy.float_), (2,)) if region_max is not None else None)

        if corner is None:
            corner = -self.extent / 2. + self.sampling/2. #* (self.shape % 2)
        else:
            corner = numpy.broadcast_to(numpy.array(corner, dtype=numpy.float_), (2,))

        object.__setattr__(self, 'corner', corner)

    @classmethod
    def from_scan(cls: t.Type[t.Self], scan_positions: NDArray[numpy.floating], sampling: ArrayLike, pad: ArrayLike = 0) -> t.Self:
        """Create an ObjectSampling around the given scan positions, padded by at least a radius `pad` in real-space."""
        sampling = numpy.array(sampling, dtype=numpy.float_)
        pad = numpy.broadcast_to(pad, (2,)).astype(numpy.int_)

        y_min, y_max = numpy.nanmin(scan_positions[..., 0]), numpy.nanmax(scan_positions[:, 0])
        x_min, x_max = numpy.nanmin(scan_positions[..., 1]), numpy.nanmax(scan_positions[:, 1])

        n_y = numpy.ceil((2.*pad[0] + y_max - y_min) / sampling[0]).astype(numpy.int_) + 1
        n_x = numpy.ceil((2.*pad[1] + x_max - x_min) / sampling[1]).astype(numpy.int_) + 1

        return cls((n_y, n_x), sampling, (y_min - pad[0], x_min - pad[1]), (y_min, x_min), (y_max, x_max))

    def _pos_to_object_idx(self, pos: NDArray[numpy.float_], cutout_shape: t.Tuple[int, ...]) -> NDArray[numpy.float_]:
        """Return starting index for the cutout closest to centered around `pos` (`(y, x)`)"""

        # for a given cutout, shift to the top left pixel of that cutout
        # e.g. a 2x2 cutout needs shifted by s/2
        shift = -numpy.maximum(0., (numpy.array(cutout_shape[-2:]) - 1.)) / 2.

        return (numpy.array(pos) - self.corner) / self.sampling + shift

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
        assert start_i >= 0 and start_j >= 0
        return (
            slice(start_i, start_i + cutout_shape[-2]),
            slice(start_j, start_j + cutout_shape[-1]),
        )

    def get_subpx_shifts(self, pos: ArrayLike, cutout_shape: t.Tuple[int, ...]) -> NDArray[numpy.float_]:
        """
        Get the subpixel shifts between `pos` and the cutout region around `pos`.

        Returns the shift from the rounded position towards the actual position.
        """
        pos = self._pos_to_object_idx(numpy.array(pos), cutout_shape)
        return pos - numpy.round(pos)

    @t.overload
    def cutout(self, arr: NDArray[DTypeT], pos: ArrayLike, shape: t.Tuple[int, ...]) -> ObjectCutout[DTypeT]:
        ...

    @t.overload
    def cutout(self, arr: numpy.ndarray, pos: ArrayLike, shape: t.Tuple[int, ...]) -> ObjectCutout[numpy.generic]:
        ...

    def cutout(self, arr: numpy.ndarray, pos: ArrayLike, shape: t.Tuple[int, ...]) -> ObjectCutout[t.Any]:
        return ObjectCutout(self, arr, numpy.array(pos), shape)

    def get_view_at_pos(self, arr: NDArray[NumT], pos: ArrayLike, shape: t.Tuple[int, ...]) -> NDArray[NumT]:
        """
        Get cutout views of `arr` of shape `shape` around positions `pos`
        """
        return self.cutout(arr, pos, shape).get()

    def set_view_at_pos(self, arr: NDArray[NumT], pos: ArrayLike, view: numpy.ndarray) -> NDArray[NumT]:
        """
        Set cutout views of `arr` of shape `shape` around positions `pos`
        """
        cutout = self.cutout(arr, pos, view.shape).set(view)
        return cutout.obj

    def add_view_at_pos(self, arr: NDArray[NumT], pos: ArrayLike, view: numpy.ndarray) -> NDArray[NumT]:
        """
        Add to cutout views of `arr` of shape `shape` around positions `pos`
        """
        cutout = self.cutout(arr, pos, view.shape).add(view)
        return cutout.obj

    def get_region_crop(self) -> t.Tuple[slice, slice]:
        if self.region_min is None:
            min_i, min_j = None, None
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

        ys = xp2.linspace(self.min[0], self.max[0], self.shape[0], endpoint=True, dtype=dtype)
        xs = xp2.linspace(self.min[1], self.max[1], self.shape[1], endpoint=True, dtype=dtype)

        return tuple(xp2.meshgrid(ys, xs, indexing='ij'))  # type: ignore

    def mpl_extent(self, center: bool = True) -> t.Tuple[float, float, float, float]:
        """
        Return the extent of the sampling grid, for use in matplotlib.

        Extent is returned as `(left, right, bottom, top)`.
        If `center` is specified (the default), samples correspond to the center of pixels.
        Otherwise, they correspond to the corners of pixels.
        """
        # shift pixel corners to centers
        shift = self.min - self.sampling/2. * int(center)
        return (shift[1], self.extent[1] + shift[1], self.extent[0] + shift[0], shift[0])


@dataclass
class ObjectCutout(t.Generic[DTypeT]):
    sampling: ObjectSampling
    obj: NDArray[DTypeT]
    pos: NDArray[numpy.floating]
    cutout_shape: t.Tuple[int, ...]

    _start_idxs: NDArray[numpy.int_] = field(init=False)

    def __post_init__(self):
        self._start_idxs = numpy.round(self.sampling._pos_to_object_idx(self.pos, self.cutout_shape)).astype(numpy.int_)
        self._start_idxs = get_array_module(self.obj).array(self._start_idxs)

    @property
    def shape(self) -> t.Tuple[int, ...]:
        return (*self.pos.shape[:-1], *self.obj.shape[:-2], *self.cutout_shape[-2:])

    def get(self) -> NDArray[DTypeT]:
        if is_jax(self.obj):
            from ._jax_kernels import get_cutouts
            return t.cast(NDArray[DTypeT], get_cutouts(self.obj, self._start_idxs, tuple(self.cutout_shape)))

        if is_cupy(self.obj):
            try:
                from ._cuda_kernels import get_cutouts
                return get_cutouts(self.obj, self._start_idxs, self.cutout_shape)  # type: ignore
            except (ImportError, NotImplementedError):
                pass

        xp = get_array_module(self.obj)
        out = xp.empty(self.shape, dtype=self.obj.dtype)
        for idx in numpy.ndindex(self.pos.shape[:-1]):
            # todo make slices outside of loop
            out[*idx] = self.obj[..., *self.sampling.slice_at_pos(self.pos[idx], self.cutout_shape)]

        return out

    def set(self, view: NDArray[DTypeT]) -> ObjectCutout[DTypeT]:
        if is_jax(self.obj):
            from ._jax_kernels import set_cutouts
            obj = t.cast(NDArray[DTypeT], set_cutouts(self.obj, view, self._start_idxs))
            return self._with_obj(obj)

        if is_cupy(self.obj):
            try:
                from ._cuda_kernels import set_cutouts
                set_cutouts(self.obj, view, self._start_idxs)
                return self
            except (ImportError, NotImplementedError):
                pass

        for idx in numpy.ndindex(self.pos.shape[:-1]):
            # todo make slices outside of loop
            self.obj[..., *self.sampling.slice_at_pos(self.pos[idx], view.shape)] = view[idx]
        return self

    def add(self, view: NDArray[DTypeT]) -> ObjectCutout[DTypeT]:
        if is_jax(self.obj):
            from ._jax_kernels import add_cutouts
            obj = t.cast(NDArray[DTypeT], add_cutouts(self.obj, view, self._start_idxs))
            return self._with_obj(obj)

        if is_cupy(self.obj):
            try:
                from ._cuda_kernels import add_cutouts
                add_cutouts(self.obj, view, self._start_idxs)
                return self
            except (ImportError, NotImplementedError) as e:
                pass

        for idx in numpy.ndindex(self.pos.shape[:-1]):
            # todo make slices outside of loop
            self.obj[..., *self.sampling.slice_at_pos(self.pos[idx], view.shape)] += view[idx]
        return self

    def _with_obj(self, obj: NDArray[DTypeT]) -> ObjectCutout[DTypeT]:
        return ObjectCutout(
            self.sampling,
            obj,
            self.pos.copy(),
            self.cutout_shape
        )

    def __cupy_get_ndarray__(self) -> NDArray[DTypeT]:
        if not is_cupy(self.obj):
            raise AttributeError()
        return self.get()

    def __array__(self) -> NDArray[DTypeT]:
        return self.get()