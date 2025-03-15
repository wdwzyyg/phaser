"""
Object & object cutout utilities
"""
from __future__ import annotations

from dataclasses import dataclass, field
import typing as t

import numpy
from numpy.typing import ArrayLike, DTypeLike, NDArray

from .num import get_array_module, cast_array_module, to_real_dtype, as_numpy, at
from .num import to_numpy, to_array, is_cupy, is_jax, NumT, ComplexT, DTypeT
from .misc import create_rng, jax_dataclass


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
      - `dtype`: Output datatype of object. Must be complex. Defaults to `numpy.float64`
      - `xp`: Array module to create object on.
    """
    xp2 = numpy if xp is None else cast_array_module(xp)

    rng = create_rng(seed, 'random_phase_object')

    real_dtype = to_real_dtype(dtype) if dtype is not None else numpy.float64
    obj_angle = xp2.array(rng.normal(0., sigma, tuple(shape)), dtype=real_dtype)
    return xp2.cos(obj_angle) + xp2.sin(obj_angle) * 1.j


def resample_slices(
    obj: NDArray[NumT], old_zs: ArrayLike, old_slice_thick: ArrayLike,
    new_zs: ArrayLike, new_slice_thick: ArrayLike, *,
    mode: t.Literal['edge', 'vacuum', 'average'] = 'vacuum',
) -> NDArray[NumT]:
    """
    Resample an object in Z, such that projected potential is conserved.

    Returns an object of shape `(len(new_zs)-1, *obj.shape[1:])`.
    """
    xp = get_array_module(obj)

    old_zs = to_numpy(numpy.asarray(old_zs))
    new_zs = to_numpy(numpy.asarray(new_zs))
    old_slice_thick = numpy.broadcast_to(as_numpy(old_slice_thick), old_zs.shape)
    new_slice_thick = numpy.broadcast_to(as_numpy(new_slice_thick), new_zs.shape)

    if obj.shape[0] != len(old_zs):
        raise ValueError(f"Expected an object with {len(old_zs)} slices, instead got object shape {obj.shape}")

    # resample from center of slices
    old_zs += old_slice_thick / 2.
    new_zs += new_slice_thick / 2.

    # log to make linear, normalize from projected potential to potential
    obj = (xp.log(obj) / old_slice_thick).astype(obj.dtype)

    if mode == 'edge':
        before_slice = obj[0]
        after_slice = obj[-1]
    elif mode == 'average':
        before_slice = after_slice = xp.nanmean(obj, axis=0)  # type: ignore
    elif mode == 'vacuum':
        before_slice = after_slice = xp.zeros_like(obj[0])
    else:
        raise ValueError(f"Unknown padding mode '{mode}'. Expected 'edge', 'vacuum', or 'average'.")

    # pad old object with outer slices
    old_zs = numpy.concatenate(([old_zs[0] - old_slice_thick[0]], old_zs, [old_zs[-1] + old_slice_thick[-1]]))
    obj = xp.concatenate((before_slice[None, ...], obj, after_slice[None, ...]), axis=0)

    new_obj = _interp1d(obj, old_zs, new_zs)

    # convert back to projected potential, undo log
    return xp.exp((new_obj * new_slice_thick).astype(obj.dtype))


def _interp1d(arr: NDArray[NumT], old_zs: NDArray[numpy.floating], new_zs: NDArray[numpy.floating]) -> NDArray[NumT]:
    """
    Interpolates along the first dimension of `arr`. Boundary values are replaced with the edge value.
    """
    xp = get_array_module(arr)
    slice_is = numpy.searchsorted(old_zs, new_zs) - 1

    new_arr = xp.empty((len(new_zs), *arr.shape[1:]), dtype=arr.dtype)

    delta_zs = numpy.diff(old_zs)

    # TODO specialize this for jax?
    for i, new_z in enumerate(new_zs):
        if new_z <= old_zs[0]: # before
            slice = arr[0]
        elif old_zs[-1] <= new_z: # after
            slice = arr[-1]
        else:
            slice_i = slice_is[i]
            # linearly interpolate
            t = float((new_z - old_zs[slice_i]) / delta_zs[slice_i])
            slice = ((1-t)*arr[slice_i] + t*arr[slice_i + 1]).astype(arr.dtype)

        new_arr = at(new_arr, i).set(slice)

    return new_arr


@jax_dataclass(frozen=True, init=False)
class ObjectSampling:
    shape: NDArray[numpy.int_]
    """Sampling shape `(n_y, n_x)`"""
    sampling: NDArray[numpy.float64]
    """Sample spacing `(s_y, s_x)`"""
    corner: NDArray[numpy.float64]
    """Corner of sampling `(y_min, x_min)`"""

    region_min: t.Optional[NDArray[numpy.float64]]
    region_max: t.Optional[NDArray[numpy.float64]]

    @property
    def min(self) -> NDArray[numpy.float64]:
        """Minimum object pixel position (y, x). Alias for `corner`."""
        return self.corner

    @property
    def max(self) -> NDArray[numpy.float64]:
        """Maximum pixel position (y, x)."""
        return (self.corner + (self.shape - 1) * self.sampling).astype(numpy.float64)

    @property
    def extent(self) -> NDArray[numpy.float64]:
        return (self.shape * self.sampling).astype(numpy.float64)

    def __init__(self, shape: t.Tuple[int, int], sampling: ArrayLike, corner: t.Optional[ArrayLike] = None,
                 region_min: t.Optional[ArrayLike] = None, region_max: t.Optional[ArrayLike] = None):
        object.__setattr__(self, 'shape', numpy.broadcast_to(as_numpy(shape).astype(numpy.int_), (2,)))
        object.__setattr__(self, 'sampling', numpy.broadcast_to(as_numpy(sampling).astype(numpy.float64), (2,)))
        object.__setattr__(self, 'region_min', numpy.broadcast_to(as_numpy(region_min).astype(numpy.float64), (2,)) if region_min is not None else None)
        object.__setattr__(self, 'region_max', numpy.broadcast_to(as_numpy(region_max).astype(numpy.float64), (2,)) if region_max is not None else None)

        if corner is None:
            corner = -self.extent / 2. + self.sampling/2. #* (self.shape % 2)
        else:
            corner = numpy.broadcast_to(as_numpy(corner).astype(numpy.float64), (2,))

        object.__setattr__(self, 'corner', corner)

    @classmethod
    def from_scan(cls: t.Type[t.Self], scan_positions: NDArray[numpy.floating], sampling: ArrayLike, pad: ArrayLike = 0) -> t.Self:
        """Create an ObjectSampling around the given scan positions, padded by at least a radius `pad` in real-space."""
        sampling = as_numpy(sampling).astype(numpy.float64)
        pad = numpy.broadcast_to(pad, (2,)).astype(numpy.int_)

        y_min, y_max = float(numpy.nanmin(scan_positions[..., 0])), float(numpy.nanmax(scan_positions[..., 0]))
        x_min, x_max = float(numpy.nanmin(scan_positions[..., 1])), float(numpy.nanmax(scan_positions[..., 1]))

        n_y = numpy.ceil((2.*pad[0] + y_max - y_min) / sampling[0]).astype(numpy.int_) + 1
        n_x = numpy.ceil((2.*pad[1] + x_max - x_min) / sampling[1]).astype(numpy.int_) + 1

        return cls((n_y, n_x), sampling, (y_min - pad[0], x_min - pad[1]), (y_min, x_min), (y_max, x_max))

    def _pos_to_object_idx(self, pos: ArrayLike, cutout_shape: t.Tuple[int, ...]) -> NDArray[numpy.float64]:
        """Return starting index for the cutout closest to centered around `pos` (`(y, x)`)"""

        if not is_jax(pos):  # allow jax tracers to work right
            pos = as_numpy(pos)

        # for a given cutout, shift to the top left pixel of that cutout
        # e.g. a 2x2 cutout needs shifted by s/2
        shift = -numpy.maximum(0., (numpy.array(cutout_shape[-2:]) - 1.)) / 2.

        return ((pos - self.corner) / self.sampling + shift).astype(numpy.float64)  # type: ignore

    def slice_at_pos(self, pos: ArrayLike, cutout_shape: t.Tuple[int, ...]) -> t.Tuple[slice, slice]:
        """
        Return slices to cutout a region of shape `cutout_shape` around the object position `pos`.

        # Parameters

         - `pos`: Object position to return a cutout around `(y, x)`
         - `cutout_shape`: Shape of cutout to return.

        Returns slices which can be used to index into an object. E.g. `obj[slice_at_pos(pos, (32, 32))]`
        will return an array of shape `(32, 32)`.
        """

        idxs = self._pos_to_object_idx(pos, cutout_shape)
        (start_i, start_j) = map(int, numpy.round(idxs).astype(numpy.int64))
        assert start_i >= 0 and start_j >= 0
        return (
            slice(start_i, start_i + cutout_shape[-2]),
            slice(start_j, start_j + cutout_shape[-1]),
        )

    def get_subpx_shifts(self, pos: ArrayLike, cutout_shape: t.Tuple[int, ...]) -> NDArray[numpy.float64]:
        """
        Get the subpixel shifts between `pos` and the cutout region around `pos`.

        Returns the shift from the rounded position towards the actual position.
        """
        pos = self._pos_to_object_idx(to_array(pos), cutout_shape)
        return (pos - get_array_module(pos).round(pos)).astype(numpy.float64)

    @t.overload
    def cutout(  # pyright: ignore[reportOverlappingOverload]
        self, arr: NDArray[DTypeT], pos: ArrayLike, shape: t.Tuple[int, ...]
    ) -> ObjectCutout[DTypeT]:
        ...

    @t.overload
    def cutout(self, arr: numpy.ndarray, pos: ArrayLike, shape: t.Tuple[int, ...]) -> ObjectCutout[numpy.generic]:
        ...

    def cutout(self, arr: numpy.ndarray, pos: ArrayLike, shape: t.Tuple[int, ...]) -> ObjectCutout[t.Any]:
        xp = get_array_module(arr, pos)
        return ObjectCutout(self, xp.array(arr), xp.array(pos), shape)

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

    def get_region_mask(self, xp: t.Any = None) -> NDArray[numpy.bool_]:
        xp2 = numpy if xp is None else cast_array_module(xp)
        mask = xp2.zeros(self.shape, dtype=numpy.bool_)
        mask[*self.get_region_crop()] = 1
        return mask

    @t.overload
    def grid(  # pyright: ignore[reportOverlappingOverload]
        self, *, dtype: t.Type[NumT], xp: t.Any = None
    ) -> t.Tuple[NDArray[NumT], NDArray[NumT]]:
        ...

    @t.overload
    def grid(self, *, dtype: t.Optional[DTypeLike] = None, xp: t.Any = None) -> t.Tuple[NDArray[numpy.floating], NDArray[numpy.floating]]:
        ...

    def grid(self, *, dtype: t.Any = None, xp: t.Any = None) -> t.Tuple[NDArray[numpy.number], NDArray[numpy.number]]:
        """Return the sampling grid `(yy, xx)` for this object"""
        xp2 = get_array_module(self.sampling, self.corner) if xp is None else cast_array_module(xp)

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
        self._start_idxs = numpy.round(self.sampling._pos_to_object_idx(self.pos, self.cutout_shape)).astype(numpy.int_) # type: ignore
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
            except (ImportError, NotImplementedError):
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
