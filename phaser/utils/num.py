"""
General numeric utilities.
"""

from dataclasses import dataclass
import typing as t

import numpy

from numpy.typing import ArrayLike, DTypeLike, NDArray


NumT = t.TypeVar('NumT', bound=numpy.number)
FloatT = t.TypeVar('FloatT', bound=numpy.floating)
ComplexT = t.TypeVar('ComplexT', bound=numpy.complexfloating)
DTypeT = t.TypeVar('DTypeT', bound=numpy.generic)
T = t.TypeVar('T')
P = t.ParamSpec('P')


def get_array_module(*arrs: ArrayLike):
    try:
        from cupy import get_array_module as f  # type: ignore
        if not t.TYPE_CHECKING:
            return f(*arrs)
    except ImportError:
        pass
    return numpy


def get_scipy_module(*arrs: ArrayLike):
    import scipy

    try:
        from cupyx.scipy import get_array_module as f  # type: ignore
        if not t.TYPE_CHECKING:
            return f(*arrs)
    except ImportError:
        pass

    return scipy


def fuse(*args, **kwargs) -> t.Callable[[T], T]:
    """
    Equivalent to `cupy.fuse`, if supported.
    """
    try:
        import cupy  # type: ignore
        if not t.TYPE_CHECKING:
            return cupy.fuse(*args, **kwargs)
    except ImportError:
        pass
    return lambda x: x


_COMPLEX_MAP: t.Dict[t.Type[numpy.floating], t.Type[numpy.complexfloating]] = {
    numpy.float_: numpy.complex_,
    numpy.float32: numpy.complex64,
    numpy.float64: numpy.complex128,
    #numpy.float80: numpy.complex160,
    #numpy.float96: numpy.complex192,
    #numpy.float128: numpy.complex256,
    #numpy.float256: numpy.complex512,
}

_REAL_MAP: t.Dict[t.Type[numpy.complexfloating], t.Type[numpy.floating]] = dict((v, k) for (k, v) in _COMPLEX_MAP.items())


@t.overload
def to_complex_dtype(dtype: t.Type[numpy.float_]) -> t.Type[numpy.complex_]:
    ...

@t.overload
def to_complex_dtype(dtype: t.Type[numpy.float32]) -> t.Type[numpy.complex64]:
    ...

@t.overload
def to_complex_dtype(dtype: t.Type[numpy.float64]) -> t.Type[numpy.complex128]:
    ...

@t.overload
def to_complex_dtype(dtype: DTypeLike) -> t.Type[numpy.complexfloating]:
    ...

def to_complex_dtype(dtype: DTypeLike) -> t.Type[numpy.complexfloating]:
    """
    Convert a floating point dtype to a complex version.
    """

    dtype = numpy.dtype(dtype).type

    if not isinstance(dtype, type) or not issubclass(dtype, (numpy.floating, numpy.complexfloating)):
        raise TypeError("Non-floating point datatype")

    if issubclass(dtype, numpy.complexfloating):
        return dtype

    try:
        return _COMPLEX_MAP[dtype]
    except KeyError:
        raise TypeError(f"Unsupported datatype '{dtype}'") from None


@t.overload
def to_real_dtype(dtype: t.Type[numpy.complex_]) -> t.Type[numpy.float_]:
    ...

@t.overload
def to_real_dtype(dtype: t.Type[numpy.complex64]) -> t.Type[numpy.float32]:
    ...

@t.overload
def to_real_dtype(dtype: t.Type[numpy.complex128]) -> t.Type[numpy.float64]:
    ...

@t.overload
def to_real_dtype(dtype: DTypeLike) -> t.Type[numpy.floating]:
    ...

def to_real_dtype(dtype: DTypeLike) -> t.Type[numpy.floating]:
    """
    Convert a complex dtype to a plain float version.
    """

    dtype = numpy.dtype(dtype).type

    if not isinstance(dtype, type) or not issubclass(dtype, (numpy.floating, numpy.complexfloating)):
        raise TypeError("Non-floating point datatype")

    if issubclass(dtype, numpy.floating):
        return dtype

    try:
        return _REAL_MAP[dtype]
    except KeyError:
        raise TypeError(f"Unsupported datatype '{dtype}'") from None


@t.overload
def ifft2(a: t.Union[NDArray[numpy.complex_], NDArray[numpy.float_]]) -> NDArray[numpy.complex_]:
    ...

@t.overload
def ifft2(a: t.Union[NDArray[numpy.float64], NDArray[numpy.complex128]]) -> NDArray[numpy.complex128]:
    ...

@t.overload
def ifft2(a: t.Union[NDArray[numpy.float32], NDArray[numpy.complex64]]) -> NDArray[numpy.complex64]:
    ...

@t.overload
def ifft2(a: NDArray[NumT]) -> NDArray[numpy.complexfloating]:
    ...

@t.overload
def ifft2(a: ArrayLike) -> NDArray[numpy.complexfloating]:
    ...

def ifft2(a: ArrayLike) -> NDArray[numpy.complexfloating]:
    """
    Perform an inverse FFT on the last two axes of `a`.
    
    Follows our convention of centering real space and normalizing reciprocal space.
    """

    xp = get_array_module(a)
    return xp.fft.fftshift(xp.fft.ifft2(a, norm='forward'), axes=(-2, -1))

@t.overload
def fft2(a: t.Union[NDArray[numpy.complex_], NDArray[numpy.float_]]) -> NDArray[numpy.complex_]:
    ...

@t.overload
def fft2(a: t.Union[NDArray[numpy.float64], NDArray[numpy.complex128]]) -> NDArray[numpy.complex128]:
    ...

@t.overload
def fft2(a: t.Union[NDArray[numpy.float32], NDArray[numpy.complex64]]) -> NDArray[numpy.complex64]:
    ...

@t.overload
def fft2(a: NDArray[NumT]) -> NDArray[numpy.complexfloating]:
    ...

@t.overload
def fft2(a: ArrayLike) -> NDArray[numpy.complexfloating]:
    ...

def fft2(a: ArrayLike) -> NDArray[numpy.complex_]:
    """
    Perform a forward FFT on the last two axes of `a`.

    Follows our convention of centering real space and normalizing reciprocal space.
    """

    xp = get_array_module(a)
    return xp.fft.fft2(xp.fft.ifftshift(a, axes=(-2, -1)), norm='forward')


def split_array(arr: NDArray[DTypeT], axis: int = 0, *, keepdims: bool = False) -> t.Tuple[NDArray[DTypeT], ...]:
    """
    Split an array along `axis`, returning a tuple of subarrays.

    # Parameters

    - `arr`: Input array to split
    - `axis`: Axis to split array on. Default: 0
    - `keepdims`: Whether to keep the split dimension in the returned arrays.
    """
    xp = get_array_module(arr)
    arrs = xp.split(arr, arr.shape[axis], axis=axis)
    return tuple(arr) if keepdims else tuple(xp.squeeze(arr, axis) for arr in arrs)


@t.overload
def abs2(x: t.Union[NDArray[numpy.complex_], NDArray[numpy.float_]]) -> NDArray[numpy.float_]:
    ...

@t.overload
def abs2(x: t.Union[NDArray[numpy.float64], NDArray[numpy.complex128]]) -> NDArray[numpy.float64]:
    ...

@t.overload
def abs2(x: t.Union[NDArray[numpy.float32], NDArray[numpy.complex64]]) -> NDArray[numpy.float32]:
    ...

@t.overload
def abs2(x: t.Union[NDArray[numpy.complexfloating], NDArray[numpy.floating]]) -> NDArray[numpy.floating]:
    ...

@t.overload
def abs2(x: ArrayLike) -> NDArray[numpy.floating]:
    ...

def abs2(x: ArrayLike) -> NDArray[numpy.floating]:
    """
    Return the squared amplitude of a complex array.

    This is cheaper than `abs(x)**2.`
    """
    x = get_array_module(x).array(x)
    return x.real**2. + x.imag**2.


@dataclass(frozen=True, init=False)
class Sampling:
    shape: NDArray[numpy.int_]
    """Sampling shape (n_y, n_x)"""
    extent: NDArray[numpy.float_]
    """Sampling diameter (b, a)"""
    sampling: NDArray[numpy.float_]
    """Sample spacing (s_y, s_x)"""

    @property
    def k_max(self) -> NDArray[numpy.float_]:
        """
        Return maximum frequency (radius) of reciprocal space (1/(2s_y), 1/(2s_x))
        """
        return 1/(2 * self.sampling)

    @t.overload
    def __init__(self,
                 shape: t.Tuple[int, int], *,
                 extent: t.Tuple[float, float],
                 sampling: None = None):
        ...

    @t.overload
    def __init__(self,
                 shape: t.Tuple[int, int], *,
                 extent: None = None,
                 sampling: t.Tuple[float, float]):
        ...

    def __init__(self,
                 shape: ArrayLike, *,
                 extent: t.Optional[ArrayLike] = None,
                 sampling: t.Optional[ArrayLike] = None):
        try:
            object.__setattr__(self, 'shape', numpy.broadcast_to(shape, (2,)).astype(numpy.int_))
        except ValueError as e:
            raise ValueError(f"Expected a shape (n_y, n_x), instead got: {shape}") from e

        if extent is not None:
            try:
                object.__setattr__(self, 'extent', numpy.broadcast_to(extent, (2,)).astype(numpy.float_))
            except ValueError as e:
                raise ValueError(f"Expected an extent (b, a), instead got: {extent}") from e
            object.__setattr__(self, 'sampling', self.extent / self.shape)
        elif sampling is not None:
            try:
                object.__setattr__(self, 'sampling', numpy.broadcast_to(sampling, (2,)).astype(numpy.float_))
            except ValueError as e:
                raise ValueError(f"Expected a sampling (s_y, s_x), instead got: {sampling}") from e
            object.__setattr__(self, 'extent', self.sampling * self.shape)
        else:
            raise ValueError("Either 'extent' or 'sampling' must be specified")

    @t.overload
    def real_grid(self, *, dtype: t.Type[NumT], xp: t.Any = None) -> t.Tuple[NDArray[NumT], NDArray[NumT]]:
        ...

    @t.overload
    def real_grid(self, *, dtype: t.Optional[DTypeLike] = None, xp: t.Any = None) -> t.Tuple[NDArray[numpy.floating], NDArray[numpy.floating]]:
        ...

    def real_grid(self, *, dtype: t.Any = None, xp: t.Any = None) -> t.Tuple[NDArray[numpy.number], NDArray[numpy.number]]:
        """Return the realspace sampling grid `(yy, xx)`. Grid is centered around `(0, 0)` (but may not contain it!)"""
        if xp is None:
            xp = get_array_module(self.shape, self.extent, self.sampling)
        elif t.TYPE_CHECKING:
            xp = numpy

        if dtype is None:
            dtype = numpy.common_type(self.extent, self.sampling)

        hp = self.sampling/2.
        ys = xp.linspace(-self.extent[0]/2. + hp[0], self.extent[0]/2. + hp[0], self.shape[0], endpoint=False, dtype=dtype)
        xs = xp.linspace(-self.extent[0]/2. + hp[1], self.extent[1]/2. + hp[1], self.shape[1], endpoint=False, dtype=dtype)
        return tuple(xp.meshgrid(ys, xs, indexing='ij'))  # type: ignore

    @t.overload
    def recip_grid(self, *, centered: bool = False, dtype: t.Type[NumT], xp: t.Any = None) -> t.Tuple[NDArray[NumT], NDArray[NumT]]:
        ...

    @t.overload
    def recip_grid(self, *, centered: bool = False, dtype: t.Optional[DTypeLike] = None, xp: t.Any = None) -> t.Tuple[NDArray[numpy.floating], NDArray[numpy.floating]]:
        ...

    def recip_grid(self, *, centered: bool = False, dtype: t.Any = None, xp: t.Any = None) -> t.Tuple[NDArray[numpy.number], NDArray[numpy.number]]:
        """
        Return the reciprocal space sampling grid `(kyy, kxx)`.

        Unless `centered` is specified, the grid is fftshifted so the zero-frequency component is in the top left.
        """
        if xp is None:
            xp2 = get_array_module(self.shape, self.extent, self.sampling)
        elif not t.TYPE_CHECKING:
            xp2 = xp
        else:
            xp2 = numpy

        if dtype is None:
            dtype = numpy.common_type(self.extent, self.sampling)

        ky: NDArray[numpy.number] = xp2.fft.fftfreq(self.shape[0], dtype(self.sampling[0]))
        kx: NDArray[numpy.number] = xp2.fft.fftfreq(self.shape[1], dtype(self.sampling[1]))
        if centered:
            ky = xp2.fft.fftshift(ky)
            kx = xp2.fft.fftshift(kx)
        return tuple(xp2.meshgrid(ky, kx, indexing='ij'))  # type: ignore (missing overload)

    def bwlim(self, wavelength: float) -> float:
        """Return the bandwidth limit (in radians) for this sampling grid with the given wavelength."""
        return float(numpy.min(self.k_max) * 2./3. * wavelength)

    def mpl_real_extent(self, center: bool = False) -> t.Tuple[float, float, float, float]:
        """
        Return the extent of real space, for use in matplotlib.

        Extent is returned as `(left, right, bottom, top)`.
        If `center` is specified, samples correspond to the center of pixels.
        Otherwise (the default), they correspond to the corners of pixels.
        """
        # shift pixel corners to centers
        shift = -self.extent/2. -self.sampling/2. * int(center)
        return (shift[1], self.extent[1] + shift[1], self.extent[0] + shift[0], shift[0])

    def mpl_recip_extent(self, center: bool = True) -> t.Tuple[float, float, float, float]:
        """
        Return the extent of reciprocal space, for use in matplotlib.

        Extent is returned as `(left, right, bottom, top)`.
        If `center` is specified (the default), samples correspond to the center of pixels.
        Otherwise, they correspond to the corners of pixels.
        """
        kmax = self.k_max
        hp = 1/(2. * self.extent)
        # for odd sampling, grid is shifted by 1/2 pixel
        shift = hp * (self.shape % 2)  
        # shift pixel corners to centers
        if center:
            shift -= hp
        return (-kmax[1] + shift[1], kmax[1] + shift[1], kmax[0] + shift[0], -kmax[0] + shift[0])