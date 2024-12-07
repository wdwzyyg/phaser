"""
Probe/optics utilities
"""

import typing as t

import numpy
from numpy.typing import NDArray, ArrayLike

from .num import get_array_module, ifft2, abs2, NumT, ufunc_outer, is_jax
from .num import to_complex_dtype, to_real_dtype, split_array


@t.overload
def make_focused_probe(ky: NDArray[numpy.float64], kx: NDArray[numpy.float64], wavelength: float,
                       aperture: float, *, defocus: float = 0.) -> NDArray[numpy.complex128]:
    ...

@t.overload
def make_focused_probe(ky: NDArray[numpy.float32], kx: NDArray[numpy.float32], wavelength: float,
                       aperture: float, *, defocus: float = 0.) -> NDArray[numpy.complex64]:
    ...

@t.overload
def make_focused_probe(ky: NDArray[numpy.floating], kx: NDArray[numpy.floating], wavelength: float,
                       aperture: float, *, defocus: float = 0.) -> NDArray[numpy.complexfloating]:
    ...

def make_focused_probe(ky: NDArray[numpy.floating], kx: NDArray[numpy.floating], wavelength: float,
                       aperture: float, *, defocus: float = 0.) -> NDArray[numpy.complexfloating]:
    """
    Create a focused probe from a circular aperture of semi-angle `aperture` (in mrad).

    Optionally, give it some defocus (positive corresponds to overfocus).
    """
    xp = get_array_module(ky, kx)

    thetay, thetax = ky * wavelength, kx * wavelength
    theta2 = thetay**2. + thetax**2.

    phase = (defocus/(2. * wavelength)) * theta2
    probe = xp.exp(-2.j*numpy.pi * phase)

    mask = theta2 <= (aperture * 1e-3)**2.
    probe *= mask

    # normalize intensity of probe
    probe /= xp.sqrt(xp.sum(abs2(probe)))

    return ifft2(probe)


def make_hermetian_modes(base_probe: NDArray[NumT], n_modes: int, powers: ArrayLike = 0.02) -> NDArray[NumT]:
    """
    Create Hermitian-Gauss probe modes based on `base_probe`.

    Creates `n_modes` modes. Additional probe modes are given intensities
    according to `powers`, which may be a list.
    """
    xp = get_array_module(base_probe)

    # TODO clean this up
    powers = numpy.array(powers, dtype=to_real_dtype(base_probe.dtype)).ravel()
    powers = numpy.pad(powers, (0, n_modes - len(powers) - 1), mode='edge')[:n_modes - 1]
    base_power = 1. - numpy.sum(powers)
    powers = numpy.concatenate(([base_power], powers))

    n_y = numpy.ceil(numpy.sqrt(n_modes)).astype(numpy.int_)
    n_x = numpy.ceil(n_modes / (n_y + 1)).astype(numpy.int_)

    modes = hermetian_modes(base_probe, n_y, n_x).reshape((-1, *base_probe.shape[-2:]))[:n_modes]
    modes *= xp.array(numpy.sqrt(powers)[:, None, None], dtype=modes.dtype)
    return t.cast(NDArray[NumT], modes)


def hermetian_modes(base_probe: NDArray[NumT], n_y: int, n_x: int) -> NDArray[NumT]:
    """
    Create a grid of Hermetian-Gauss modes, n_y in y direction and n_x in x direction.

    Return a ndarray of shape `(n_y, n_x, *base_probe.shape)`. Each mode is orthogonal
    and normalized in reciprocal space.
    """
    xp = get_array_module(base_probe)
    real_dtype = to_real_dtype(base_probe.dtype)

    (yy, xx) = xp.indices(base_probe.shape, dtype=real_dtype)

    base_probe_mag = abs2(base_probe)

    (com_y, com_x) = (xp.sum(a * base_probe_mag) / xp.sum(base_probe_mag) for a in (yy, xx))
    yy -= com_y
    xx -= com_x
    (var_y, var_x) = (xp.sum(a**2. * base_probe_mag) / xp.sum(base_probe_mag) for a in (yy, xx))

    modes = xp.empty((n_y * n_x, *base_probe.shape), dtype=base_probe.dtype)

    i = 0
    for y_power in range(n_y):
        for x_power in range(n_x):
            mode = yy**y_power * xx**x_power * base_probe
            if y_power > 1 or x_power > 1:
                mode = mode * xp.exp(-xx**2./(2 * var_x) - yy*2./(2 * var_y))
                mode /= xp.sqrt(xp.sum(abs2(mode)))

            # orthogonalize to other modes
            for prev_i in range(i):  # TODO do this in a smarter way
                mode -= modes[prev_i] * xp.sum(modes[prev_i] * xp.conj(mode))
            # renormalize
            mode /= xp.sqrt(xp.sum(abs2(mode)))

            if is_jax(modes) and not t.TYPE_CHECKING:
                modes = modes.at[i].set(mode)
            else:
                modes[i] = mode
            i += 1

    return modes.reshape((n_y, n_x, *base_probe.shape))


@t.overload
def fourier_shift_filter(ky: NDArray[numpy.float64], kx: NDArray[numpy.float64], shifts: ArrayLike) -> NDArray[numpy.complex128]:
    ...

@t.overload
def fourier_shift_filter(ky: NDArray[numpy.float32], kx: NDArray[numpy.float32], shifts: ArrayLike) -> NDArray[numpy.complex64]:
    ...

@t.overload
def fourier_shift_filter(ky: NDArray[numpy.floating], kx: NDArray[numpy.floating], shifts: ArrayLike) -> NDArray[numpy.complexfloating]:
    ...

def fourier_shift_filter(ky: NDArray[numpy.floating], kx: NDArray[numpy.floating], shifts: ArrayLike) -> NDArray[numpy.complexfloating]:
    """
    Create a phase ramp / Fourier shift filter, using the reciprocal space frequencies `ky` & `kx`.

    A frequency-shift filter can be created using `fourier_shift_filter(yy, xx, -shifts)`.

    # Parameters:

     - `ky`, `kx`: Frequency grid filter is created with
     - `shifts`: Vector(s) to shift by. Should be an array of shape `(..., 2)`, with the last dimension
       representing `(y, x)` coordinates.

    Returns a ndarray of shape `(*shifts.shape[:-1], *ky.shape)`
    """
    xp = get_array_module(ky, kx)
    dtype = to_complex_dtype(ky.dtype)

    (y, x) = split_array(xp.array(shifts, dtype=ky.dtype), axis=-1)

    return xp.exp(xp.array(-2.j*numpy.pi, dtype=dtype) * (ufunc_outer(xp.multiply, x, kx) + ufunc_outer(xp.multiply, y, ky)))


@t.overload
def fresnel_propagator(ky: NDArray[numpy.float64], kx: NDArray[numpy.float64], wavelength: float,
                       delta_z: float, tilt: t.Tuple[float, float] = (0., 0.)) -> NDArray[numpy.complex128]:
    ...

@t.overload
def fresnel_propagator(ky: NDArray[numpy.float32], kx: NDArray[numpy.float32], wavelength: float,
                       delta_z: float, tilt: t.Tuple[float, float] = (0., 0.)) -> NDArray[numpy.complex64]:
    ...


@t.overload
def fresnel_propagator(ky: NDArray[numpy.floating], kx: NDArray[numpy.floating], wavelength: float,
                       delta_z: float, tilt: t.Tuple[float, float] = (0., 0.)) -> NDArray[numpy.complexfloating]:
    ...


def fresnel_propagator(ky: NDArray[numpy.floating], kx: NDArray[numpy.floating], wavelength: float,
                       delta_z: float, tilt: t.Tuple[float, float] = (0., 0.)) -> NDArray[numpy.complexfloating]:
    """
    Return a Fresnel diffraction filter in frequency space, for use in free-space propagation. Roughly taken from Kirkland [1].

    # Parameters

     - `ky`, `kx`: Frequency grid filter is created with
     - `wavelength`: Wavelength
     - `delta_z`: Distance to propagate by
     - `tilt`: `(x, y)` mistilt to apply (in mrad). 

    [1]. Kirkland, E. J. Advanced Computing in Electron Microscopy. (Springer US, Boston, MA, 2010). doi:10.1007/978-1-4419-6533-2.

    """
    xp = get_array_module(ky, kx)

    (tiltx, tilty) = numpy.tan(tilt[0]*1e-3), numpy.tan(tilt[1]*1e-3)

    k2 = ky**2 + kx**2
    return xp.exp(-1.j * numpy.pi * delta_z * (wavelength * k2 - 2.*(kx*tiltx + ky*tilty)))

