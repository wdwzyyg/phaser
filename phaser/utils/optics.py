"""
Probe/optics utilities
"""

import typing as t

import numpy
from numpy.typing import NDArray, ArrayLike

from .num import get_array_module, ifft2, abs2, NumT, ufunc_outer, is_jax, cast_array_module
from .num import Sampling, to_complex_dtype, to_real_dtype, split_array, to_numpy


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


def make_hermetian_modes(base_probe: NDArray[NumT], n_modes: int, powers: ArrayLike = 0.05) -> NDArray[NumT]:
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

    # we make modes like the following:
    #       0 1 2 3 4
    #     +----------
    #   0 | 0 2 5   |
    #   1 | 1 4 8   |
    #   2 | 3 7     |
    #   3 | 6       |
    #   4 +----------

    # the diagonal each mode is on (indexing starting from 0):
    i = numpy.arange(n_modes)
    diag_num = numpy.ceil((numpy.sqrt(8*i + 9) - 3) / 2).astype(numpy.int_)

    # and the corresponding grid indices
    xx = (i - diag_num*(diag_num+1)/2).astype(numpy.int_)
    yy = diag_num - xx

    max_n = int(numpy.max(yy)) + 1

    modes = hermetian_modes(base_probe, max_n, max_n)[yy, xx]
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
            #if y_power > 0 or x_power > 0:
            #    mode = mode * xp.exp(-xx**2./(2 * var_x) - yy*2./(2 * var_y))
            #    mode /= xp.sqrt(xp.sum(abs2(mode)))

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
    return xp.exp(-1.j * numpy.pi * delta_z * (wavelength * k2 - 2.*(kx*tiltx + ky*tilty))) \
        .astype(to_complex_dtype(k2.dtype))



def estimate_probe_radius(wavelength: float, aperture: float, defocus: float, *,
                          threshold: t.Union[t.Literal['geom'], float] = 0.9, xp: t.Any = None) -> float:
    """
    Estimate the radius of a probe created by a circular aperture of
    maximum semi-angle `aperture` (in mrad) defocused by `defocus` (in length units).

    `threshold` is used to determine the intensity cutoff for size calcuation.
    'geom' may be specified to use the geometric approximation (valid for large defocuses).
    """
    xp2 = numpy if xp is None else cast_array_module(xp)

    aperture *= 1e-3  # mrad -> rad

    if threshold == 'geom':
        return defocus * aperture

    rel_defocus = numpy.abs(defocus) / wavelength

    # 4*nyquist
    sampling = 1/(8 * aperture)
    min_box_size = 5. * rel_defocus * aperture
    # number of points required is size / sampling
    # we round up to nearest power of 2
    n = max(512, int(numpy.exp2(numpy.ceil(numpy.log2(min_box_size / sampling)))))
    if n > 2**14:
        raise ValueError("Can't calculate probe radii (requires too much phase space)") from None

    # TODO: this should be possible in 1D, using a Henkel transform or similar
    # also, we can cache a lot of this

    samp = Sampling((n, n), sampling=(sampling, sampling))
    ky, kx = samp.recip_grid(xp=xp2)
    yy, xx = samp.real_grid(xp=xp2)

    probe_int = abs2(make_focused_probe(ky, kx, 1.0, aperture*1e3, defocus=rel_defocus))

    rs = xp2.sqrt(yy**2 + xx**2)
    r_samp = samp.sampling[0] / 3.
    r_i = xp2.floor(rs / r_samp).astype(int)

    probe_ints = xp2.bincount(r_i.ravel(), weights=probe_int.ravel())
    cum_ints = to_numpy(xp2.cumsum(probe_ints))

    try:
        probe_n = int(numpy.argwhere(cum_ints > float(threshold))[0, 0])
        # redimensionalize
        return float(probe_n * r_samp * wavelength)
    except IndexError:
        raise ValueError("Couldn't calculate probe radii (didn't reach threshold, is probe too big for calculation box?)") from None


def calc_metrics(*,
    wavelength: t.Optional[float] = None, kv: t.Optional[float] = None,
    conv_angle: float, defocus: float, scan_step: float, diff_step: float,
    threshold: t.Union[t.Literal['geom'], float] = 0.9, xp: t.Any = None,
) -> t.Dict[str, float]:
    """
    Calculate sampling metrics for the given parameters. Units are as follows:

    wavelength, scan_step, defocus: length units (must be consistent)
    conv_angle, diff_step: mrad
    """

    if wavelength is None:
        if kv is None:
            raise ValueError("One of 'wavelength' or 'kv' must be specified")
        from phaser.utils.physics import Electron
        wavelength = Electron(kv * 1e3).wavelength

    probe_radius = estimate_probe_radius(
        wavelength=wavelength, aperture=conv_angle,
        defocus=defocus, threshold=threshold, xp=xp
    )

    # diff_step in 1/A
    d_step = diff_step*1e-3 / wavelength

    return {
        'probe_radius': probe_radius,
        'fund_samp': 1/(scan_step * d_step),
        'probe_samp': 1/(2.*probe_radius * d_step),
        'linear_oversamp': 2.*probe_radius / scan_step,
        'areal_oversamp': numpy.pi*probe_radius**2 / scan_step**2,
        'ronchi_mag': conv_angle / (diff_step * probe_radius),

        'wavelength': wavelength,
        'conv_angle': conv_angle,
        'defocus': defocus,
        'scan_step': scan_step,
        'diff_step': diff_step,
    }