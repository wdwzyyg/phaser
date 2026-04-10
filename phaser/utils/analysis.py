import typing as t

import numpy
from numpy.typing import ArrayLike, NDArray
import warnings

from phaser.utils.num import fft2, ifft2, abs2, Sampling, get_array_module, at, to_numpy, NumT
from phaser.utils.object import ObjectSampling
from phaser.utils.image import remove_linear_ramp, translation_matrix, rotation_matrix
from phaser.state import ObjectState
from phaser.types import cast_length


def split_image(img: NDArray[NumT]) -> t.Tuple[NDArray[NumT], NDArray[NumT], NDArray[NumT], NDArray[NumT]]:
    """
    Split image into 4 subimages (upper left, upper right, lower left, lower right).
    """
    shape = img.shape[-2:]
    # crop image divisible by two
    shape = tuple(s - s%2 for s in shape)
    img = img[(Ellipsis, *(slice(0, s) for s in shape))]

    return cast_length((
        img[(Ellipsis, slice(row_start, shape[0], 2), slice(col_start, shape[1], 2))]
        for row_start in (0, 1)
        for col_start in (0, 1)
    ), 4)


def fourier_correlate(img1: NDArray[NumT], img2: NDArray[NumT]) -> NDArray[numpy.complex128]:
    assert img1.shape == img2.shape
    from skimage.filters import window

    xp = get_array_module(img1, img2)

    win = xp.asarray(window('hann', img1.shape))
    img1_fft = numpy.fft.fftshift(fft2(img1.astype(numpy.float64) * win), axes=(-2, -1))
    img2_fft = xp.conj(numpy.fft.fftshift(fft2(img2.astype(numpy.float64) * win), axes=(-2, -1)))
    fft1_mag = abs2(img1_fft)
    fft2_mag = abs2(img2_fft)

    with xp.errstate(invalid='ignore'):
        corr = img1_fft * img2_fft / xp.sqrt(fft1_mag * fft2_mag)
    return xp.nan_to_num(corr, nan=1.)


def _calc_r_binning(
    shape: t.Tuple[int, ...], r_spacing: float = 1.0,
    xp: t.Any = None
) -> NDArray[numpy.int64]:
    if xp is None or t.TYPE_CHECKING:
        xp = numpy

    c_y, c_x = tuple(int(s//2) for s in shape[-2:])
    y, x = xp.indices(shape[-2:])
    r = xp.sqrt((y - c_y)**2 + (x - c_x)**2)
    return xp.floor(r / r_spacing).astype(numpy.int64).ravel()


def contrast_transfer(
    img1: NDArray[NumT], img2: NDArray[NumT], *,
    inscribed: bool = True, r_spacing: float = 1.0
) -> t.Tuple[NDArray[numpy.float64], NDArray[numpy.float64], NDArray[numpy.float64]]:
    assert img1.shape == img2.shape
    from skimage.filters import window

    xp = get_array_module(img1, img2)

    win = xp.asarray(window(('tukey', 0.3), img1.shape), dtype=numpy.float64)
    img1_fft = xp.fft.fftshift(fft2(img1.astype(numpy.float64) * win), axes=(-2, -1))
    img2_fft = xp.conj(xp.fft.fftshift(fft2(img2.astype(numpy.float64) * win), axes=(-2, -1)))
    fft1_mag = abs2(img1_fft)
    fft2_mag = abs2(img2_fft)

    r_i = _calc_r_binning(img1.shape, r_spacing, xp=xp)
    mag1_count = xp.bincount(r_i, fft1_mag.ravel())
    mag2_count = xp.bincount(r_i, fft2_mag.ravel())

    with numpy.errstate(invalid='ignore', divide='ignore'):
        vals = to_numpy(xp.sqrt(mag1_count / mag2_count))
    vals = numpy.nan_to_num(vals, nan=1.)

    rs = numpy.linspace(0., len(vals) * r_spacing, len(vals), endpoint=False, dtype=numpy.float64)
    freq = rs / numpy.sqrt(numpy.prod(img1.shape[-2:]))

    if inscribed:
        # crop to inner radii
        n_r = int(numpy.floor(min(img1.shape[-2:]) / (2.*r_spacing)))
        return (vals[:n_r], rs[:n_r], freq[:n_r])
    return (vals, rs, freq)


def fourier_ring_correlate(
    img1: NDArray[NumT], img2: NDArray[NumT], *,
    inscribed: bool = True, r_spacing: float = 1.0
) -> t.Tuple[NDArray[numpy.float64], NDArray[numpy.float64], NDArray[numpy.float64]]:
    assert img1.shape == img2.shape
    from skimage.filters import window

    xp = get_array_module(img1, img2)

    win = xp.asarray(window(('tukey', 0.3), img1.shape), dtype=numpy.float64)
    img1_fft = xp.fft.fftshift(fft2(img1.astype(numpy.float64) * win), axes=(-2, -1))
    img2_fft = xp.conj(xp.fft.fftshift(fft2(img2.astype(numpy.float64) * win), axes=(-2, -1)))
    fft1_mag = abs2(img1_fft)
    fft2_mag = abs2(img2_fft)

    r_i = _calc_r_binning(img1.shape, r_spacing, xp=xp)
    # sum real and imaginary separately
    real_count = xp.bincount(r_i, (img1_fft * xp.conj(img2_fft)).real.ravel())
    mag1_count = xp.bincount(r_i, fft1_mag.ravel())
    mag2_count = xp.bincount(r_i, fft2_mag.ravel())

    with numpy.errstate(invalid='ignore', divide='ignore'):
        vals = to_numpy(real_count / xp.sqrt(mag1_count * mag2_count))
    vals = numpy.nan_to_num(vals, posinf=0., nan=1.)

    #vals = numpy.bincount(r_i.ravel(), corr.ravel()) / numpy.bincount(r_i.ravel())
    rs = numpy.linspace(0., len(vals) * r_spacing, len(vals), endpoint=False, dtype=numpy.float64)
    freq = rs / numpy.sqrt(numpy.prod(img1.shape[-2:]))

    if inscribed:
        # crop to inner radii
        n_r = int(numpy.floor(min(img1.shape[-2:]) / (2.*r_spacing)))
        return (vals[:n_r], rs[:n_r], freq[:n_r])
    return (vals, rs, freq)


def frc_intersect_threshold(frc: NDArray[numpy.floating], freq: NDArray[numpy.floating], threshold: float) -> t.Tuple[float, float]:
    diff = frc - threshold
    lastdiff = numpy.roll(diff, shift=1)
    # find negative zero crossing
    try:
        i = numpy.nonzero((diff[1:] < 0) & (lastdiff[1:] > 0))[0][0]
    except IndexError:
        raise ValueError("No crossing found when evaluating FRC resolution")

    m = (diff[i+1] - diff[i]) / (freq[i+1] - freq[i])
    m_f = (frc[i+1] - frc[i]) / (freq[i+1] - freq[i])
    x_d = diff[i]/m
    return (float(freq[i] - x_d), float(frc[i] - x_d * m_f))


def _cross_correlate(x: NDArray[numpy.floating], y: NDArray[numpy.floating], max_shift: float) -> t.Tuple[float, float]:
    xp = get_array_module(x, y)

    samp = Sampling(tuple(x.shape), sampling=(1.0, 1.0))  # type: ignore
    yy, xx = samp.real_grid(xp=xp)

    cross_corr = ifft2(fft2(x) * xp.conj(fft2(y))).real
    # limit maximum shift
    cross_corr = at(cross_corr, yy**2 + xx**2 > max_shift**2).set(numpy.nan)  # type: ignore

    max_i = xp.nanargmax(cross_corr)
    y = yy.ravel()[max_i]
    x = xx.ravel()[max_i]
    return (float(y), float(x))

@t.overload
def align_object_to_ground_truth(
    object: ObjectState,
    ground_truth: NDArray[numpy.floating],
    ground_truth_sampling: ArrayLike, *,
    rotation_angle: float = 0.0,
    refinement_niter: int = 0,
    order: int = 1,
    return_crop: t.Literal[False] = False,
) -> t.Tuple[NDArray[numpy.floating], NDArray[numpy.floating]]:
    ...

@t.overload
def align_object_to_ground_truth(
    object: ObjectState,
    ground_truth: NDArray[numpy.floating],
    ground_truth_sampling: ArrayLike, *,
    rotation_angle: float = 0.0,
    refinement_niter: int = 0,
    order: int = 1,
    return_crop: t.Literal[True],
) -> t.Tuple[NDArray[numpy.floating], NDArray[numpy.floating], t.Tuple[slice, slice]]:
    ...

def align_object_to_ground_truth(
    object: ObjectState,
    ground_truth: NDArray[numpy.floating],
    ground_truth_sampling: ArrayLike, *,
    rotation_angle: float = 0.0,
    refinement_niter: int = 0,
    order: int = 1,
    return_crop: bool = False,
) -> t.Union[t.Tuple[NDArray[numpy.floating], NDArray[numpy.floating]], t.Tuple[NDArray[numpy.floating], NDArray[numpy.floating], t.Tuple[slice, slice]]]:
    """
    ground_truth: Ground truth phase (in radians, or radians/angstrom for multislice data)

    Returns a tuple (object, ground_truth)
    """
    xp = get_array_module(object.data, ground_truth)
    ground_truth = xp.asarray(ground_truth)
    object_phase = xp.angle(xp.asarray(object.data))

    # normalize multislice objects to radians/angstrom
    if len(object.thicknesses):
        object_phase /= object.thicknesses[:, None, None]

    # remove linear ramp
    object_roi = object.sampling.get_region_mask(xp=xp)
    object_phase = remove_linear_ramp(object_phase, object_roi)
    object_phase -= xp.nanquantile(object_phase[..., object_roi], 0.01, axis=-1)[:, None, None]
    # and get average
    object_mean = xp.mean(object_phase, axis=0)

    # initially center ground truth on object roi
    obj_center = object.sampling.get_region_center()
    ground_truth_corner = obj_center - numpy.array(ground_truth.shape) * ground_truth_sampling / 2.
    ground_truth_samp = ObjectSampling(
        t.cast(t.Tuple[int, int], ground_truth.shape),
        sampling=ground_truth_sampling, corner=ground_truth_corner,
        region_min=object.sampling.region_min, region_max=object.sampling.region_max,
    )
    crop = ground_truth_samp.get_region_crop(pad=-0.05 * object.sampling.get_region_extent())

    # perform initial resampling
    upsamp_obj = object.sampling.resample(object_mean, ground_truth_samp, cval=0., rotation=rotation_angle, order=order)

    # cross correlate
    max_shift = float(numpy.min(
        (ground_truth_samp.extent - object.sampling.get_region_extent()) / (3.0 * ground_truth_samp.sampling)
    ))
    if max_shift < 0:
        raise ValueError("Error: Ground truth extent smaller than object extent")
    elif max_shift * ground_truth_samp.sampling[0] < 10.0:  # 10 A = 1 nm
        warnings.warn(
            "Ground truth extent is only slightly larger than the object extent, which might not be enough for cross-correlation to find the best alignment.\n"
            f"Maximum shift is limited to {max_shift:.3f} px ({max_shift * ground_truth_samp.sampling[0]:.3f} angstrom)"
        )

    shift = numpy.array(_cross_correlate(upsamp_obj[tuple(crop)], ground_truth[tuple(crop)], max_shift))

    # shift ground truth to match experiment
    ground_truth_samp = ObjectSampling(
        tuple(ground_truth_samp.shape),
        sampling=ground_truth_sampling, corner=ground_truth_corner + shift * ground_truth_sampling,
        region_min=object.sampling.region_min, region_max=object.sampling.region_max, 
    )
    crop = ground_truth_samp.get_region_crop()

    if refinement_niter == 0:
        # perform final upsampling
        upsamp_obj = object.sampling.resample(object_phase, ground_truth_samp, cval=0., rotation=rotation_angle, order=order)
        if return_crop:
            return upsamp_obj[(slice(None), *crop)], ground_truth[tuple(crop)], crop
        return upsamp_obj[(slice(None), *crop)], ground_truth[tuple(crop)]

    import scipy.optimize

    def _make_affine(mat: NDArray[numpy.floating]) -> NDArray[numpy.floating]:
        (a, b, c, d, e, f) = mat
        affine = translation_matrix(obj_center) @ numpy.array([
            [a, b, e],
            [c, d, f],
            [0., 0., 1],
        ]) @ rotation_matrix(rotation_angle) @ translation_matrix(-obj_center)
        return affine

    def align_and_correlate(mat: NDArray[numpy.floating]) -> NDArray[numpy.floating]:
        affine = _make_affine(mat)
        upsamp_obj = object.sampling.resample(object_mean, ground_truth_samp, cval=0., affine=affine, order=order)
        return to_numpy(upsamp_obj[tuple(crop)] - ground_truth[tuple(crop)]).ravel()

    max_shift_refine = max_shift * ground_truth_samp.sampling[0] / 5.

    # (a, b, c, d, e, f)
    init_mat = numpy.array([1., 0., 0., 1., 0., 0.])
    min_bound = numpy.array([0.9, -0.1, -0.1, 0.9, -max_shift_refine, -max_shift_refine])
    max_bound = numpy.array([1.1, 0.1, 0.1, 1.1, max_shift_refine, max_shift_refine])

    result = scipy.optimize.least_squares(align_and_correlate, init_mat, bounds=(min_bound, max_bound),
                                          method='dogbox', max_nfev=refinement_niter, xtol=1e-4)
    print(f"""\
Refinement result: {result.message}
    nfev: {result.nfev}
    matrix: {result.x}""")

    affine = _make_affine(result.x)
    (a, b, c, d, e, f) = result.x
    affine = translation_matrix(obj_center) @ numpy.array([
        [a, b, e],
        [c, d, f],
        [0., 0., 1],
    ]) @ rotation_matrix(rotation_angle) @ translation_matrix(-obj_center)

    # perform final upsampling
    upsamp_obj = object.sampling.resample(object_phase, ground_truth_samp, cval=0., affine=affine, order=order)

    if return_crop:
        return upsamp_obj[(slice(None), *crop)], ground_truth[tuple(crop)], crop
    return upsamp_obj[(slice(None), *crop)], ground_truth[tuple(crop)]


def _uniform_filter_spatial(im, size: int, xp: t.Any):
    """
    Separable box filter over the last two spatial dims only (any ndim >= 2).
    Accepts stacked inputs e.g. (N, H, W), filtering H and W only — enabling
    fused multi-statistic computation in one call.

    Dispatches to:
    - scipy.ndimage.uniform_filter        for numpy
    - cupyx.scipy.ndimage.uniform_filter  for cupy (GPU-native)
    - cumsum-based separable filter       for JAX / other backends (XLA-friendly)
    """
    xp_name = getattr(xp, '__name__', '')
    sizes = [1] * (im.ndim - 2) + [size, size]

    if xp_name == 'numpy':
        from scipy.ndimage import uniform_filter
        return uniform_filter(im, sizes)

    if 'cupy' in xp_name:
        from cupyx.scipy.ndimage import uniform_filter
        return uniform_filter(im, sizes)

    # JAX or other: cumsum box filter along axes -2 and -1 only (XLA-friendly)
    def _along_axis(arr, axis: int):
        pad = size // 2
        pad_config = [(0, 0)] * arr.ndim
        pad_config[axis] = (pad, pad)
        padded = xp.pad(arr, pad_config, mode='reflect')
        zero_shape = list(padded.shape)
        zero_shape[axis] = 1
        cs = xp.concatenate(
            [xp.zeros(zero_shape, dtype=padded.dtype), xp.cumsum(padded, axis=axis)],
            axis=axis,
        )
        n = arr.shape[axis]
        sl_end = [slice(None)] * arr.ndim
        sl_end[axis] = slice(size, size + n)
        sl_beg = [slice(None)] * arr.ndim
        sl_beg[axis] = slice(0, n)
        return (cs[tuple(sl_end)] - cs[tuple(sl_beg)]) / size

    return _along_axis(_along_axis(im, -2), -1)


def structural_similarity(
    im1,
    im2,
    data_range=None,
    win_size: int = 3,
    num_scales: int = 3,
    **kwargs,
) -> float:
    """
    Multi-scale contrast-structure similarity (geometric mean across scales).

    Computes the contrast-structure (CS) component of SSIM at each scale of a
    bilinear downsampling pyramid, then combines as a geometric mean:
        result = (cs_1 * cs_2 * ... * cs_N)^(1/N)

    Luminance is omitted. Equal scale weights are used.

    Efficient implementation:
      - fused filter pass: all statistics filtered in one call per scale
      - bilinear downsampling pyramid via affine_transform
      - fully on-device: only the final scalar crosses the device boundary

    Parameters
    ----------
    im1, im2 : ndarray
        Arrays from any supported backend (numpy, JAX, cupy).
    data_range : float, optional
        Computed from im2 if not provided.
    win_size : int
        Box filter size in pixels (default 3).
    num_scales : int
        Number of pyramid levels (default 3).

    Returns
    -------
    mssim : float
        MS-SSIM value in [0, 1].
    """
    from phaser.utils.image import affine_transform as _affine_transform

    def _resample(im, target_shape):
        scale_y = im.shape[-2] / target_shape[-2]
        scale_x = im.shape[-1] / target_shape[-1]
        matrix = numpy.array([[scale_y, 0.0], [0.0, scale_x]])
        offset = numpy.array([0.5 * (scale_y - 1.0), 0.5 * (scale_x - 1.0)])
        return _affine_transform(im, matrix, offset=offset, output_shape=target_shape[-2:], order=1)

    xp = get_array_module(im1, im2)

    im1 = im1.astype(numpy.float64)
    im2 = im2.astype(numpy.float64)

    if im1.shape != im2.shape:
        im2 = _resample(im2, im1.shape)
    if data_range is None:
        data_range = float(im2.max() - im2.min())

    C2 = (0.03 * data_range) ** 2

    pad = (win_size - 1) // 2
    weight = 1.0 / num_scales

    mssim = 1.0
    for scale in range(num_scales):
        if min(im1.shape[-2:]) < win_size:
            break

        # fused: stack [im1, im2, im1², im2², im1·im2] and filter in one pass
        stacked = xp.stack([im1, im2, im1 * im1, im2 * im2, im1 * im2])
        f = _uniform_filter_spatial(stacked, win_size, xp)
        ux, uy, uxx, uyy, uxy = f[0], f[1], f[2], f[3], f[4]

        vx = uxx - ux * ux
        vy = uyy - uy * uy
        vxy = uxy - ux * uy

        # crop boundary artifacts
        s = (slice(pad, -pad), slice(pad, -pad))
        vx, vy, vxy = vx[s], vy[s], vxy[s]

        cs = float(xp.mean((2 * vxy + C2) / (vx + vy + C2)))
        mssim *= cs ** weight

        if scale < num_scales - 1:
            new_shape = (im1.shape[0], im1.shape[-2] // 2, im1.shape[-1] // 2)
            im1 = _resample(im1, new_shape)
            im2 = _resample(im2, new_shape)

    return mssim