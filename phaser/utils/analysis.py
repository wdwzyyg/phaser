from pathlib import Path
import typing as t

import numpy
from numpy.typing import ArrayLike, NDArray
import tifffile
from matplotlib import pyplot

from phaser.utils.num import fft2, ifft2, get_array_module
from phaser.utils.object import ObjectSampling
from phaser.utils.image import remove_linear_ramp, affine_transform
from phaser.state import ObjectState


def translation_matrix(vec: ArrayLike) -> NDArray[numpy.float64]:
    mat = numpy.eye(3)
    mat[:2, -1] = vec
    return mat


def scale_matrix(scale: ArrayLike) -> NDArray[numpy.float64]:
    return numpy.diag(numpy.concatenate([scale, [1.0]]))


def rotation_matrix(theta: float) -> NDArray[numpy.float64]:
    c, s = numpy.cos(theta), numpy.sin(theta)
    return numpy.array([
        [c, -s, 0.],
        [s, c, 0.],
        [0., 0., 1.],
    ])


def _upscale_matrix(
    object_sampling: ObjectSampling, ground_truth_sampling: ObjectSampling,
    theta: float = 0.0, shift: ArrayLike = 0.0,
) -> NDArray[numpy.floating]:
    # TODO: smarter alignment here

    return translation_matrix(object_sampling.shape / 2.) @ \
        scale_matrix(ground_truth_sampling.sampling / object_sampling.sampling) @ \
        rotation_matrix(-theta) @ translation_matrix(-numpy.array(ground_truth_sampling.shape) / 2. + numpy.array(shift))


def _cross_correlate(x: NDArray[numpy.floating], y: NDArray[numpy.floating], max_shift: float) -> t.Tuple[float, float]:
    xp = get_array_module(x, y)

    ky, kx = (xp.fft.fftfreq(s, 1/s) for s in x.shape)
    ky, kx = xp.meshgrid(ky, kx, indexing='ij')

    cross_corr = ifft2(fft2(x) * xp.conj(fft2(y))).real
    # limit shift to corner size
    cross_corr[ky**2 + kx**2 > max_shift**2] = numpy.nan

    max_i = xp.nanargmax(cross_corr)
    y = ky.ravel()[max_i]
    x = kx.ravel()[max_i]
    return (float(y), float(x))


def align_object_to_ground_truth(
    object: ObjectState,
    ground_truth: NDArray[numpy.floating],
    ground_truth_sampling: ArrayLike,
    rotation_angle: float = 0.0
) -> NDArray[numpy.floating]:
    """
    ground_truth: Ground truth phase (in radians/angstrom)
    """
    xp = get_array_module(object.data, ground_truth)
    theta = rotation_angle * numpy.pi/180.

    ground_truth_samp = ObjectSampling(
        t.cast(t.Tuple[int, int], ground_truth.shape),
        sampling=ground_truth_sampling, corner=[0., 0.]
    )

    object_sampling = object.sampling
    object_roi = object_sampling.get_region_mask(xp=xp)

    max_shift = float(numpy.min(
        (ground_truth_samp.extent - object_sampling.get_region_extent()) / (3.0 * ground_truth_samp.sampling)
    ))
    if max_shift < 0:
        raise ValueError("Error: Ground truth extent smaller than object extent")

    print(f"max_shift: {max_shift:.3f} px ({max_shift * ground_truth_samp.sampling[0]:.3f} angstrom)")

    # convert to radians/angstrom
    object_phase = remove_linear_ramp(
        xp.angle(object.data) / object.thicknesses[:, None, None],
        object_roi
    )
    # and average
    object_mean = xp.mean(object_phase, axis=0)

    # upscale object to ground truth
    matrix = _upscale_matrix(object_sampling, ground_truth_samp, theta=theta)

    upsamp_obj = affine_transform(object_mean * object_roi, matrix, output_shape=ground_truth.shape)

    shift = _cross_correlate(ground_truth, upsamp_obj, max_shift)

    # perform final upsampling, and mask with nan
    matrix = _upscale_matrix(object_sampling, ground_truth_samp, theta=theta, shift=shift)
    upsamp_obj = affine_transform(
        numpy.where(object_roi, object_phase, numpy.nan),
        matrix, output_shape=ground_truth.shape, cval=numpy.nan
    )

    return upsamp_obj