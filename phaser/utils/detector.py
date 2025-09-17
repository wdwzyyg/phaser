"""
Detector utilities
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import repeat
import warnings
import typing as t

import numpy
from numpy.typing import ArrayLike, DTypeLike, NDArray
from typing_extensions import Self

from .num import get_array_module, cast_array_module, is_torch, to_real_dtype, as_numpy, at
from .num import as_array, is_cupy, is_jax, NumT, ComplexT, DTypeT
from .tree import tree_dataclass
from .misc import create_rng
from ..hooks.mtf import DetectorMtf


def resample_1D_mtf(mtf: DetectorMtf, bin_factor):
    
    nu = mtf['inverse_pixel']
    M_native = mtf['values']
    
    ratio = numpy.sinc(bin_factor*nu) / numpy.maximum(numpy.sinc(nu), 1e-12)
    
    M_bin = M_native * ratio

    # Keep only valid baseband for binned detector
    mask = nu <= 0.5 / bin_factor
    nu_binned = nu[mask] / bin_factor     # rescale to cycles per binned pixel
    M_binned = M_bin[mask]
    
    return DetectorMtf(inverse_pixel=nu_binned,values=M_binned)


def calc_2D_mtf(mtf: DetectorMtf, detector_shape: tuple, bin_factor: t.Optional[int] = 1):

    resampled_mtf = resample_1D_mtf(mtf, bin_factor)

    rows = detector_shape[0]
    cols = detector_shape[1]

    freq_x = numpy.fft.fftfreq(cols, 1)
    freq_y = numpy.fft.fftfreq(rows, 1)

    # Create a 2D grid of frequencies
    freq_grid_x, freq_grid_y = numpy.meshgrid(freq_x, freq_y)

    k2 = numpy.sqrt(freq_grid_y**2+freq_grid_x**2)

    mtf2d = numpy.interp(k2, mtf['inverse_pixel'], mtf['values'])

    return mtf2d


def apply_mtf(intensities, mtf:DetectorMtf):
    
    return intensities
