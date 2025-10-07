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
from .num import Sampling


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


def gaussian2d_mtf(sampling:Sampling, sigma:numpy.floating, norm):
    ky = numpy.fft.fftfreq(sampling.shape[0], sampling.sampling[0])
    kx = numpy.fft.fftfreq(sampling.shape[1], sampling.sampling[1])
    ky, kx = numpy.meshgrid(ky, kx, indexing='ij')
    k2 = ky**2 + kx**2
    return numpy.exp(- (2 * numpy.pi**2 * sigma**2) * k2)


def apply_mtf(intensities, mtf:DetectorMtf):
    
    return intensities
