from dataclasses import dataclass
import typing as t

import numpy
from numpy.typing import NDArray

from phaser.utils.num import Sampling
from phaser.utils.object import ObjectSampling


@dataclass
class IterState():
    engine_num: int
    engine_iter: int
    total_iter: int


@dataclass
class ProbeState():
    sampling: Sampling
    data: NDArray[numpy.complexfloating]


@dataclass
class ObjectState():
    sampling: ObjectSampling
    data: NDArray[numpy.complexfloating]
    zs: NDArray[numpy.floating]


@dataclass
class ErrorsState():
    iters: NDArray[numpy.integer]
    detector_errors: NDArray[numpy.floating]


@dataclass
class ReconsState():
    iter: IterState

    probe: ProbeState
    object: ObjectState
    scan: NDArray[numpy.floating]

    errors: ErrorsState

    wavelength: float


StateObserver: t.TypeAlias = t.Callable[[ReconsState], t.Any]