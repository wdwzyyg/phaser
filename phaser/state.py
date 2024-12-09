from dataclasses import dataclass
import typing as t

import numpy
from numpy.typing import NDArray

from phaser.utils.num import Sampling, to_numpy
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

    def to_numpy(self) -> t.Self:
        return self.__class__(
            self.sampling, to_numpy(self.data)
        )


@dataclass
class ObjectState():
    sampling: ObjectSampling
    data: NDArray[numpy.complexfloating]
    zs: NDArray[numpy.floating]

    def to_numpy(self) -> t.Self:
        return self.__class__(
            self.sampling, to_numpy(self.data), to_numpy(self.zs)
        )


@dataclass
class ProgressState():
    iters: NDArray[numpy.integer]
    detector_errors: NDArray[numpy.floating]

    def to_numpy(self) -> t.Self:
        return self.__class__(
            to_numpy(self.iters), to_numpy(self.detector_errors)
        )


@dataclass(kw_only=True)
class ReconsState:
    iter: IterState
    wavelength: float

    probe: ProbeState
    object: ObjectState
    scan: NDArray[numpy.floating]
    progress: ProgressState

    def to_numpy(self) -> t.Self:
        return self.__class__(
            iter=self.iter,
            probe=self.probe.to_numpy(),
            object=self.object.to_numpy(),
            scan=to_numpy(self.scan),
            progress=self.progress.to_numpy(),
            wavelength=self.wavelength,
        )


@dataclass(kw_only=True)
class PartialReconsState:
    iter: IterState
    wavelength: float

    probe: t.Optional[ProbeState] = None
    object: t.Optional[ObjectState] = None
    scan: t.Optional[NDArray[numpy.floating]] = None
    progress: t.Optional[ProgressState] = None

    def to_numpy(self) -> t.Self:
        return self.__class__(
            iter=self.iter,
            probe=self.probe.to_numpy() if self.probe is not None else None,
            object=self.object.to_numpy() if self.object is not None else None,
            scan=to_numpy(self.scan) if self.scan is not None else None,
            progress=self.progress.to_numpy() if self.progress is not None else None,
            wavelength=self.wavelength,
        )


StateObserver: t.TypeAlias = t.Callable[[t.Union[ReconsState, PartialReconsState]], t.Any]


try:
    import jax.tree_util
except ImportError:
    pass
else:
    jax.tree_util.register_dataclass(
        ReconsState,
        ('probe', 'object', 'scan'),
        ('progress', 'wavelength', 'iter')
    )