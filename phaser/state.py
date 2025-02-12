from dataclasses import dataclass
import typing as t

import numpy
from numpy.typing import NDArray

from phaser.utils.num import Sampling, to_numpy
from phaser.utils.object import ObjectSampling

if t.TYPE_CHECKING:
    from phaser.utils.io import HdfLike


@dataclass
class Patterns():
    patterns: NDArray[numpy.floating]
    """Raw diffraction patterns, with 0-frequency sample in corner"""
    pattern_mask: NDArray[numpy.floating]
    """Mask indicating which portions of the diffraction patterns contain data."""

    def to_numpy(self) -> t.Self:
        return self.__class__(
            to_numpy(self.patterns), to_numpy(self.pattern_mask)
        )


@dataclass
class IterState():
    engine_num: int
    """Engine number. 1-indexed (0 means before any reconstruction)."""
    engine_iter: int
    """Iteration number on this engine. 1-indexed (0 means before any iterations)."""
    total_iter: int
    """Total iteration number. 1-indexed (0 means before any iterations)."""

    def to_numpy(self) -> t.Self:
        return self.__class__(
            int(self.engine_num), int(self.engine_iter), int(self.total_iter)
        )

    def copy(self) -> t.Self:
        import copy
        return copy.deepcopy(self)

    @staticmethod
    def empty() -> 'IterState':
        return IterState(0, 0, 0)


@dataclass
class ProbeState():
    sampling: Sampling
    """Probe coordinate system. See `Sampling` for more details."""
    data: NDArray[numpy.complexfloating]
    """Probe wavefunction, in realspace. Shape (modes, y, x)"""

    def to_numpy(self) -> t.Self:
        return self.__class__(
            self.sampling, to_numpy(self.data)
        )

    def copy(self) -> t.Self:
        import copy
        return copy.deepcopy(self)


@dataclass
class ObjectState():
    sampling: ObjectSampling
    """Object coordinate system. See `ObjectSampling` for more details."""
    data: NDArray[numpy.complexfloating]
    """Object wavefunction. Shape (z, y, x)"""
    zs: NDArray[numpy.floating]
    """Z positions of slices (in length units)."""

    def to_numpy(self) -> t.Self:
        return self.__class__(
            self.sampling, to_numpy(self.data), to_numpy(self.zs)
        )

    def copy(self) -> t.Self:
        import copy
        return copy.deepcopy(self)


@dataclass
class ProgressState():
    iters: NDArray[numpy.integer]
    """Iterations error measurements were taken at."""
    detector_errors: NDArray[numpy.floating]
    """Detector error measurements at those iterations"""

    def to_numpy(self) -> t.Self:
        return self.__class__(
            to_numpy(self.iters), to_numpy(self.detector_errors)
        )

    def copy(self) -> t.Self:
        import copy
        return copy.deepcopy(self)

    @staticmethod
    def empty() -> 'ProgressState':
        return ProgressState(
            numpy.array([], dtype=numpy.uint64),
            numpy.array([], dtype=numpy.float64),
        )


@dataclass(kw_only=True)
class ReconsState:
    iter: IterState
    wavelength: float

    probe: ProbeState
    object: ObjectState
    scan: NDArray[numpy.floating]
    """Scan coordinates (y, x), in length units. Shape (..., 2)"""
    progress: ProgressState

    def to_numpy(self) -> t.Self:
        return self.__class__(
            iter=self.iter.to_numpy(),
            probe=self.probe.to_numpy(),
            object=self.object.to_numpy(),
            scan=to_numpy(self.scan),
            progress=self.progress.to_numpy(),
            wavelength=float(self.wavelength),
        )

    def copy(self) -> t.Self:
        import copy
        return copy.deepcopy(self)

    def write_hdf5(self, file: 'HdfLike'):
        from phaser.utils.io import hdf5_write_state
        hdf5_write_state(self, file)

    @staticmethod
    def read_hdf5(file: 'HdfLike') -> 'ReconsState':
        from phaser.utils.io import hdf5_read_state
        return hdf5_read_state(file).to_complete()


@dataclass(kw_only=True)
class PartialReconsState:
    iter: IterState
    wavelength: float

    probe: t.Optional[ProbeState] = None
    object: t.Optional[ObjectState] = None
    scan: t.Optional[NDArray[numpy.floating]] = None
    """Scan coordinates (y, x), in length units. Shape (..., 2)"""
    progress: t.Optional[ProgressState] = None

    def to_numpy(self) -> t.Self:
        return self.__class__(
            iter=self.iter.to_numpy(),
            probe=self.probe.to_numpy() if self.probe is not None else None,
            object=self.object.to_numpy() if self.object is not None else None,
            scan=to_numpy(self.scan) if self.scan is not None else None,
            progress=self.progress.to_numpy() if self.progress is not None else None,
            wavelength=float(self.wavelength),
        )

    def to_complete(self) -> ReconsState:
        missing = tuple(filter(lambda k: getattr(self, k) is None, ('probe', 'object', 'scan')))
        if len(missing):
            raise ValueError(f"ReconsState missing {', '.join(map(repr, missing))}")

        progress = self.progress if self.progress is not None else ProgressState.empty()
        iter = self.iter if self.iter is not None else IterState.empty()

        return ReconsState(
            wavelength=self.wavelength,
            probe=t.cast(ProbeState, self.probe),
            object=t.cast(ObjectState, self.object),
            scan=t.cast(NDArray[numpy.floating], self.scan),
            progress=progress, iter=iter,
        )

    def write_hdf5(self, file: 'HdfLike'):
        from phaser.utils.io import hdf5_write_state
        hdf5_write_state(self, file)

    @staticmethod
    def read_hdf5(file: 'HdfLike') -> 'PartialReconsState':
        from phaser.utils.io import hdf5_read_state
        return hdf5_read_state(file)



StateObserver: t.TypeAlias = t.Callable[[t.Union[ReconsState, PartialReconsState]], t.Any]


try:
    import jax.tree_util
except ImportError:
    pass
else:
    jax.tree_util.register_dataclass(ProbeState, ('data',), ('sampling',))
    jax.tree_util.register_dataclass(ObjectState, ('data', 'zs'), ('sampling',))
    jax.tree_util.register_dataclass(IterState, ('engine_num', 'engine_iter', 'total_iter'), ())

    jax.tree_util.register_dataclass(
        ReconsState, ('probe', 'object', 'scan', 'wavelength', 'iter'), ('progress',)
    )