from __future__ import annotations

import abc
from pathlib import Path
import typing as t

import numpy
from numpy.typing import NDArray, DTypeLike

from ..types import Dataclass, Slices
from .hook import Hook

if t.TYPE_CHECKING:
    from phaser.utils.num import Sampling
    from phaser.utils.object import ObjectSampling
    from ..state import ObjectState, ProbeState, ReconsState, Patterns
    from ..execute import Observer


class RawData(t.TypedDict):
    patterns: NDArray[numpy.floating]
    mask: NDArray[numpy.floating]
    sampling: 'Sampling'
    wavelength: t.Optional[float]
    scan: t.Optional[NDArray[numpy.floating]]
    probe_options: t.Any
    seed: t.Optional[object]


class LoadEmpadProps(Dataclass):
    path: Path

    diff_step: float
    kv: float


class RawDataHook(Hook[None, RawData]):
    known = {
        'empad': ('phaser.hooks.io.empad:load_empad', LoadEmpadProps),
    }


class ProbeHookArgs(t.TypedDict):
    sampling: 'Sampling'
    wavelength: float
    seed: t.Optional[object]
    dtype: DTypeLike
    xp: t.Any


class FocusedProbeProps(Dataclass):
    defocus: float  # defocus, + is overfocus [A]
    conv_angle: float  # semiconvergence angle [mrad]


class ProbeHook(Hook[ProbeHookArgs, 'ProbeState']):
    known = {
        'focused': ('phaser.hooks.probe:focused_probe', FocusedProbeProps),
    }


class ObjectHookArgs(t.TypedDict):
    sampling: 'ObjectSampling'
    wavelength: float
    slices: t.Optional[Slices]
    seed: t.Optional[object]
    dtype: DTypeLike
    xp: t.Any


class RandomObjectProps(Dataclass):
    sigma: float = 1e-6


class ObjectHook(Hook[ObjectHookArgs, 'ObjectState']):
    known = {
        'random': ('phaser.hooks.object:random_object', RandomObjectProps),
    }


class ScanHookArgs(t.TypedDict):
    seed: t.Optional[object]
    dtype: DTypeLike
    xp: t.Any


class RasterScanProps(Dataclass):
    shape: t.Tuple[int, int]  # ny, nx (total shape)
    step_size: float          # A
    rotation: float = 0.0     # degrees CCW
    affine: t.Optional[t.Tuple[t.Tuple[float, float], t.Tuple[float, float]]] = None


class ScanHook(Hook[ScanHookArgs, NDArray[numpy.floating]]):
    known = {
        'raster': ('phaser.hooks.scan:raster_scan', RasterScanProps),
    }


class PostInitArgs(t.TypedDict):
    data: 'Patterns'
    state: 'ReconsState'
    seed: t.Optional[object]
    dtype: DTypeLike
    xp: t.Any


class ScaleProps(Dataclass):
    scale: float


class CropDataProps(Dataclass):
    crop: t.Tuple[
        # y_i, y_f, x_i, x_f
        t.Optional[int], t.Optional[int], t.Optional[int], t.Optional[int],
    ] 


class PoissonProps(Dataclass):
    scale: t.Optional[float] = None
    gaussian: t.Optional[float] = 1.0e-3


class DropNanProps(Dataclass):
    threshold: float = 0.9


class DiffractionAlignProps(Dataclass):
    ...


class PostLoadHook(Hook[RawData, RawData]):
    known = {
        'crop_data': ('phaser.hooks.preprocessing:crop_data', CropDataProps),
        'poisson': ('phaser.hooks.preprocessing:add_poisson_noise', PoissonProps),
        'scale': ('phaser.hooks.preprocessing:scale_patterns', ScaleProps),
    }


class PostInitHook(Hook[PostInitArgs, t.Tuple['Patterns', 'ReconsState']]):
    known = {
        'drop_nans': ('phaser.hooks.preprocessing:drop_nan_patterns', DropNanProps),
        'diffraction_align': ('phaser.hooks.preprocessing:diffraction_align', DiffractionAlignProps),
    }


class EngineArgs(t.TypedDict):
    data: 'Patterns'
    state: 'ReconsState'
    dtype: DTypeLike
    xp: t.Any
    recons_name: str
    engine_i: int
    observer: 'Observer'
    seed: t.Any


class EngineHook(Hook[EngineArgs, 'ReconsState']):
    known = {}  # filled in by plan.py


class FlagArgs(t.TypedDict):
    state: 'ReconsState'
    niter: int


class FlagHook(Hook[FlagArgs, bool]):
    known = {}