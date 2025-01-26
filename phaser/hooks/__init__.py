from __future__ import annotations

import abc
from pathlib import Path
import typing as t

import numpy
from numpy.typing import NDArray, DTypeLike

from phaser.utils.num import Sampling
from phaser.utils.object import ObjectSampling
from ..types import Dataclass, Slices
from ..state import ObjectState, ProbeState, ReconsState, StateObserver
from .hook import Hook


class RawData(t.TypedDict):
    patterns: NDArray[numpy.floating]
    mask: NDArray[numpy.floating]
    sampling: Sampling
    wavelength: t.Optional[float]
    scan: t.Optional[NDArray[numpy.floating]]
    probe_options: t.Any


class LoadEmpadProps(Dataclass):
    path: Path

    diff_step: float
    kv: float


class RawDataHook(Hook[None, RawData]):
    known = {
        'empad': ('phaser.hooks.io.empad:load_empad', LoadEmpadProps),
    }


class ProbeHookArgs(t.TypedDict):
    sampling: Sampling
    wavelength: float
    seed: t.Optional[object]
    dtype: DTypeLike
    xp: t.Any


class FocusedProbeProps(Dataclass):
    defocus: float  # defocus, + is overfocus [A]
    conv_angle: float  # semiconvergence angle [mrad]


class ProbeHook(Hook[ProbeHookArgs, ProbeState]):
    known = {
        'focused': ('phaser.hooks.probe:focused_probe', FocusedProbeProps),
    }


class ObjectHookArgs(t.TypedDict):
    sampling: ObjectSampling
    wavelength: float
    slices: t.Optional[Slices]
    seed: t.Optional[object]
    dtype: DTypeLike
    xp: t.Any


class RandomObjectProps(Dataclass):
    sigma: float = 1e-6


class ObjectHook(Hook[ObjectHookArgs, ObjectState]):
    known = {
        'random': ('phaser.hooks.object:random_object', RandomObjectProps),
    }


class ScanHookArgs(t.TypedDict):
    seed: t.Optional[object]
    dtype: DTypeLike
    xp: t.Any


class RasterScanProps(Dataclass):
    shape: t.Tuple[int, int]  # ny, nx
    step_size: float          # A
    rotation: float = 0.0     # degrees CCW


class ScanHook(Hook[ScanHookArgs, NDArray[numpy.floating]]):
    known = {
        'raster': ('phaser.hooks.scan:raster_scan', RasterScanProps),
    }


class EngineArgs(t.TypedDict):
    state: ReconsState
    patterns: NDArray[numpy.floating]
    pattern_mask: NDArray[numpy.floating]
    dtype: DTypeLike
    xp: t.Any
    engine_i: int


class EngineHook(Hook[EngineArgs, ReconsState]):
    known = {}  # filled in by plan.py