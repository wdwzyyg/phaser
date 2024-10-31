from __future__ import annotations

import typing as t

#from pane.annotations import Tagged

from .types import Dataclass, Slices
from .hooks import RawDataHook, ProbeHook, ObjectHook, ScanHook, EngineHook


class Engine(Dataclass, kw_only=True):
    sim_shape: t.Optional[t.Tuple[int, int]] = None
    resize_method: t.Literal['pad_crop', 'resample'] = 'pad_crop'

    probe_modes: int = 1

    slices: t.Optional[Slices] = None

    niter: int = 10
    grouping: t.Optional[int] = None


class ModulusConstraint(Dataclass, kw_only=True):
    type: t.Literal['modulus'] = 'modulus'
    bias: float = 1e-10


class ConventionalEngine(Engine, kw_only=True):
    #detector_model: t.Annotated[t.Union[ModulusConstraint], Tagged('type')]
    detector_model: ModulusConstraint = ModulusConstraint()


EngineHook.known['conventional'] = ('phaser.engines.conventional.run:run_engine', ConventionalEngine)


class GradientEngine(Engine):
    ...


EngineHook.known['gradient'] = ('phaser.engines.gradient.run:run_engine', GradientEngine)


class ReconsPlan(Dataclass, kw_only=True):
    backend: t.Optional[t.Literal['gpu', 'cpu']] = None
    dtype: t.Literal['float32', 'float64'] = 'float32'

    wavelength: t.Optional[float] = None

    raw_data: RawDataHook

    init_probe: ProbeHook
    init_object: ObjectHook
    init_scan: ScanHook

    engines: t.List[EngineHook]
    #engines: t.List[t.Annotated[t.Union[ConventionalEngine, GradientEngine], Tagged('type')]]