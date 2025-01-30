from __future__ import annotations

import typing as t

from .types import Dataclass, Slices, BackendName
from .hooks import RawDataHook, ProbeHook, ObjectHook, ScanHook, EngineHook
from .hooks.solver import NoiseModelHook, ConventionalSolverHook


class EnginePlan(Dataclass, kw_only=True):
    sim_shape: t.Optional[t.Tuple[int, int]] = None
    resize_method: t.Literal['pad_crop', 'resample'] = 'pad_crop'

    probe_modes: int = 1

    slices: t.Optional[Slices] = None

    niter: int = 10
    grouping: t.Optional[int] = None
    compact: bool = False


class AnascombeNoisePlan(Dataclass, kw_only=True):
    type: t.Literal['anascombe'] = 'anascombe'
    bias: float = 1e-10


class GaussianNoisePlan(Dataclass, kw_only=True):
    type: t.Literal['gaussian'] = 'gaussian'


NoiseModelHook.known['anascombe'] = ('phaser.engines.common.noise_models:AnascombeNoiseModel', AnascombeNoisePlan)
NoiseModelHook.known['gaussian'] = ('phaser.engines.common.noise_models:GaussianNoiseModel', GaussianNoisePlan)


class LSQMLSolverPlan(Dataclass, kw_only=True):
    type: t.Literal['lsqml'] = 'lsqml'
    stochastic: bool = True

    beta_object: float = 1.0
    beta_probe: float = 1.0

    illum_reg_object: float = 1e-2
    illum_reg_probe: float = 1e-2

    gamma: float = 1e-4


class EPIESolverPlan(Dataclass, kw_only=True):
    type: t.Literal['epie'] = 'epie'

    beta_object: float = 1.0
    beta_probe: float = 1.0


ConventionalSolverHook.known['lsqml'] = ('phaser.engines.conventional.solvers:LSQMLSolver', LSQMLSolverPlan)
ConventionalSolverHook.known['epie'] = ('phaser.engines.conventional.solvers:EPIESolver', EPIESolverPlan)


class ConventionalEnginePlan(EnginePlan, kw_only=True):
    #detector_model: t.Annotated[t.Union[ModulusConstraint], Tagged('type')]
    noise_model: NoiseModelHook
    solver: ConventionalSolverHook


class GradientEnginePlan(EnginePlan):
    ...


EngineHook.known['conventional'] = ('phaser.engines.conventional.run:run_engine', ConventionalEnginePlan)
EngineHook.known['gradient'] = ('phaser.engines.gradient.run:run_engine', GradientEnginePlan)


class ReconsPlan(Dataclass, kw_only=True):
    backend: t.Optional[BackendName] = None
    dtype: t.Literal['float32', 'float64'] = 'float32'

    wavelength: t.Optional[float] = None

    raw_data: RawDataHook

    init_probe: ProbeHook
    init_object: ObjectHook
    init_scan: ScanHook

    slices: t.Optional[Slices] = None

    engines: t.List[EngineHook]
    #engines: t.List[t.Annotated[t.Union[ConventionalEngine, GradientEngine], Tagged('type')]]