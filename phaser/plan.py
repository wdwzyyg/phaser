from __future__ import annotations

import typing as t

from .types import Dataclass, Slices, BackendName, Flag
from .hooks import RawDataHook, ProbeHook, ObjectHook, ScanHook, EngineHook, FlagHook, PreprocessingHook
from .hooks.solver import NoiseModelHook, ConventionalSolverHook, RegularizerHook


FlagLike: t.TypeAlias = t.Union[bool, Flag, FlagHook]

class EnginePlan(Dataclass, kw_only=True):
    sim_shape: t.Optional[t.Tuple[int, int]] = None
    resize_method: t.Literal['pad_crop', 'resample'] = 'pad_crop'

    probe_modes: int = 1

    slices: t.Optional[Slices] = None

    niter: int = 10
    grouping: t.Optional[int] = None
    compact: bool = False

    regularizers: t.List[RegularizerHook]

    update_probe: FlagLike = True
    update_object: FlagLike = True

    calc_error: FlagLike = Flag(every=1)
    calc_error_fraction: float = 0.1

    save: FlagLike = Flag(every=10)
    send_every_group: bool = False


class AmplitudeNoisePlan(Dataclass, kw_only=True):
    gaussian_variance: float = 0.1
    eps: float = 1e-8
    offset: float = 0.0


class AnscombeNoisePlan(AmplitudeNoisePlan, kw_only=True):
    offset: float = 0.375


NoiseModelHook.known['amplitude'] = ('phaser.engines.common.noise_models:AmplitudeNoiseModel', AmplitudeNoisePlan)
NoiseModelHook.known['anscombe'] = ('phaser.engines.common.noise_models:AnscombeNoiseModel', AnscombeNoisePlan)


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
    noise_model: NoiseModelHook
    solver: ConventionalSolverHook


class GradientEnginePlan(EnginePlan):
    pass


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

    preprocessing: t.Sequence[PreprocessingHook] = ()

    slices: t.Optional[Slices] = None

    engines: t.List[EngineHook]
    #engines: t.List[t.Annotated[t.Union[ConventionalEngine, GradientEngine], Tagged('type')]]