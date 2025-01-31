import abc
import typing as t

import numpy
from numpy.typing import NDArray

from phaser.types import Dataclass
from . import Hook, FlagArgs

StateT = t.TypeVar('StateT')

if t.TYPE_CHECKING:
    from phaser.engines.common.simulation import SimulationState
    from phaser.execute import Observer
    from phaser.plan import ConventionalEnginePlan


class NoiseModel(abc.ABC, t.Generic[StateT]):
    @abc.abstractmethod
    def init_state(self, sim: 'SimulationState') -> StateT:
        ...

    @abc.abstractmethod
    def calc_loss(
        self,
        model_wave: NDArray[numpy.complexfloating],
        model_intensity: NDArray[numpy.floating],
        exp_patterns: NDArray[numpy.floating],
        mask: NDArray[numpy.floating], 
        state: StateT,
    ) -> t.Tuple[float, StateT]:
        """
        Return the calculated loss, summed across the detector and averaged across the scan.

        May be called in a JAX jit context, so must have no side effects.
        """
        ...

    @abc.abstractmethod
    def calc_wave_update(
        self,
        model_wave: NDArray[numpy.complexfloating],
        model_intensity: NDArray[numpy.floating],
        exp_patterns: NDArray[numpy.floating],
        mask: NDArray[numpy.floating], 
        state: StateT,
    ) -> t.Tuple[NDArray[numpy.complexfloating], StateT]:
        """
        Return the calculated wave update `chi` in reciprocal space.

        May be called in a JAX jit context, so must have no side effects.
        """
        ...


class NoiseModelHook(Hook[None, NoiseModel]):
    known = {}


# refactoring:
# HasState should be an interface, as well as the different ways of regularization + noise model
# then there can be a list of each of these types in the engine


class ConstraintRegularizer(abc.ABC, t.Generic[StateT]):
    @abc.abstractmethod
    def init_state(self, sim: 'SimulationState') -> StateT:
        ...

    def apply_group(self, group: NDArray[numpy.integer], sim: 'SimulationState', state: StateT) -> t.Tuple['SimulationState', StateT]:
        return (sim, state)

    def apply_iter(self, sim: 'SimulationState', state: StateT) -> t.Tuple['SimulationState', StateT]:
        return (sim, state)


class GradientRegularizer(abc.ABC, t.Generic[StateT]):
    @abc.abstractmethod
    def init_state(self, sim: 'SimulationState') -> StateT:
        ...

    @abc.abstractmethod
    def calc_loss_group(self, group: NDArray[numpy.integer], sim: 'SimulationState', state: StateT) -> t.Tuple[float, StateT]:
        ...


class ClampObjectAmplitudeProps(Dataclass):
    amplitude: float = 1.1


class LimitProbeSupportProps(Dataclass):
    max_angle: float


class RegularizerHook(Hook[None, t.Union[ConstraintRegularizer, GradientRegularizer]]):
    known = {
        'clamp_object_amplitude': ('phaser.engines.common.regularizers:ClampObjectAmplitude', ClampObjectAmplitudeProps),
        'limit_probe_support': ('phaser.engines.common.regularizers:LimitProbeSupport', LimitProbeSupportProps),
    }


class ConventionalSolver(abc.ABC):
    @abc.abstractmethod
    def solve(
        self,
        sim: 'SimulationState',
        engine_i: int,
        observer: 'Observer',
        update_probe: t.Callable[[FlagArgs], bool],
        update_object: t.Callable[[FlagArgs], bool],
    ) -> 'SimulationState':
        ...


class ConventionalSolverHook(Hook['ConventionalEnginePlan', ConventionalSolver]):
    known = {}