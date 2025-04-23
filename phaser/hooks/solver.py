import abc
import typing as t

import numpy
from numpy.typing import NDArray

from phaser.types import Dataclass, ReconsVar
from phaser.utils.num import Float
from . import Hook

StateT = t.TypeVar('StateT')

if t.TYPE_CHECKING:
    from phaser.engines.common.simulation import SimulationState
    from phaser.execute import Observer
    from phaser.plan import ConventionalEnginePlan, GradientEnginePlan
    from phaser.state import ReconsState


class HasState(t.Protocol[StateT]):  # type: ignore
    def init_state(self, sim: 'ReconsState') -> StateT:
        ...


class NoiseModel(HasState[StateT], t.Protocol[StateT]):
    @classmethod
    def name(cls) -> str:
        ...

    def calc_loss(
        self,
        model_wave: NDArray[numpy.complexfloating],
        model_intensity: NDArray[numpy.floating],
        exp_patterns: NDArray[numpy.floating],
        mask: NDArray[numpy.floating], 
        state: StateT,
    ) -> t.Tuple[Float, StateT]:
        """
        Return the calculated loss, summed across the detector and averaged across the scan.

        May be called in a JAX jit context, so must have no side effects.
        """
        ...

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


class PositionSolver(HasState[StateT], t.Protocol[StateT]):
    def perform_update(
        self,
        positions: NDArray[numpy.floating],
        gradients: NDArray[numpy.floating],
        state: StateT
    ) -> t.Tuple[NDArray[numpy.floating], StateT]:
        """
        Return the calculated position updates
        """
        ...


class SteepestDescentPositionSolverProps(Dataclass):
    # fraction of optimal step to take
    step_size: float = 1e-2
    # maximum step size (in angstroms)
    max_step_size: t.Optional[float] = None


class MomentumPositionSolverProps(Dataclass):
    # fraction of optimal step to take
    step_size: float = 1e-2
    # maximum step size (in angstroms)
    max_step_size: t.Optional[float] = None
    # momentum decay rate
    momentum: float = 0.9


class PositionSolverHook(Hook[None, PositionSolver]):
    known = {
        'steepest_descent': ('phaser.engines.common.position_correction:SteepestDescentPositionSolver', SteepestDescentPositionSolverProps),
        'momentum': ('phaser.engines.common.position_correction:MomentumPositionSolver', MomentumPositionSolverProps),
    }


# refactoring:
# HasState should be an interface, as well as the different ways of regularization + noise model
# then there can be a list of each of these types in the engine


@t.runtime_checkable
class GroupConstraint(HasState[StateT], t.Protocol[StateT]):
    def apply_group(self, group: NDArray[numpy.integer], sim: 'ReconsState', state: StateT) -> t.Tuple['ReconsState', StateT]:
        ...


@t.runtime_checkable
class IterConstraint(HasState[StateT], t.Protocol[StateT]):
    def apply_iter(self, sim: 'ReconsState', state: StateT) -> t.Tuple['ReconsState', StateT]:
        ...


@t.runtime_checkable
class CostRegularizer(HasState[StateT], t.Protocol[StateT]):
    def calc_loss_group(self, group: NDArray[numpy.integer], sim: 'ReconsState', state: StateT) -> t.Tuple[float, StateT]:
        ...


class ClampObjectAmplitudeProps(Dataclass):
    amplitude: float = 1.1


class LimitProbeSupportProps(Dataclass):
    max_angle: float


class RegularizeLayersProps(Dataclass):
    weight: float = 0.9  # weight of regularization to apply
    sigma: float = 50.0  # standard deviation of gaussian filter


class ObjLowPassProps(Dataclass):
    max_freq: float = 0.4  # 1/px (nyquist = 0.5)


class IterConstraintHook(Hook[None, IterConstraint]):
    known = {
        'clamp_object_amplitude': ('phaser.engines.common.regularizers:ClampObjectAmplitude', ClampObjectAmplitudeProps),
        'limit_probe_support': ('phaser.engines.common.regularizers:LimitProbeSupport', LimitProbeSupportProps),
        'layers': ('phaser.engines.common.regularizers:RegularizeLayers', RegularizeLayersProps),
        'obj_low_pass': ('phaser.engines.common.regularizers:ObjLowPass', ObjLowPassProps),
        'remove_phase_ramp': ('phaser.engines.common.regularizers:RemovePhaseRamp', t.Dict[str, t.Any]),
    }


class GroupConstraintHook(Hook[None, GroupConstraint]):
    known = {
        'clamp_object_amplitude': ('phaser.engines.common.regularizers:ClampObjectAmplitude', ClampObjectAmplitudeProps),
        'limit_probe_support': ('phaser.engines.common.regularizers:LimitProbeSupport', LimitProbeSupportProps),
        'obj_low_pass': ('phaser.engines.common.regularizers:ObjLowPass', ObjLowPassProps),
        'remove_phase_ramp': ('phaser.engines.common.regularizers:RemovePhaseRamp', t.Dict[str, t.Any]),
    }


class CostRegularizerProps(Dataclass):
    cost: float


class TVRegularizerProps(Dataclass):
    cost: float
    eps: float = 1.0e-8


class CostRegularizerHook(Hook[None, CostRegularizer]):
    known = {
        'obj_l1': ('phaser.engines.common.regularizers:ObjL1', CostRegularizerProps),
        'obj_l2': ('phaser.engines.common.regularizers:ObjL2', CostRegularizerProps),
        'obj_phase_l1': ('phaser.engines.common.regularizers:ObjPhaseL1', CostRegularizerProps),
        'obj_recip_l1': ('phaser.engines.common.regularizers:ObjRecipL1', CostRegularizerProps),
        'obj_tv': ('phaser.engines.common.regularizers:ObjTotalVariation', TVRegularizerProps),
        'obj_tikh': ('phaser.engines.common.regularizers:ObjTikhonov', CostRegularizerProps),
        'obj_tikhonov': ('phaser.engines.common.regularizers:ObjTikhonov', CostRegularizerProps),
        'layers_tv': ('phaser.engines.common.regularizers:LayersTotalVariation', CostRegularizerProps),
        'layers_tikh': ('phaser.engines.common.regularizers:LayersTikhonov', CostRegularizerProps),
        'layers_tikhonov': ('phaser.engines.common.regularizers:LayersTikhonov', CostRegularizerProps),
        'probe_phase_tikh': ('phaser.engines.common.regularizers:ProbePhaseTikhonov', CostRegularizerProps),
        'probe_phase_tikhonov': ('phaser.engines.common.regularizers:ProbePhaseTikhonov', CostRegularizerProps),
    }


class ConventionalSolver(abc.ABC):
    @classmethod
    @abc.abstractmethod
    def name(cls) -> str:
        ...

    @abc.abstractmethod
    def init(self, sim: 'SimulationState') -> 'SimulationState':
        ...

    @abc.abstractmethod
    def presolve(
        self,
        sim: 'SimulationState',
        groups: t.Iterator[NDArray[numpy.int_]], *,
        patterns: NDArray[numpy.floating],
        pattern_mask: NDArray[numpy.floating],
        propagators: t.Optional[NDArray[numpy.complexfloating]],
    ) -> 'SimulationState':
        ...

    @abc.abstractmethod
    def run_iteration(
        self,
        sim: 'SimulationState',
        groups: t.Iterator[NDArray[numpy.int_]], *,
        patterns: NDArray[numpy.floating],
        pattern_mask: NDArray[numpy.floating],
        propagators: t.Optional[NDArray[numpy.complexfloating]],
        update_object: bool,
        update_probe: bool,
        update_positions: bool,
        calc_error: bool,
        calc_error_mask: NDArray[numpy.bool_],
        observer: 'Observer',
    ) -> t.Tuple['SimulationState', NDArray[numpy.floating], t.List[NDArray[numpy.floating]]]:
        """
        Run an iteration of the reconstruction. Return a tuple `(sim, pos_update, iter_errors)`.
        """
        ...


class ConventionalSolverHook(Hook['ConventionalEnginePlan', ConventionalSolver]):
    known = {}


class GradientSolver(HasState[StateT], t.Protocol[StateT]):
    name: str
    params: t.FrozenSet[ReconsVar]

    def init_state(self, sim: 'ReconsState') -> StateT:
        ...

    def update(
        self, sim: 'ReconsState', state: StateT, grad: t.Dict[ReconsVar, numpy.ndarray], loss: float,
    ) -> t.Tuple[t.Dict[ReconsVar, numpy.ndarray], StateT]:
        ...


class GradientSolverArgs(t.TypedDict):
    plan: 'GradientEnginePlan'
    params: t.Iterable[ReconsVar]


class GradientSolverHook(Hook['GradientSolverArgs', GradientSolver]):
    known = {}