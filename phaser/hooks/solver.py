import abc
import typing as t

import numpy
from numpy.typing import NDArray

from . import Hook

StateT = t.TypeVar('StateT')

if t.TYPE_CHECKING:
    from phaser.engines.common.simulation import SimulationState
    from phaser.state import StateObserver


class NoiseModel(abc.ABC, t.Generic[StateT]):
    @abc.abstractmethod
    def init_state(self) -> StateT:
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


class ConventionalSolverArgs(t.TypedDict):
    niter: int
    grouping: int
    compact: bool


class ConventionalSolver(abc.ABC):
    @abc.abstractmethod
    def solve(
        self,
        sim: 'SimulationState',
        engine_i: int,
        observers: t.Sequence['StateObserver'] = ()
    ) -> 'SimulationState':
        ...


class ConventionalSolverHook(Hook[ConventionalSolverArgs, ConventionalSolver]):
    known = {}