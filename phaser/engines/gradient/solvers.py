import logging
import typing as t

import numpy
from numpy.typing import NDArray, ArrayLike

from phaser.utils.num import as_array, abs2
from phaser.hooks.solver import GradientSolver, GradientSolverArgs
from phaser.hooks.schedule import FlagArgs, ScheduleLike
from phaser.types import ReconsVar, process_schedule
from phaser.plan import GradientEnginePlan, AdamSolverPlan, PolyakSGDSolverPlan, SGDSolverPlan
from phaser.state import ReconsState
from .run import extract_vars, apply_update

import optax
from optax import GradientTransformation, GradientTransformationExtraArgs
from optax.schedules import StatefulSchedule


class OptaxScheduleWrapper(StatefulSchedule):
    def __init__(self, schedule: ScheduleLike):
        self.inner = process_schedule(schedule)

    def init(self) -> t.Optional[float]:
        return None

    def update_for_iter(self, sim: ReconsState, state: t.Optional[float], niter: int) -> float:
        return self.inner({'state': sim, 'niter': niter})

    # mock update from inside jax
    def update(
        self, state: t.Optional[float],
        **extra_args,
    ) -> t.Optional[float]:
        return state

    def __call__(
        self, state: t.Optional[float],
        **extra_args,
    ) -> float:
        assert state is not None
        return state


OptaxSolverState: t.TypeAlias = t.Tuple[t.Any, t.Dict[str, t.Optional[float]]]


class OptaxSolver(GradientSolver[OptaxSolverState]):
    def __init__(self, name: str, factory: t.Callable[..., GradientTransformation], hyperparams: t.Mapping[str, ScheduleLike],
                 params: t.Iterable[ReconsVar]):
        self.factory: t.Callable[..., GradientTransformation] = factory
        #self.inner: GradientTransformationExtraArgs = optax.with_extra_args_support(solver)

        self.hyperparams: t.Dict[str, OptaxScheduleWrapper] = {k: OptaxScheduleWrapper(v) for (k, v) in hyperparams.items()}
        self.params: t.FrozenSet[ReconsVar] = frozenset(params)

        self.name: str = name # or self.inner.__class__.__name__

    def init_state(self, sim: ReconsState) -> OptaxSolverState:
        return (
            None,
            {k: v.init() for (k, v) in self.hyperparams.items()},
        )

    def _resolve(self, hparams: t.Mapping[str, t.Optional[float]]) -> GradientTransformationExtraArgs:
        return optax.with_extra_args_support(
            self.factory(**{k: v(hparams[k]) for (k, v) in self.hyperparams.items()})
        )

    def update_for_iter(self, sim: ReconsState, state: OptaxSolverState, niter: int) -> OptaxSolverState:
        hparams_state: t.Dict[str, t.Optional[float]] = {k: v.update_for_iter(sim, state[1][k], niter) for (k, v) in self.hyperparams.items()}
        return (
            self._resolve(hparams_state).init(params=extract_vars(sim, self.params)[0]) if state[0] is None else state[0],
            hparams_state
        )

    def update(
        self, sim: 'ReconsState', state: OptaxSolverState, grad: t.Dict[ReconsVar, numpy.ndarray], loss: float,
    ) -> t.Tuple[t.Dict[ReconsVar, numpy.ndarray], OptaxSolverState]:
        (inner_state, hparams_state) = state
        hparams_state = {k: v.update(hparams_state[k]) for (k, v) in self.hyperparams.items()}
        (updates, inner_state) = self._resolve(hparams_state).update(
            grad, inner_state, params=extract_vars(sim, self.params)[0], value=loss, loss=loss
        )
        return (t.cast(t.Dict[ReconsVar, t.Any], updates), (inner_state, hparams_state))


class SGDSolver(OptaxSolver):
    def __init__(self, args: GradientSolverArgs, props: SGDSolverPlan):
        hparams = {
            'learning_rate': props.learning_rate
        }

        if props.momentum is not None:
            hparams['momentum'] = props.momentum
            def factory(**kwargs: t.Any) -> GradientTransformation:
                return optax.chain(
                    optax.trace(kwargs['momentum'], props.nesterov),
                    optax.scale_by_learning_rate(kwargs['learning_rate'], flip_sign=False),
                )
        else:
            def factory(**kwargs: t.Any) -> GradientTransformation:
                return optax.scale_by_learning_rate(kwargs['learning_rate'], flip_sign=False)

        super().__init__('sgd', factory, hparams, args['params'])


class AdamSolver(OptaxSolver):
    def __init__(self, args: GradientSolverArgs, props: AdamSolverPlan):
        hparams = {
            'learning_rate': props.learning_rate
        }

        def factory(**kwargs) -> GradientTransformation:
            return optax.chain(
                optax.scale_by_adam(props.b1, props.b2, props.eps, props.eps_root, nesterov=props.nesterov),
                optax.scale_by_learning_rate(learning_rate=kwargs['learning_rate'], flip_sign=False),
            )

        super().__init__('adam', factory, hparams, args['params'])


class PolyakSGDSolver(OptaxSolver):
    def __init__(self, args: GradientSolverArgs, props: PolyakSGDSolverPlan):
        hparams = {
            'max_learning_rate': props.max_learning_rate,
            'scaling': props.scaling,
        }

        def factory(**kwargs) -> GradientTransformation:
            return optax.chain(
                optax.scale_by_learning_rate(kwargs['scaling'], flip_sign=False),
                optax.scale_by_polyak(
                    max_learning_rate=kwargs['max_learning_rate'], f_min=props.f_min,
                    eps=props.eps, #variant='sps',
                )
            )

        super().__init__('polyak_sgd', factory, hparams, args['params'])