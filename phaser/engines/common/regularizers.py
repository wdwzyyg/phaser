import typing as t

import numpy
from numpy.typing import NDArray

from phaser.utils.num import get_array_module
from phaser.hooks.solver import GradientRegularizer, ConstraintRegularizer, ClampObjectAmplitudeProps
from .simulation import SimulationState


class ClampObjectAmplitude(ConstraintRegularizer[None]):
    def __init__(self, args: None, props: ClampObjectAmplitudeProps):
        self.amplitude = props.amplitude

    def init_state(self, sim: SimulationState) -> None:
        return None

    def apply_group(self, group: NDArray[numpy.integer], sim: SimulationState, state: None) -> t.Tuple[SimulationState, None]:
        xp = get_array_module(sim.state.object.data)

        obj_amp = xp.abs(sim.state.object.data)
        scale = xp.minimum(obj_amp, self.amplitude) / obj_amp
        sim.state.object.data *= scale

        return (sim, None)