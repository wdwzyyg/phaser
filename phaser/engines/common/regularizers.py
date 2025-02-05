from functools import partial
import typing as t

import numpy
from numpy.typing import NDArray

from phaser.utils.num import get_array_module, jit, fft2, ifft2, abs2
from phaser.hooks.solver import (
    GradientRegularizer, ConstraintRegularizer, ClampObjectAmplitudeProps,
    LimitProbeSupportProps
)
from .simulation import SimulationState


class ClampObjectAmplitude(ConstraintRegularizer[None]):
    def __init__(self, args: None, props: ClampObjectAmplitudeProps):
        self.amplitude = props.amplitude

    def init_state(self, sim: SimulationState) -> None:
        return None

    def apply_group(self, group: NDArray[numpy.integer], sim: SimulationState, state: None) -> t.Tuple[SimulationState, None]:
        amp: float = numpy.dtype(sim.dtype).type(self.amplitude)  # type: ignore
        sim.state.object.data = clamp_amplitude(sim.state.object.data, amp)
        return (sim, None)


@partial(jit, donate_argnames=('obj',), cupy_fuse=True)
def clamp_amplitude(obj: NDArray[numpy.complexfloating], amplitude: float) -> NDArray[numpy.complexfloating]:
    xp = get_array_module(obj)

    obj_amp = xp.abs(obj)
    scale = xp.minimum(obj_amp, amplitude) / obj_amp
    return obj * scale


class LimitProbeSupport(ConstraintRegularizer[NDArray[numpy.bool_]]):
    def __init__(self, args: None, props: LimitProbeSupportProps):
        self.max_angle = props.max_angle

    def init_state(self, sim: SimulationState) -> NDArray[numpy.bool_]:
        mask = sim.kx**2 + sim.ky**2 <= (self.max_angle*1e-3 / sim.state.wavelength)**2
        return mask

    def apply_iter(self, sim: SimulationState, state: NDArray[numpy.bool_]) -> t.Tuple[SimulationState, NDArray[numpy.bool_]]:
        mask = state
        xp = get_array_module(sim.state.probe.data)
        print(f"intensity before: {xp.sum(abs2(sim.state.probe.data))}")
        sim.state.probe.data = ifft2(fft2(sim.state.probe.data) * mask)
        print(f"intensity after: {xp.sum(abs2(sim.state.probe.data))}")
        return (sim, mask)