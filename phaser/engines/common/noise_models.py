
import typing as t

import numpy
from numpy.typing import NDArray

from phaser.hooks.solver import NoiseModel
from phaser.plan import AnascombeNoisePlan, GaussianNoisePlan
from phaser.utils.num import get_array_module


class AnascombeNoiseModel(NoiseModel[None]):
    def __init__(self, args: None, props: AnascombeNoisePlan):
        self.bias: float = props.bias

    def init_state(self) -> None:
        return None

    def calc_loss(
        self,
        model_wave: NDArray[numpy.complexfloating],
        model_intensity: NDArray[numpy.floating],
        exp_patterns: NDArray[numpy.floating],
        mask: NDArray[numpy.floating], 
        state: None
    ) -> t.Tuple[float, None]:
        xp = get_array_module(model_wave, model_intensity, exp_patterns, mask)

        return (xp.linalg.norm(
            mask * (xp.sqrt(exp_patterns + self.bias) - xp.sqrt(model_intensity + self.bias))
        ) / model_wave.shape[0], state)  # type: ignore

    def calc_wave_update(
        self,
        model_wave: NDArray[numpy.complexfloating],
        model_intensity: NDArray[numpy.floating],
        exp_patterns: NDArray[numpy.floating],
        mask: NDArray[numpy.floating], 
        state: None
    ) -> t.Tuple[NDArray[numpy.complexfloating], None]:
        xp = get_array_module(model_wave, model_intensity, exp_patterns, mask)

        update = (xp.sqrt(
            exp_patterns / (model_intensity + self.bias)
        ) - 1.0) * mask

        # broacast across incoherent modes
        return (update * model_wave, state)


class GaussianNoiseModel(NoiseModel[None]):
    def __init__(self, args: None, props: GaussianNoisePlan):
        pass

    def init_state(self) -> None:
        return None

    def calc_loss(
        self,
        model_wave: NDArray[numpy.complexfloating],
        model_intensity: NDArray[numpy.floating],
        exp_patterns: NDArray[numpy.floating],
        mask: NDArray[numpy.floating], 
        state: None
    ) -> t.Tuple[float, None]:
        xp = get_array_module(model_wave, model_intensity, exp_patterns, mask)

        return (xp.linalg.norm(
            mask * (exp_patterns - model_intensity)
        ) / model_wave.shape[0], state)

    def calc_wave_update(
        self,
        model_wave: NDArray[numpy.complexfloating],
        model_intensity: NDArray[numpy.floating],
        exp_patterns: NDArray[numpy.floating],
        mask: NDArray[numpy.floating], 
        state: None
    ) -> t.Tuple[NDArray[numpy.complexfloating], None]:
        raise NotImplementedError()