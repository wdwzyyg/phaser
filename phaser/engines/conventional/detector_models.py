
import numpy
from numpy.typing import NDArray

from phaser.plan import ModulusConstraint


def modulus_constraint(
    xp,
    model_wave: NDArray[numpy.complexfloating],
    model_intensity: NDArray[numpy.floating],
    exp_patterns: NDArray[numpy.floating],
    props: ModulusConstraint
) -> NDArray[numpy.complexfloating]:
    intensity_update = xp.sqrt((exp_patterns + props.bias) / (model_intensity + props.bias)) - 1.0
    return model_wave * intensity_update[:, None, ...]