
import numpy

from phaser.utils.num import cast_array_module, to_complex_dtype
from phaser.utils.object import random_phase_object
from ..state import ObjectState
from . import ObjectHookArgs, RandomObjectProps


def random_object(args: ObjectHookArgs, props: RandomObjectProps) -> ObjectState:
    sampling = args['sampling']

    obj = random_phase_object(
        sampling.shape, props.sigma,
        dtype=to_complex_dtype(args['dtype']),
        xp=cast_array_module(args['xp'])
    )
    return ObjectState(sampling, obj, numpy.array([0.]))