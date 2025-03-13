import typing as t

import numpy
from numpy.typing import ArrayLike
from numpy.testing import assert_array_almost_equal
import pytest

from .utils import with_backends, get_backend_module

from phaser.utils.num import to_numpy
from phaser.utils.image import (
    affine_transform, _BoundaryMode
)


@with_backends('cpu', 'jax', 'cuda')
@pytest.mark.parametrize(('mode', 'order', 'expected'), [
    ('grid-constant', 0, [ 1.0,  1.0,  1.0,  1.0, -2.0, -2.0, -2.0, -1.0, -1.0,  0.0,  0.0,  0.0,  1.0,  1.0,  2.0,  2.0,  2.0,  1.0,  1.0,  1.0,  1.0]),
    ('nearest'      , 0, [-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -1.0, -1.0,  0.0,  0.0,  0.0,  1.0,  1.0,  2.0,  2.0,  2.0,  2.0,  2.0,  2.0,  2.0]),
    ('mirror'       , 0, [ 0.0,  0.0, -1.0, -1.0, -2.0, -2.0, -2.0, -1.0, -1.0,  0.0,  0.0,  0.0,  1.0,  1.0,  2.0,  2.0,  2.0,  1.0,  1.0,  0.0,  0.0]),
    ('reflect'      , 0, [-1.0, -1.0, -2.0, -2.0, -2.0, -2.0, -2.0, -1.0, -1.0,  0.0,  0.0,  0.0,  1.0,  1.0,  2.0,  2.0,  2.0,  2.0,  2.0,  1.0,  1.0]),
    ('grid-wrap'    , 0, [ 1.0,  1.0,  2.0,  2.0, -2.0, -2.0, -2.0, -1.0, -1.0,  0.0,  0.0,  0.0,  1.0,  1.0,  2.0,  2.0,  2.0, -2.0, -2.0, -1.0, -1.0]),
    ('grid-constant', 1, [ 1.0,  1.0,  1.0,  0.4, -0.8, -2.0, -1.6, -1.2, -0.8, -0.4, -0.0,  0.4,  0.8,  1.2,  1.6,  2.0,  1.6,  1.2,  1.0,  1.0,  1.0]),
    ('nearest'      , 1, [-2.0, -2.0, -2.0, -2.0, -2.0, -2.0, -1.6, -1.2, -0.8, -0.4, -0.0,  0.4,  0.8,  1.2,  1.6,  2.0,  2.0,  2.0,  2.0,  2.0,  2.0]),
    ('mirror'       , 1, [ 0.0, -0.4, -0.8, -1.2, -1.6, -2.0, -1.6, -1.2, -0.8, -0.4, -0.0,  0.4,  0.8,  1.2,  1.6,  2.0,  1.6,  1.2,  0.8,  0.4,  0.0]),
    ('reflect'      , 1, [-1.0, -1.4, -1.8, -2.0, -2.0, -2.0, -1.6, -1.2, -0.8, -0.4, -0.0,  0.4,  0.8,  1.2,  1.6,  2.0,  2.0,  2.0,  1.8,  1.4,  1.0]),
    ('grid-wrap'    , 1, [ 1.0,  1.4,  1.8,  1.2, -0.4, -2.0, -1.6, -1.2, -0.8, -0.4, -0.0,  0.4,  0.8,  1.2,  1.6,  2.0,  0.4, -1.2, -1.8, -1.4, -1.0]),
])
def test_affine_transform_1d(mode: str, order: int, expected: ArrayLike, backend: str):
    xp = get_backend_module(backend)

    in_ys = numpy.array([-2., -1., 0., 1., 2.])

    # interpolates at coords `numpy.linspace(-2., 6., 21, endpoint=True)`
    assert_array_almost_equal(numpy.array(expected), to_numpy(affine_transform(
        xp.array(in_ys), [0.4], -2.0,
        mode=t.cast(_BoundaryMode, mode), order=order, cval=1.0, output_shape=(21,)
    )), decimal=8)