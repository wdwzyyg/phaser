
import numpy
from numpy.testing import assert_array_almost_equal, assert_array_equal
import pytest

from .utils import with_backends, get_backend_module

from phaser.utils.num import to_numpy
from phaser.utils.object import random_phase_object, ObjectSampling


@with_backends('cpu', 'cuda')
def test_random_phase_object(backend: str):
    xp = get_backend_module(backend)

    obj = random_phase_object((8, 8), 1e-4, seed=2620771887, dtype=numpy.complex64, xp=xp)

    assert obj.dtype == numpy.complex64
    assert_array_almost_equal(to_numpy(obj), numpy.array([
        [1.-1.5272086e-05j, 1.+1.0225522e-04j, 1.-8.0865902e-05j, 1.-1.7328106e-05j, 1.-1.2898073e-04j, 1.+2.2908196e-05j, 1.+8.1173976e-06j, 1.+2.1377344e-05j],
        [1.+7.4363430e-05j, 1.-9.1323782e-05j, 1.-2.0272582e-04j, 1.-4.8823396e-05j, 1.+9.3021641e-05j, 1.+1.0718761e-04j, 1.+5.0221975e-06j, 1.-5.5743083e-05j],
        [1.+9.5888179e-05j, 1.+7.0838556e-05j, 1.-1.1567964e-04j, 1.+1.3202346e-04j, 1.+1.3625837e-04j, 1.-5.2489726e-05j, 1.-1.3756646e-04j, 1.+2.8579381e-05j],
        [1.-6.3651263e-05j, 1.-4.5127890e-05j, 1.+5.5954431e-05j, 1.-1.8197308e-04j, 1.+6.3579530e-05j, 1.-4.6506138e-05j, 1.+5.1510222e-05j, 1.+1.0700211e-04j],
        [0.99999994-2.6713090e-04j, 1.+1.8953861e-04j, 1.+1.1097628e-04j, 1.-5.9648257e-05j, 1.+4.2086729e-05j, 1.+6.0222395e-05j, 1.-8.4926840e-05j, 0.99999994-2.6520275e-04j],
        [1.-4.1160638e-05j, 1.+8.4538617e-05j, 1.+4.1620955e-05j, 1.+1.6012797e-05j, 1.-1.3888512e-05j, 1.+9.1871625e-06j, 1.+5.0595980e-05j, 1.+2.3048995e-04j],
        [1.-2.8506602e-05j, 1.+2.4769653e-05j, 1.-2.3920753e-05j, 1.+8.0796681e-06j, 0.99999994-2.5373933e-04j, 1.+6.9838488e-06j, 1.+3.8624425e-05j, 1.+1.1229565e-04j],
        [1.-4.8519720e-05j, 1.+9.4494520e-05j, 1.+4.9148810e-05j, 1.-1.3229759e-04j, 1.-2.6898948e-05j, 1.-9.8376579e-05j, 1.+6.9485272e-05j, 1.-9.8597156e-05j]
    ], dtype=numpy.complex64))


def test_object_sampling_grid():
    # asymmetric, mixed even & odd sampling grid
    samp = ObjectSampling((16, 15), (0.5, 2.0))

    assert_array_equal(samp.shape, [16, 15])
    assert_array_almost_equal(samp.sampling, [0.5, 2.0])
    # centered around (0, 0)
    assert_array_almost_equal(samp.corner, [-3.75, -14.0])
    assert_array_almost_equal(samp.min, [-3.75, -14.0])
    assert_array_almost_equal(samp.max, [3.75, 14.0])

    assert samp.region_min is None
    assert samp.region_max is None

    assert samp.mpl_extent() == pytest.approx((-15.0, 15.0, 4.0, -4.0))
    assert samp.mpl_extent(False) == pytest.approx((-14.0, 16.0, 4.25, -3.75))

    yy, xx = samp.grid()

    assert_array_almost_equal(xx[0, :], [
        -14., -12., -10., -8., -6., -4., -2., 0., 2., 4., 6., 8., 10., 12., 14.
    ])
    assert_array_almost_equal(yy[:, 0], [
        -3.75, -3.25, -2.75, -2.25, -1.75, -1.25, -0.75, -0.25,  0.25, 0.75,  1.25,  1.75,  2.25,  2.75,  3.25,  3.75
    ])

    assert samp.get_region_crop() == (slice(None, None), slice(None, None))


def test_object_from_scan():
    scan = numpy.array([
        [-4.8, -1.],
        [2.1, 1.],
        [4.19, 9.9],
        [0., numpy.nan],
    ])

    samp = ObjectSampling.from_scan(scan, sampling=[0.5, 0.5], pad=2.)

    assert_array_almost_equal(samp.min, [-6.8, -3.])
    assert_array_almost_equal(samp.max, [6.2, 12.0])
    assert_array_almost_equal(samp.shape, (27, 31))

    assert_array_almost_equal(samp.region_min, [-4.8, -1.])  # type: ignore
    assert_array_almost_equal(samp.region_max, [4.19, 9.9])  # type: ignore

    assert samp.get_region_crop() == (slice(4, 22), slice(4, 26))


def test_object_slicing():
    samp = ObjectSampling((40, 31), (0.5, 2.0))

    (yy, xx) = samp.grid()

    # y pixels on 0.25 + 0.5 * i
    # x pixels on 2 * j

    # when exact pixel coord, should round-trip
    yp, xp = numpy.round(samp._pos_to_object_idx(numpy.array((-1.25, 12.0)), (1, 1))).astype(numpy.int_)
    assert yy[yp, xp] == pytest.approx(-1.25)
    assert xx[yp, xp] == pytest.approx(12.0)

    # should get the same pixel when perturbed by <1/2 sample
    assert_array_equal(numpy.round(samp._pos_to_object_idx(numpy.array((-1.25 + 0.24, 12.0 - 0.99)), (1, 1))).astype(numpy.int_), (yp, xp))

    # odd cutout n*2 + 1, n samples away, same result
    n = 2
    pos = numpy.array((-1.25 - 0.24 + 0.5 * n, 12.0 - 0.99 + 2.0 * n))
    cutout_shape = (1 + 2*n, 1 + 2*n)
    assert_array_equal(numpy.round(samp._pos_to_object_idx(pos, cutout_shape)).astype(numpy.int_), (yp, xp))

    # check slicing as well
    assert samp.slice_at_pos(pos, cutout_shape) == (
        slice(yp, yp + 1 + 2*n),
        slice(xp, xp + 1 + 2*n)
    )

    n = 6
    # even cutout n*2, n - 1/2 samples away, same result
    pos = numpy.array((-1.25 - 0.24 + 0.5 * (n - 1/2.), 12.0 - 0.99 + 2.0 * (n - 1/2.)))
    cutout_shape = (2*n, 2*n)
    assert_array_equal(numpy.round(samp._pos_to_object_idx(pos, cutout_shape)).astype(numpy.int_), (yp, xp))

    # check slicing as well
    assert samp.slice_at_pos(pos, cutout_shape) == (
        slice(yp, yp + 2*n),
        slice(xp, xp + 2*n)
    )
