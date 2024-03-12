
import numpy
from numpy.testing import assert_array_almost_equal

from .utils import with_backends, get_backend_module

from phaser.utils.num import to_numpy
from phaser.utils.object import random_phase_object


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