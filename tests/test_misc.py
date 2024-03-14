
import numpy

from phaser.utils.misc import create_groupings


def test_create_groupings():

    assert [arr.tolist() for arr in create_groupings(32, 8, 'seed1')] == [
        [[23, 31,  0, 30, 29,  3,  4, 15]],
        [[14,  6,  9, 28, 13, 18, 17, 26]],
        [[24, 19, 10, 27,  5,  8, 20, 22]],
        [[ 1, 12,  2, 11,  7, 21, 25, 16]]
    ]

    # distributes evenly
    assert [arr.tolist() for arr in create_groupings(33, 8, 'seed1')] == [
        [[24, 16,  0, 30, 25,  3,  4]],
        [[15, 14,  6,  9,  7, 13, 21]],
        [[19, 31, 28, 17, 23, 18, 10]],
        [[29, 27, 32,  5,  8, 20]],
        [[22,  1, 26, 12,  2, 11]]
    ]

    # multi-dimensional
    arr = numpy.zeros((32, 32, 4))

    groupings = create_groupings(arr.shape, 128)
    assert groupings[0].shape == (3, 128)
    # make sure we can index
    assert arr[*groupings[0]].shape == (128,)