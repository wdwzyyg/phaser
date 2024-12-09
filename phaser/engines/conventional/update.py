import numpy
from numpy.typing import NDArray

from phaser.utils.num import get_array_module, abs2
from phaser.state import ReconsState
from phaser.hooks.conventional import Updater
from phaser.plan import LSQMLUpdate, EPIEUpdate


class LSQMLUpdater(Updater):
    def __init__(self, args: ReconsState, props: LSQMLUpdate):
        self.stochastic: bool = props.stochastic
        self.xp = get_array_module(args.object.data)

        self.gamma = 1e-4

    def update_group_slice(
        self, slice: int,
        group_objs: NDArray[numpy.complexfloating], group_probes: NDArray[numpy.complexfloating],
        chi: NDArray[numpy.complexfloating]
    ) -> NDArray[numpy.complexfloating]:
        xp = self.xp

        delta_O = chi * xp.conj(group_probes)
        delta_P = chi * xp.conj(group_objs)

        # calculate step size per probe position
        alpha_O = xp.sum(xp.sum(xp.real(chi * xp.conj(delta_O * group_probes)), axis=(-1, -2), keepdims=True), axis=1) / (xp.sum(abs2(delta_O * group_probes)) + gamma)

        # average object update direction
        delta_O_avg = xp.zeros_like(obj)
        # sum over probe modes as well
        delta_O_avg = obj_grid.add_view_at_pos(delta_O_avg, group_scan, xp.sum(delta_O, axis=1))
        delta_O_avg /= (probe_mag + illum_reg_O)

        # compute final object update, using per-position step size
        obj_update = xp.sum(alpha_O * delta_O_avg * group_probe_mag, axis=0) / (group_probe_mag + eps)

        return delta_P

    def update_iteration():
        pass


class EPIEUpdater(Updater):
    def __init__(self, args: ReconsState, props: LSQMLUpdate):
        ...

    def update_group_slice():
        pass

    def update_iteration():
        pass