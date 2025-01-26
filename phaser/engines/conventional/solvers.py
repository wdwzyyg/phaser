from functools import partial
from itertools import zip_longest
import logging
import typing as t

import numpy
from numpy.typing import NDArray

from phaser.utils.num import cast_array_module, at, abs2, fft2, ifft2, jit, check_finite
from phaser.utils.misc import create_compact_groupings, create_sparse_groupings
from phaser.hooks.solver import ConventionalSolver, ConventionalSolverArgs
from phaser.plan import LSQMLSolverPlan, EPIESolverPlan
from phaser.state import IterState, StateObserver
from phaser.engines.common.simulation import SimulationState, make_propagators, cutout_group, slice_forwards, slice_backwards


class LSQMLSolver(ConventionalSolver):
    def __init__(self, args: ConventionalSolverArgs, props: LSQMLSolverPlan):
        self.plan: LSQMLSolverPlan = props
        self.niter: int = args['niter']
        self.grouping: int = args['grouping']
        self.compact: bool = args['compact']

    def solve(self, sim: SimulationState, engine_i: int, observers: t.Sequence[StateObserver] = ()) -> SimulationState:
        logger = logging.getLogger(__name__)
        xp = cast_array_module(sim.xp)
        real_dtype = sim.dtype

        start_i = sim.state.iter.total_iter
        sim.state.iter = IterState(engine_i, 0, start_i)

        for observer in observers:
            observer(sim.state)

        if self.compact:
            groups = create_compact_groupings(sim.state.scan, self.grouping)
        else:
            groups = create_sparse_groupings(sim.state.scan, self.grouping)

        cutout_shape = sim.state.probe.data.shape[-2:]
        props = make_propagators(sim)

        obj_mag = xp.zeros(cutout_shape, dtype=real_dtype)
        probe_mag = xp.zeros_like(sim.state.object.data, dtype=real_dtype)

        # dry run to pre-compute obj_mag and probe_mag
        for (group_i, group) in enumerate(groups):
            (obj_mag, probe_mag) = group_dry_run(sim, props, group, obj_mag, probe_mag)

        logging.info("Pre-calculated intensities")

        # TODO: rescale probe intensity

        for i in range(self.niter):
            new_obj_mag = xp.zeros(cutout_shape, dtype=real_dtype)
            new_probe_mag = xp.zeros_like(sim.state.object.data, dtype=real_dtype)

            for (group_i, group) in enumerate(groups):
                #if group_i % 10:
                #    for observer in observers:
                #        observer(sim.state)
                (sim, obj_mag, probe_mag, new_obj_mag, new_probe_mag) = run_group(
                    sim, group, props=props,
                    obj_mag=obj_mag, probe_mag=probe_mag,
                    new_obj_mag=new_obj_mag, new_probe_mag=new_probe_mag,
                    beta_object=self.plan.beta_object,
                    beta_probe=self.plan.beta_probe,
                    illum_reg_object=self.plan.illum_reg_object,
                    illum_reg_probe=self.plan.illum_reg_probe,
                    gamma=self.plan.gamma,
                )
                check_finite(sim.state.object.data, sim.state.probe.data, context=f"object or probe, group {group_i}")

            obj_mag = new_obj_mag
            probe_mag = new_probe_mag

            logger.info(f"Finished iter {i+1:3}/{self.niter:3}")

            sim.state.iter = IterState(engine_i, i, start_i + i)
            for observer in observers:
                observer(sim.state)

        return sim


@partial(jit, donate_argnames=('obj_mag', 'probe_mag'))
def group_dry_run(
    sim: SimulationState,
    props: t.Optional[NDArray[numpy.complexfloating]],
    group: NDArray[numpy.integer],
    obj_mag: NDArray[numpy.floating],
    probe_mag: NDArray[numpy.floating]
) -> t.Tuple[NDArray[numpy.floating], NDArray[numpy.floating]]:
    xp = cast_array_module(sim.xp)
    (psi, group_obj, group_scan) = cutout_group(sim, group)

    obj_mag += xp.sum(abs2(xp.prod(group_obj, axis=1)), axis=0)
    obj_grid = sim.state.object.sampling

    def run_slice(slice_i: int, prop: t.Optional[NDArray[numpy.complexfloating]], state):
        (probe_mag, psi) = state
        probe_mag = at(probe_mag, slice_i).set(
            obj_grid.add_view_at_pos(probe_mag[slice_i], group_scan, xp.sum(abs2(psi), axis=1))
        )

        if prop is not None:
            psi = ifft2(fft2(psi * group_obj[:, slice_i, None]) * prop)

        return (probe_mag, psi)

    (probe_mag, psi) = slice_forwards(props, (probe_mag, psi), run_slice)

    return (obj_mag, probe_mag)

# TODO: pass LSQMLSolverPlan in here for parameters

@partial(jit, donate_argnames=('sim', 'obj_mag', 'probe_mag', 'new_obj_mag', 'new_probe_mag'))
def run_group(
    sim: SimulationState,
    group: NDArray[numpy.integer], *,
    props: t.Optional[NDArray[numpy.complexfloating]],
    obj_mag: NDArray[numpy.floating],
    probe_mag: NDArray[numpy.floating],
    new_obj_mag: NDArray[numpy.floating],
    new_probe_mag: NDArray[numpy.floating],
    beta_object: float = 0.9,
    beta_probe: float = 0.9,
    illum_reg_object: float,
    illum_reg_probe: float,
    gamma: float,
) -> t.Tuple[SimulationState, NDArray[numpy.floating], NDArray[numpy.floating], NDArray[numpy.floating], NDArray[numpy.floating]]:
    xp = cast_array_module(sim.xp)
    obj_grid = sim.state.object.sampling
    n_slices = sim.state.object.data.shape[0]

    eps = 1e-16

    (probes, group_obj, group_scan, subpx_filters) = cutout_group(sim, group, return_filters=True)
    psi = xp.zeros((n_slices, *probes.shape), dtype=probes.dtype)
    psi = at(psi, 0).set(probes)

    group_probe_mag = xp.zeros_like(probe_mag)
    group_obj_mag = xp.sum(abs2(xp.prod(group_obj, axis=1)), axis=0)

    def sim_slice(slice_i: int, prop: t.Optional[NDArray[numpy.complexfloating]], state):
        (group_probe_mag, psi) = state

        group_probe_mag = at(group_probe_mag, slice_i).set(
            obj_grid.add_view_at_pos(group_probe_mag[slice_i], group_scan, xp.sum(abs2(psi[slice_i]), axis=1))
        )

        if prop is not None:
            psi = at(psi, slice_i + 1).set(
                ifft2(fft2(psi[slice_i] * group_obj[:, slice_i, None]) * prop)
            )

        return (group_probe_mag, psi)

    (group_probe_mag, psi) = slice_forwards(props, (group_probe_mag, psi), sim_slice)

    new_obj_mag += group_obj_mag
    new_probe_mag += group_probe_mag

    model_wave = fft2(psi[-1] * group_obj[:, -1, None])
    # sum over incoherent modes
    model_intensity = xp.sum(abs2(model_wave), axis=1, keepdims=True)
    # experimental data
    group_patterns = xp.array(sim.patterns[*group])
    (chi, sim.noise_model_state) = sim.noise_model.calc_wave_update(model_wave, model_intensity, group_patterns, sim.pattern_mask, sim.noise_model_state)
    chi = ifft2(chi)

    def update_slice(slice_i: int, prop: t.Optional[NDArray[numpy.complexfloating]], state):
        (sim, chi) = state

        delta_O = xp.conj(psi[slice_i]) * chi
        delta_P = xp.conj(group_obj[:, slice_i, None]) * chi
        alpha_O = xp.sum(xp.sum(xp.real(chi * xp.conj(delta_O * psi[slice_i])), axis=(-1, -2), keepdims=True), axis=1) / (xp.sum(abs2(delta_O * psi[slice_i])) + gamma)

        # average object update
        delta_O_avg = xp.zeros_like(sim.state.object.data[0])
        delta_O_avg = obj_grid.add_view_at_pos(delta_O_avg, group_scan, xp.sum(delta_O, axis=1))
        delta_O_avg /= (probe_mag[slice_i] + illum_reg_object)

        obj_update = beta_object * xp.sum(alpha_O * delta_O_avg * group_probe_mag[slice_i], axis=0) / (group_probe_mag[slice_i] + eps)
        sim.state.object.data = at(sim.state.object.data, slice_i).add(obj_update)

        if prop is None:
            delta_P_avg = ifft2(xp.sum(fft2(delta_P) * subpx_filters.conj(), axis=0))
            delta_P_avg /= (obj_mag + illum_reg_probe)

            # update step per probe mode
            alpha_P = xp.sum(xp.real(chi * xp.conj(delta_P * group_obj[:, slice_i, None])), axis=(-1, -2), keepdims=True) / (xp.sum(abs2(delta_P * group_obj[:, slice_i, None])) + gamma)

            probe_update = beta_probe * xp.sum(alpha_P * delta_P_avg * group_obj_mag, axis=0) / (group_obj_mag + eps)
            sim.state.probe.data += probe_update

            #import jax
            #jax.debug.print("alpha_P: {}", alpha_P)
        else:
            chi = ifft2(fft2(delta_P) * prop.conj())

        return (sim, chi)

    (sim, chi) = slice_backwards(props, (sim, chi), update_slice)

    return (sim, obj_mag, probe_mag, new_obj_mag, new_probe_mag)


"""
class EPIESolver(ConventionalSolver):
    def __init__(self, args: ReconsState, props: EPIESolverPlan):
        ...

    def update_group_slice():
        pass

    def update_iteration():
        pass
"""