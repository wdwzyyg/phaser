from functools import partial
from itertools import zip_longest
import logging
import math
import typing as t

import numpy
from numpy.typing import NDArray

from phaser.utils.num import cast_array_module, at, abs2, fft2, ifft2, jit, check_finite, to_complex_dtype, to_numpy
from phaser.utils.misc import create_compact_groupings, create_sparse_groupings, mask_fraction_of_groups
from phaser.hooks.solver import ConstraintRegularizer, ConventionalSolver
from phaser.plan import ConventionalEnginePlan, LSQMLSolverPlan, EPIESolverPlan
from phaser.execute import Observer, process_flag
from phaser.engines.common.simulation import SimulationState, make_propagators, cutout_group, slice_forwards, slice_backwards


class LSQMLSolver(ConventionalSolver):
    def __init__(self, plan: ConventionalEnginePlan, props: LSQMLSolverPlan):
        self.plan: LSQMLSolverPlan = props
        self.engine_plan: ConventionalEnginePlan = plan

    def solve(
        self, sim: SimulationState, engine_i: int, observer: Observer
    ) -> SimulationState:
        logger = logging.getLogger(__name__)
        xp = cast_array_module(sim.xp)
        real_dtype = sim.dtype

        update_probe = process_flag(self.engine_plan.update_probe)
        update_object = process_flag(self.engine_plan.update_object)
        update_positions = process_flag(self.engine_plan.update_positions)
        calc_error = process_flag(self.engine_plan.calc_error)

        position_solver = None if self.engine_plan.position_solver is None else self.engine_plan.position_solver(None)
        position_solver_state = None if position_solver is None else position_solver.init_state(sim)

        if self.engine_plan.compact:
            groups = create_compact_groupings(sim.state.scan, self.engine_plan.grouping or 64)
        else:
            groups = create_sparse_groupings(sim.state.scan, self.engine_plan.grouping or 64)

        calc_error_mask = mask_fraction_of_groups(len(groups), self.engine_plan.calc_error_fraction)

        cutout_shape = sim.state.probe.data.shape[-2:]
        props = make_propagators(sim)

        observer.init_solver(sim.state, engine_i)

        obj_mag = xp.zeros(cutout_shape, dtype=real_dtype)
        probe_mag = xp.zeros_like(sim.state.object.data, dtype=real_dtype)

        rescale_factors = []

        # dry run to pre-compute obj_mag and probe_mag
        for (group_i, group) in enumerate(groups):
            (obj_mag, probe_mag, group_rescale_factors) = lsqml_dry_run(sim, props, group, obj_mag, probe_mag)

            rescale_factors.append(to_numpy(group_rescale_factors))

        rescale_factors = numpy.concatenate(rescale_factors, axis=0)
        rescale_factor = numpy.mean(rescale_factors)

        logger.info("Pre-calculated intensities")
        logger.info(f"Rescaling initial probe intensity by {rescale_factor:.2e}")
        sim.state.probe.data *= numpy.sqrt(rescale_factor)
        probe_mag *= rescale_factor

        observer.start_solver()

        for i in range(1, self.engine_plan.niter+1):
            new_obj_mag = xp.zeros_like(obj_mag)
            new_probe_mag = xp.zeros_like(probe_mag)
            iter_update_object = update_object({'state': sim.state, 'niter': self.engine_plan.niter})
            iter_update_probe = update_probe({'state': sim.state, 'niter': self.engine_plan.niter})
            iter_update_positions = update_positions({'state': sim.state, 'niter': self.engine_plan.niter})
            iter_calc_error = calc_error({'state': sim.state, 'niter': self.engine_plan.niter})
            iter_errors = []
            pos_update = xp.zeros_like(sim.state.scan)

            for (group_i, group) in enumerate(groups):
                group_calc_error = iter_calc_error and calc_error_mask[group_i]

                (sim, obj_mag, probe_mag, new_obj_mag, new_probe_mag, errors, group_pos_update) = lsqml_run(
                    sim, group, props=props,
                    obj_mag=obj_mag, probe_mag=probe_mag,
                    new_obj_mag=new_obj_mag, new_probe_mag=new_probe_mag,
                    beta_object=self.plan.beta_object,
                    beta_probe=self.plan.beta_probe,
                    update_object=iter_update_object,
                    update_probe=iter_update_probe,
                    update_position=iter_update_positions,
                    calc_error=group_calc_error,
                    illum_reg_object=self.plan.illum_reg_object,
                    illum_reg_probe=self.plan.illum_reg_probe,
                    gamma=self.plan.gamma,
                )
                check_finite(sim.state.object.data, sim.state.probe.data, context=f"object or probe, group {group_i}")

                sim = apply_regularizers_group(sim, group)

                if iter_update_positions:
                    assert group_pos_update is not None
                    pos_update[*group] = group_pos_update

                assert sim.state.object.data.dtype == to_complex_dtype(sim.dtype)
                assert sim.state.probe.data.dtype == to_complex_dtype(sim.dtype)

                observer.update_group(sim.state, self.engine_plan.send_every_group)

                if group_calc_error:
                    assert errors is not None
                    iter_errors.append(errors)

            obj_mag = new_obj_mag
            probe_mag = new_probe_mag

            if iter_update_positions:
                if not position_solver:
                    raise ValueError("Updating positions with no PositionSolver specified")

                # subtract mean position update
                pos_update -= xp.mean(pos_update, tuple(range(pos_update.ndim - 1)))
                pos_update, position_solver_state = position_solver.perform_update(sim.state.scan, pos_update, position_solver_state)
                # subtract mean again (this can change with momentum)
                pos_update -= xp.mean(pos_update, tuple(range(pos_update.ndim - 1)))
                update_mag = xp.linalg.norm(pos_update, axis=-1, keepdims=True)
                logger.info(f"Position update: mean {xp.mean(update_mag)}")
                sim.state.scan += pos_update

            sim = apply_regularizers_iter(sim)

            error = None
            if iter_calc_error:
                error = float(to_numpy(xp.nanmean(xp.concatenate(iter_errors))))  # type: ignore

                # TODO don't do this
                sim.state.progress.iters = numpy.concatenate([sim.state.progress.iters, [i]])
                sim.state.progress.detector_errors = numpy.concatenate([sim.state.progress.detector_errors, [error]])

            observer.update_iteration(sim.state, i, self.engine_plan.niter, error)

        observer.finish_solver()

        return sim


@partial(jit, donate_argnames=('obj_mag', 'probe_mag'))
def lsqml_dry_run(
    sim: SimulationState,
    props: t.Optional[NDArray[numpy.complexfloating]],
    group: NDArray[numpy.integer],
    obj_mag: NDArray[numpy.floating],
    probe_mag: NDArray[numpy.floating]
) -> t.Tuple[NDArray[numpy.floating], NDArray[numpy.floating], NDArray[numpy.floating]]:
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

    # modeled and experimental intensity
    # summed over incoherent modes and over the pattern
    model_intensity = xp.sum(abs2(fft2(psi)), axis=(1, -2, -1))
    exp_intensity = xp.sum(xp.array(sim.patterns[*group]), axis=(-2, -1))

    return (obj_mag, probe_mag, exp_intensity / model_intensity)

# TODO: pass LSQMLSolverPlan in here for parameters

@partial(
    jit,
    donate_argnames=('sim', 'obj_mag', 'probe_mag', 'new_obj_mag', 'new_probe_mag'),
    static_argnames=('update_object', 'update_probe', 'update_position', 'calc_error'),
)
def lsqml_run(
    sim: SimulationState,
    group: NDArray[numpy.integer], *,
    props: t.Optional[NDArray[numpy.complexfloating]],
    obj_mag: NDArray[numpy.floating],
    probe_mag: NDArray[numpy.floating],
    new_obj_mag: NDArray[numpy.floating],
    new_probe_mag: NDArray[numpy.floating],
    beta_object: float = 0.9,
    beta_probe: float = 0.9,
    update_object: bool = True,
    update_probe: bool = True,
    update_position: bool = True,
    calc_error: bool = True,
    illum_reg_object: float,
    illum_reg_probe: float,
    gamma: float,
) -> t.Tuple[SimulationState, NDArray[numpy.floating], NDArray[numpy.floating], NDArray[numpy.floating], NDArray[numpy.floating], t.Optional[NDArray[numpy.floating]], t.Optional[NDArray[numpy.floating]]]:
    xp = cast_array_module(sim.xp)
    obj_grid = sim.state.object.sampling
    n_slices = sim.state.object.data.shape[0]

    eps = 1e-16

    (probes, group_obj, group_scan, subpx_filters) = cutout_group(sim, group, return_filters=True)
    psi = xp.zeros((n_slices, *probes.shape), dtype=probes.dtype)
    psi = at(psi, 0).set(probes)

    group_probe_mag = xp.zeros_like(probe_mag)
    #group_obj_mag = xp.sum(abs2(group_obj[:, 0]), axis=0)
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
    group_patterns = xp.array(sim.patterns[*group])[:, None]

    errors = xp.sqrt(xp.nansum((model_intensity - group_patterns)**2, axis=(1, -1, -2))) if calc_error else None

    (chi, sim.noise_model_state) = sim.noise_model.calc_wave_update(model_wave, model_intensity, group_patterns, sim.pattern_mask, sim.noise_model_state)
    chi = ifft2(chi)

    def update_slice(slice_i: int, prop: t.Optional[NDArray[numpy.complexfloating]], state):
        (sim, chi) = state

        delta_P = chi * xp.conj(group_obj[:, slice_i, None])

        if update_object:
            delta_O = chi * xp.conj(psi[slice_i])
            alpha_O = xp.sum(xp.sum(xp.real(chi * xp.conj(delta_O * psi[slice_i])), axis=(-1, -2), keepdims=True), axis=1) / (xp.sum(abs2(delta_O * psi[slice_i])) + gamma)

            # average object update
            delta_O_avg = xp.zeros_like(sim.state.object.data[0])
            delta_O_avg = obj_grid.add_view_at_pos(delta_O_avg, group_scan, xp.sum(delta_O, axis=1))
            delta_O_avg /= (group_probe_mag[slice_i] + illum_reg_object)

            obj_update = beta_object * xp.sum(alpha_O * delta_O_avg * group_probe_mag[slice_i], axis=0) / (group_probe_mag[slice_i] + eps)
            sim.state.object.data = at(sim.state.object.data, slice_i).add(obj_update)

        if prop is not None:
            chi = ifft2(fft2(delta_P) * prop.conj())
        elif update_probe:
            delta_P_avg = ifft2(xp.sum(fft2(delta_P) * subpx_filters.conj(), axis=0))
            delta_P_avg /= (group_obj_mag + illum_reg_probe)

            # update step per probe mode
            alpha_P = xp.sum(xp.real(chi * xp.conj(delta_P * group_obj[:, slice_i, None])), axis=(-1, -2), keepdims=True) / (xp.sum(abs2(delta_P * group_obj[:, slice_i, None])) + gamma)

            probe_update = beta_probe * xp.sum(alpha_P * delta_P_avg * group_obj_mag, axis=0) / (group_obj_mag + eps)
            sim.state.probe.data += probe_update

        return (sim, chi)

    (sim, chi) = slice_backwards(props, (sim, chi), update_slice)

    if update_position:
        def calc_pos_step(probes_fft: NDArray[numpy.complexfloating], kx: NDArray[numpy.floating]) -> NDArray[numpy.floating]:
            delta_P_x = ifft2(probes_fft * -2.j*numpy.pi * kx)

            prod = delta_P_x * group_obj[:, 0, None]
            alpha = xp.sum(xp.real(chi * xp.conj(prod)), axis=(1, -1, -2)) / xp.sum(abs2(prod))
            return alpha

        # update directions
        probes_fft = fft2(probes)
        probes_fft /= xp.sum(abs2(probes), axis=(1, -1, -2), keepdims=True)
        pos_update = xp.stack(tuple(calc_pos_step(probes_fft, k) for k in (sim.ky, sim.kx)), axis=-1)
    else:
        pos_update = None

    return (sim, obj_mag, probe_mag, new_obj_mag, new_probe_mag, errors, pos_update)


class EPIESolver(ConventionalSolver):
    def __init__(self, engine_plan: ConventionalEnginePlan, props: EPIESolverPlan):
        self.engine_plan: ConventionalEnginePlan = engine_plan
        self.plan: EPIESolverPlan = props

    def solve(
        self, sim: SimulationState, engine_i: int, observer: Observer
    ) -> SimulationState:
        logger = logging.getLogger(__name__)
        xp = cast_array_module(sim.xp)

        update_probe = process_flag(self.engine_plan.update_probe)
        update_object = process_flag(self.engine_plan.update_object)
        calc_error = process_flag(self.engine_plan.calc_error)

        observer.init_solver(sim.state, engine_i)

        if self.engine_plan.compact:
            groups = create_compact_groupings(sim.state.scan, self.engine_plan.grouping or 64)
        else:
            groups = create_sparse_groupings(sim.state.scan, self.engine_plan.grouping or 64)

        calc_error_mask = mask_fraction_of_groups(len(groups), self.engine_plan.calc_error_fraction)

        props = make_propagators(sim)

        # dry run to pre-compute obj_mag and probe_mag
        rescale_factors = []
        for (group_i, group) in enumerate(groups):
            group_rescale_factors = epie_dry_run(sim, props, group)
            rescale_factors.append(to_numpy(group_rescale_factors))

        rescale_factors = numpy.concatenate(rescale_factors, axis=0)
        rescale_factor = numpy.mean(rescale_factors)

        logger.info("Pre-calculated intensities")
        logger.info(f"Rescaling initial probe intensity by {rescale_factor:.2e}")
        sim.state.probe.data *= numpy.sqrt(rescale_factor)

        observer.start_solver()

        for i in range(1, self.engine_plan.niter+1):
            iter_update_object = update_object({'state': sim.state, 'niter': self.engine_plan.niter})
            iter_update_probe = update_probe({'state': sim.state, 'niter': self.engine_plan.niter})
            iter_calc_error = calc_error({'state': sim.state, 'niter': self.engine_plan.niter})
            iter_errors = []

            for (group_i, group) in enumerate(groups):
                group_calc_error = iter_calc_error and calc_error_mask[group_i]

                (sim, errors) = epie_run(
                    sim, group, props=props,
                    beta_object=self.plan.beta_object,
                    beta_probe=self.plan.beta_probe,
                    update_object=iter_update_object,
                    update_probe=iter_update_probe,
                )
                check_finite(sim.state.object.data, sim.state.probe.data, context=f"object or probe, group {group_i}")

                sim = apply_regularizers_group(sim, group)
                assert sim.state.object.data.dtype == to_complex_dtype(sim.dtype)
                assert sim.state.probe.data.dtype == to_complex_dtype(sim.dtype)

                observer.update_group(sim.state, self.engine_plan.send_every_group)

                if group_calc_error:
                    assert errors is not None
                    iter_errors.append(errors)

            sim = apply_regularizers_iter(sim)

            error = None
            if iter_calc_error:
                error = float(to_numpy(xp.nanmean(xp.concatenate(iter_errors))))  # type: ignore

                # TODO don't do this
                sim.state.progress.iters = numpy.concatenate([sim.state.progress.iters, [i]])
                sim.state.progress.detector_errors = numpy.concatenate([sim.state.progress.detector_errors, [error]])

            observer.update_iteration(sim.state, i, self.engine_plan.niter, error)

        observer.finish_solver()

        return sim


@partial(jit)
def epie_dry_run(
    sim: SimulationState,
    props: t.Optional[NDArray[numpy.complexfloating]],
    group: NDArray[numpy.integer],
) -> NDArray[numpy.floating]:
    xp = cast_array_module(sim.xp)
    (psi, group_obj, group_scan) = cutout_group(sim, group)

    def run_slice(slice_i: int, prop: t.Optional[NDArray[numpy.complexfloating]], psi):
        if prop is not None:
            psi = ifft2(fft2(psi * group_obj[:, slice_i, None]) * prop)

        return psi

    psi = slice_forwards(props, psi, run_slice)

    # modeled and experimental intensity
    # summed over incoherent modes and over the pattern
    model_intensity = xp.sum(abs2(fft2(psi)), axis=(1, -2, -1))
    exp_intensity = xp.sum(xp.array(sim.patterns[*group]), axis=(-2, -1))

    return exp_intensity / model_intensity


@partial(jit, donate_argnames=('sim',), static_argnames=('update_object', 'update_probe', 'calc_error'))
def epie_run(
    sim: SimulationState,
    group: NDArray[numpy.integer], *,
    props: t.Optional[NDArray[numpy.complexfloating]],
    beta_object: float = 0.9,
    beta_probe: float = 0.9,
    update_object: bool = True,
    update_probe: bool = True,
    calc_error: bool = True,
) -> t.Tuple[SimulationState, t.Optional[NDArray[numpy.floating]]]:
    xp = cast_array_module(sim.xp)
    obj_grid = sim.state.object.sampling
    n_slices = sim.state.object.data.shape[0]

    (probes, group_obj, group_scan, subpx_filters) = cutout_group(sim, group, return_filters=True)
    psi = xp.zeros((n_slices, *probes.shape), dtype=probes.dtype)
    psi = at(psi, 0).set(probes)

    def sim_slice(slice_i: int, prop: t.Optional[NDArray[numpy.complexfloating]], psi):
        if prop is not None:
            psi = at(psi, slice_i + 1).set(
                ifft2(fft2(psi[slice_i] * group_obj[:, slice_i, None]) * prop)
            )

        return psi

    psi = slice_forwards(props, psi, sim_slice)

    model_wave = fft2(psi[-1] * group_obj[:, -1, None])
    # sum over incoherent modes
    model_intensity = xp.sum(abs2(model_wave), axis=1, keepdims=True)
    # experimental data
    group_patterns = xp.array(sim.patterns[*group])[:, None]

    errors = xp.sqrt(xp.nansum((model_intensity - group_patterns)**2, axis=(1, -1, -2))) if calc_error else None

    (chi, sim.noise_model_state) = sim.noise_model.calc_wave_update(model_wave, model_intensity, group_patterns, sim.pattern_mask, sim.noise_model_state)
    chi = ifft2(chi)

    def update_slice(slice_i: int, prop: t.Optional[NDArray[numpy.complexfloating]], state):
        (sim, chi) = state

        probe_update = group_obj[:, slice_i, None].conj() * chi / xp.max(
            abs2(group_obj[:, slice_i, None]),
            axis=(-1, -2), keepdims=True
        )

        if update_object:
            # average incoherent modes
            group_obj_update = beta_object/n_slices * xp.sum(psi[slice_i].conj() * chi, axis=1) / xp.max(
                xp.sum(abs2(psi[slice_i]), axis=1),
                axis=(-1, -2), keepdims=True
            )

            sim.state.object.data = at(sim.state.object.data, slice_i).set(
                obj_grid.add_view_at_pos(sim.state.object.data[slice_i], group_scan, group_obj_update)
            )

        if prop is not None:
            chi = ifft2(fft2(probe_update) * prop.conj())
        elif update_probe:
            # average probe updates in group
            probe_update = ifft2(xp.mean(fft2(probe_update) * subpx_filters.conj(), axis=0))
            sim.state.probe.data += beta_probe * probe_update

        return (sim, chi)

    (sim, chi) = slice_backwards(props, (sim, chi), update_slice)

    return (sim, errors)


def apply_regularizers_group(sim: SimulationState, group: NDArray[numpy.integer]) -> SimulationState:

    def apply_reg(reg: ConstraintRegularizer, state: t.Any):
        nonlocal sim
        (sim, state) = reg.apply_group(group, sim, state)
        return state

    sim.regularizer_states = tuple(
        apply_reg(reg, state) for (reg, state) in zip(sim.regularizers, sim.regularizer_states)
    )

    return sim


def apply_regularizers_iter(sim: SimulationState) -> SimulationState:

    def apply_reg(reg: ConstraintRegularizer, state: t.Any):
        nonlocal sim
        (sim, state) = reg.apply_iter(sim, state)
        return state

    sim.regularizer_states = tuple(
        apply_reg(reg, state) for (reg, state) in zip(sim.regularizers, sim.regularizer_states)
    )

    return sim