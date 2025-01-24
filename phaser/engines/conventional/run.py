import logging
import typing as t

import numpy
from numpy.typing import NDArray

from phaser.utils.num import cast_array_module, fft2, ifft2, abs2, jit
from phaser.utils.optics import fourier_shift_filter, fresnel_propagator
from phaser.utils.misc import create_compact_groupings, create_sparse_groupings
from phaser.hooks import EngineArgs
from phaser.plan import ConventionalEnginePlan
from phaser.engines.common.simulation import SimulationState
from phaser.state import ReconsState, IterState, ProgressState, StateObserver

def run_engine(args: EngineArgs, props: ConventionalEnginePlan) -> ReconsState:
    logger = logging.getLogger(__name__)

    xp = cast_array_module(args['xp'])
    dtype = args['dtype']
    observers: t.Sequence[StateObserver] = args.get('observers', [])

    logger.info(f"Starting engine #{args['engine_i'] + 1}...")

    noise_model = props.noise_model(None)

    sim = SimulationState(
        state=args['state'], noise_model=noise_model,
        patterns=args['patterns'], pattern_mask=args['pattern_mask'],
        xp=xp, dtype=dtype
    )

    solver = props.solver({
        'niter': props.niter,
        'compact': props.compact,
        'grouping': props.grouping or 64,
    })

    sim = solver.solve(sim, observers=observers, engine_i=args['engine_i'])

    return sim.state

    # #if props.compact:
    # #    groups = create_compact_groupings(patterns.shape[:-2], grouping)
    # #else:
    # #    groups = create_sparse_groupings(patterns.shape[:-2], grouping)

    # (probe_scale, engine_state['solver']) = solver.dry_run(
    #     ky, kx, state, patterns, groups,
    #     engine_state['solver']
    # )
    # logger.info(f"Rescaling incident probe by {probe_scale:.2f}x...")
    # state.probe.data *= probe_scale

    # for iteration in range(props.niter):
    #     sim.state.iter = IterState(args['engine_i'], iteration, iteration + start_iter)
    #     for observe in observers:
    #         observe(state)

    #     losses = []

    #     (probe_scale, engine_state['solver']) = solver.iter(
    #         ky, kx, state, patterns, groups,
    #         engine_state['solver']
    #     )

    #     for (group_i, group) in enumerate(groups):
    #         group_scan = state.scan[*group]
    #         group_obj = state.object.sampling.get_view_at_pos(state.object.data, group_scan, tuple(state.probe.sampling.shape))

    #         (model_wave, engine_state) = group_forward_model(ky, kx, state, group, engine_state)
    #         model_intensity = xp.sum(abs2(model_wave), axis=1)
    #         exp_patterns = patterns[*group]

    #         (loss, noise_model_state) = noise_model.calc_loss(model_wave, model_intensity, exp_patterns, args['pattern_mask'], noise_model_state)
    #         losses.append(float(loss))

    #         (wave_diff, noise_model_state) = noise_model.calc_wave_update(model_wave, model_intensity, exp_patterns, args['pattern_mask'], noise_model_state)
    #         wave_diff = ifft2(wave_diff)

    #         if check_nan(wave_diff):
    #             raise ValueError("NaN encountered in wave update")

    #         for slice_i in reversed(range(obj.shape[0])):
    #             obj.at[slice_i] = solver.update_group_slice



    #         # average object update across incoherent modes
    #         group_obj_update = alpha * xp.mean(group_probes.conj() / xp.max(abs2(group_probes), axis=(-1, -2), keepdims=True) * wave_diff, axis=1)

    #         # average probe update in group
    #         probe_update = beta * group_objs.conj() / xp.max(abs2(group_objs), axis=(-1, -2), keepdims=True) * wave_diff
    #         probe_update = ifft2(xp.mean(fft2(probe_update) / group_subpx_filters, axis=0))

    #         obj_update = xp.zeros_like(state.object.data)
    #         obj_update = obj_samp.set_view_at_pos(obj_update, group_scan, group_obj_update)

    #         state.object.data += obj_update
    #         state.probe.data += probe_update

    #     iteration_loss = numpy.mean(iteration_mses)
    #     state.progress = ProgressState(
    #         numpy.concatenate([state.progress.iters, [iteration + start_iter]]),
    #         numpy.concatenate([state.progress.detector_errors, [iteration_loss]]),
    #     )

    #     logger.info(f"Iteration {iteration} finished, loss: {iteration_loss}")

    # return args['state']


