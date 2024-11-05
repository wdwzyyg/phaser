import logging
import typing as t

import numpy

from phaser.utils.num import cast_array_module, fft2, ifft2, abs2
from phaser.utils.optics import fourier_shift_filter
from phaser.utils.misc import create_groupings
from phaser.hooks import EngineArgs
from phaser.plan import ConventionalEngine
from phaser.state import ReconsState, IterState, ProgressState, StateObserver
from .detector_models import modulus_constraint


def run_engine(args: EngineArgs, props: ConventionalEngine) -> ReconsState:
    logger = logging.getLogger(__name__)

    state = args['state']
    xp = cast_array_module(args['xp'])
    dtype = args['dtype']
    observers: t.Sequence[StateObserver] = args.get('observers', [])

    samp = state.probe.sampling
    obj_samp = state.object.sampling

    (ky, kx) = samp.recip_grid(dtype=dtype, xp=xp)

    patterns = args['patterns']

    logger.info(f"Starting engine #{args['engine_i'] + 1}...")
    start_iter = state.iter.total_iter

    alpha = 1.0
    beta = 0.5

    for iteration in range(props.niter):
        state.iter = IterState(args['engine_i'], iteration, iteration + start_iter)
        for observe in observers:
            observe(state)

        groups = create_groupings(patterns.shape[:-2], props.grouping or 64)

        iteration_mses = []

        for (group_i, group) in enumerate(groups):
            group_scan = state.scan[*group]

            # shape (len(group), 1, ny, nx)
            group_subpx_filters = fourier_shift_filter(ky, kx, obj_samp.get_subpx_shifts(group_scan, ky.shape))[:, None, ...]

            # shape (len(group), nmodes, ny, nx)
            group_probes = ifft2(group_subpx_filters * fft2(state.probe.data))

            group_objs = obj_samp.get_view_at_pos(state.object.data, group_scan, tuple(samp.shape))

            model_wave = fft2(group_objs * group_probes)
            model_intensity = xp.sum(abs2(model_wave), axis=1)

            group_patterns = patterns[*group]

            iteration_mses.append(float(xp.mean(xp.sum(abs2(model_intensity - group_patterns), axis=(-1, -2)))))

            wave_diff = ifft2(modulus_constraint(xp, model_wave, model_intensity, patterns[*group], props.detector_model))

            # average object update across incoherent modes
            group_obj_update = alpha * xp.mean(group_probes.conj() / xp.max(abs2(group_probes), axis=(-1, -2), keepdims=True) * wave_diff, axis=1)

            # average probe update in group
            probe_update = beta * group_objs.conj() / xp.max(abs2(group_objs), axis=(-1, -2), keepdims=True) * wave_diff
            probe_update = ifft2(xp.mean(fft2(probe_update) / group_subpx_filters, axis=0))

            obj_update = xp.zeros_like(state.object.data)
            obj_update = obj_samp.set_view_at_pos(obj_update, group_scan, group_obj_update)

            state.object.data += obj_update
            state.probe.data += probe_update

        iteration_mse = numpy.mean(iteration_mses)
        state.progress = ProgressState(
            numpy.concatenate([state.progress.iters, [iteration + start_iter]]),
            numpy.concatenate([state.progress.detector_errors, [iteration_mse]]),
        )

        logger.info(f"Iteration {iteration} finished, MSE: {iteration_mse}")

    return args['state']