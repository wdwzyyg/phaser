import logging
import typing as t

import numpy
from numpy.typing import NDArray

from phaser.utils.num import cast_array_module, to_numpy
from phaser.utils.object import ObjectSampling
from .plan import ReconsPlan, EnginePlan
from .state import ObjectState, ReconsState, IterState, ProgressState, StateObserver


def execute_plan(plan: ReconsPlan, observers: t.Sequence[StateObserver] = ()):
    import jax.numpy
    #import cupy

    logging.basicConfig(level=logging.INFO)

    logging.info("Executing plan...")

    seed = None
    dtype: type = numpy.float32 if plan.dtype == 'float32' else numpy.float64
    xp = cast_array_module(jax.numpy)
    logging.info(f"dtype: {dtype} array backend: {xp.__name__}")

    raw_data = plan.raw_data(None)

    wavelength = plan.wavelength or raw_data['wavelength']
    if wavelength is None:
        raise ValueError("`wavelength` must be specified by raw_data or manually")

    sampling = raw_data['sampling']
    patterns = raw_data['patterns']
    pattern_mask = raw_data['mask']

    # normalize pattern intensity
    patterns /= xp.mean(xp.sum(patterns, axis=(-1, -2)))

    logging.info("Initializing probe...")
    probe = plan.init_probe({'sampling': sampling, 'wavelength': wavelength, 'dtype': dtype, 'seed': seed, 'xp': xp})
    if probe.data.ndim == 2:
        probe.data = probe.data.reshape((1, *probe.data.shape))

    logging.info("Initializing scan...")
    if plan.init_scan is None:
        if raw_data['scan'] is None:
            raise ValueError("`init_scan` must be specified by raw_data or manually")
        scan = xp.array(raw_data['scan']).astype(dtype)
    else:
        scan = plan.init_scan({'dtype': dtype, 'seed': seed, 'xp': xp})

    obj_sampling = ObjectSampling.from_scan(scan, sampling.sampling, sampling.extent / 2. + 3. * sampling.sampling)

    logging.info("Initializing object...")
    obj = plan.init_object({
        'sampling': obj_sampling, 'slices': plan.slices, 'wavelength': wavelength,
        'dtype': dtype, 'seed': seed, 'xp': xp
    })
    if obj.data.ndim == 2:
        obj.data = obj.data.reshape((1, *obj.data.shape))
        obj.zs = numpy.array([0.], dtype=dtype)

    state = ReconsState(
        iter=IterState(0, 0, 0),
        probe=probe,
        object=obj,
        scan=scan,
        progress=ProgressState(iters=numpy.array([]), detector_errors=numpy.array([])),
        wavelength=wavelength
    )

    for (engine_i, engine) in enumerate(plan.engines):
        logging.info(f"Preparing for engine #{engine_i + 1}...")
        state, patterns = prepare_for_engine(state, patterns, t.cast(EnginePlan, engine.props))
        state = engine({
            'state': state,
            'patterns': patterns,
            'pattern_mask': pattern_mask,
            'dtype': dtype,
            'xp': xp,
            'engine_i': engine_i,
            'observers': observers,
        })

    logging.info("Reconstruction finished!")


def prepare_for_engine(state: ReconsState, patterns: NDArray[numpy.floating], engine: EnginePlan) -> t.Tuple[ReconsState, NDArray[numpy.floating]]:
    if engine.sim_shape is not None:
        if engine.sim_shape != state.probe.data.shape[-2:]:
            raise NotImplementedError()

    current_probe_modes = state.probe.data.shape[0]
    if engine.probe_modes != current_probe_modes:
        # fix probe modes
        if engine.probe_modes < current_probe_modes:
            # TODO: redistribute intensity here
            state.probe.data = state.probe.data[:engine.probe_modes]
        elif current_probe_modes == 1:
            from phaser.utils.optics import make_hermetian_modes

            state.probe.data = make_hermetian_modes(state.probe.data[0], engine.probe_modes, powers=0.1)
        else:
            raise NotImplementedError()

    if engine.slices is not None:
        if not numpy.allclose(engine.slices.zs, state.object.zs):
            # TODO this is a quick hack
            if len(state.object.zs) == 1:
                new_obj = numpy.pad(state.object.data, ((len(engine.slices.zs) - 1, 0), (0, 0), (0, 0)), constant_values=1.0)
                state.object = ObjectState(state.object.sampling, new_obj, numpy.array(engine.slices.zs))
            else:
                raise NotImplementedError()

    return state, patterns