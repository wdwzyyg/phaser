#!/usr/bin/env python3

import itertools
import functools
import logging
import time
import sys
import json
import typing as t

import polars
import pane

from phaser.utils.num import get_backend_module, to_numpy, Sampling
from phaser.plan import ReconsPlan, EnginePlan, EngineHook, BackendName
from phaser.state import ReconsState, IterState, PartialReconsState, Patterns
from phaser.execute import Observer, initialize_reconstruction, prepare_for_engine


class BenchmarkObserver(Observer):
    def __init__(self, n_warmup: int = 2):
        self.n_warmup: int = n_warmup
        self.iter_times: t.List[float] = []
        super().__init__()

    def update_iteration(self, state: t.Union[ReconsState, PartialReconsState],
                         i: int, n: int, error: t.Optional[float] = None):
        finish_time = time.monotonic()

        if self.iter_start_time is not None:
            delta = finish_time - self.iter_start_time
            time_s = f" [{self._format_mmss(delta)}]"

            if i > self.n_warmup:
                self.iter_times.append(delta)
        else:
            time_s = ""

        w = len(str(n))

        error_s = f" Error: {error:.3e}" if error is not None else ""
        logging.info(f"Finished iter {i:{w}}/{n}{time_s}{error_s}")

        state.iter = IterState(self.engine_i, i + 1, self.start_iter + i + 1)
        self.iter_start_time = finish_time


@functools.lru_cache(1)
def initialize(sim_size: int = 128) -> t.Tuple[ReconsPlan, Patterns, ReconsState]:
    plan = ReconsPlan.from_data({
        "name": "mos2_grad",
        "backend": "jax",
        'dtype': 'float32',
        'raw_data': {
            'type': 'empad',
            'path': '~/Downloads/mos2/1/mos2/mos2_0.00_dstep1.0_x64_y64_4DSTEM.raw',
            'diff_step': 1.0,
            'kv': 120.0
        },
        'post_load': [
            {'type': 'poisson', 'scale': 1.0e6},
        ],
        'init_probe': {'type': 'focused', 'conv_angle': 25.0, 'defocus': 300.0},
        'init_object': 'random',
        'init_scan': {'type': 'raster', 'shape': (64, 64), 'step_size': 0.6},
        'post_init': [],
        'engines': [],
    })
    xp = get_backend_module(plan.backend)

    (patterns, state) = initialize_reconstruction(plan, xp, Observer())

    if sim_size != 128:
        # pad reconstruction
        new_sampling = Sampling((sim_size, sim_size), extent=tuple(state.probe.sampling.extent))
        print(f"Resampling probe and patterns to shape {new_sampling.shape}...", file=sys.stderr, flush=True)
        state.probe.data = state.probe.sampling.resample(state.probe.data, new_sampling)
        patterns.patterns = state.probe.sampling.resample_recip(patterns.patterns, new_sampling)
        patterns.pattern_mask = state.probe.sampling.resample_recip(patterns.pattern_mask, new_sampling)
        state.probe.sampling = new_sampling

    return (plan, patterns, state.to_numpy())


def benchmark_lsqml(grouping: int, sim_size: int, backend: BackendName) -> t.List[float]:
    (plan, patterns, init_state) = initialize(sim_size)
    xp = get_backend_module(backend)

    engine = pane.convert({
        'type': 'conventional',
        'probe_modes': 4,
        'niter': 12,
        'grouping': grouping,
        'noise_model': {'type': 'amplitude', 'eps': 1.0e-4},
        'solver': {
            'type': 'lsqml',
            'gamma': 1.0e-4,
        },
        'iter_constraints': [],
        'group_constraints': [
            {'type': 'clamp_object_amplitude', 'amplitude': 1.1},
        ],
        'update_probe': True,
        'update_object': True,
        'update_positions': False,
    }, EngineHook)

    observer = BenchmarkObserver()

    (patterns, state) = prepare_for_engine(patterns, init_state, xp, t.cast(EnginePlan, engine.props))

    state = engine({
        'data': patterns,
        'state': state,
        'dtype': patterns.patterns.dtype,
        'xp': xp,
        'recons_name': plan.name,
        'seed': None,
        'engine_i': 0,
        'observer': observer 
    })

    iter_times = observer.iter_times
    print(f"Mean time: {sum(iter_times) / len(iter_times):.3f} s", file=sys.stderr)
    return iter_times


def benchmark_grad(grouping: int, sim_size: int) -> t.List[float]:
    (plan, patterns, init_state) = initialize(sim_size)
    xp = get_backend_module('jax')

    engine = pane.convert({
        'type': 'gradient',
        'probe_modes': 4,
        'niter': 12,
        'grouping': grouping,
        'noise_model': {'type': 'amplitude', 'eps': 1.0e-4},
        'solvers': {
            'object': {
                'type': 'sgd',
                'learning_rate': 1.0,
                'momentum': 0.99,
            },
            'probe': {
                'type': 'sgd',
                'learning_rate': 1.0e-3,
                'momentum': 0.99,
            },
        },
        'regularizers': [
            {'type': 'obj_l1', 'cost': 15.0},
        ],
        'iter_constraints': [ ],
        'group_constraints': [
            {'type': 'clamp_object_amplitude', 'amplitude': 1.1},
        ],
        'update_probe': True,
        'update_object': True,
        'update_positions': False,
    }, EngineHook)

    observer = BenchmarkObserver()

    (patterns, state) = prepare_for_engine(patterns, init_state, xp, t.cast(EnginePlan, engine.props))

    state = engine({
        'data': patterns,
        'state': state,
        'dtype': patterns.patterns.dtype,
        'xp': xp,
        'recons_name': plan.name,
        'seed': None,
        'engine_i': 0,
        'observer': observer 
    })

    iter_times = observer.iter_times
    print(f"Mean time: {sum(iter_times) / len(iter_times):.3f} s", file=sys.stderr)
    return iter_times


if __name__ == '__main__':
    import jax

    device_name = jax.devices()[0].device_kind
    print(f"device: {device_name}", file=sys.stderr)

    for sim_size, backend, grouping in itertools.product((128, 192), ('cupy', 'jax'), (16, 32, 64, 128)):
    #for sim_size, backend, grouping in itertools.product((128,), ('cupy', 'jax'), (128,)):
        try:
            iter_times = benchmark_lsqml(grouping, sim_size, backend)
        except Exception as e:
            print(f"Failed to run, error:\n{e}", file=sys.stderr)
        else:
            json.dump({
                'engine': 'lsqml',
                'backend': backend,
                'sim_size': sim_size,
                'n_positions': 4096,
                'n_slices': 1,
                'grouping': grouping,
                'device': device_name,
                'code': 'v3',
                'iter_times': iter_times,
            }, sys.stdout)
            sys.stdout.write("\n")
            sys.stdout.flush()

    for sim_size, grouping in itertools.product((128, 192), (16, 32, 64, 128)):
    #for sim_size, grouping in itertools.product((128,), (128,)):
        try:
            iter_times = benchmark_grad(grouping, sim_size)
        except Exception as e:
            print(f"Failed to run, error:\n{e}", file=sys.stderr)
        else:
            
            json.dump({
                'engine': 'grad',
                'backend': 'jax',
                'sim_size': sim_size,
                'n_positions': 4096,
                'n_slices': 1,
                'grouping': grouping,
                'device': device_name,
                'code': 'v3',
                'iter_times': iter_times,
            }, sys.stdout)
            sys.stdout.write("\n")
            sys.stdout.flush()
