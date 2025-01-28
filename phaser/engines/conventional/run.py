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