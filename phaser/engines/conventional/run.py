import logging
import typing as t

from phaser.hooks.solver import ConstraintRegularizer
from phaser.utils.num import cast_array_module
from phaser.hooks import EngineArgs
from phaser.plan import ConventionalEnginePlan
from phaser.execute import Observer, process_flag
from phaser.engines.common.simulation import SimulationState
from phaser.state import ReconsState


def run_engine(args: EngineArgs, props: ConventionalEnginePlan) -> ReconsState:
    logger = logging.getLogger(__name__)

    xp = cast_array_module(args['xp'])
    dtype = args['dtype']
    observer: Observer = args.get('observer', [])

    logger.info(f"Starting engine #{args['engine_i'] + 1}...")

    noise_model = props.noise_model(None)
    regularizers = t.cast(t.Tuple[ConstraintRegularizer, ...], tuple(
        reg(None) for reg in props.regularizers
    ))

    update_probe = process_flag(props.update_probe)
    update_object = process_flag(props.update_object)

    sim = SimulationState(
        state=args['state'], noise_model=noise_model, regularizers=regularizers,
        patterns=args['data'].patterns, pattern_mask=args['data'].pattern_mask,
        xp=xp, dtype=dtype
    )

    solver = props.solver(props)
    sim = solver.solve(
        sim, observer=observer, engine_i=args['engine_i'],
        update_probe=update_probe, update_object=update_object
    )

    return sim.state