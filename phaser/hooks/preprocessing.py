import logging
import typing as t

from phaser.utils.num import cast_array_module, to_numpy
from phaser.utils.misc import create_rng
from phaser.state import Patterns, ReconsState
from . import PreprocessingArgs, PoissonProps, ScaleProps, DropNanProps

logger = logging.getLogger(__name__)


def scale_patterns(args: PreprocessingArgs, props: ScaleProps) -> t.Tuple[Patterns, ReconsState]:
    args['data'].patterns *= props.scale
    return (args['data'], args['state'])


def add_poisson_noise(args: PreprocessingArgs, props: PoissonProps) -> t.Tuple[Patterns, ReconsState]:
    xp = cast_array_module(args['xp'])
    dtype = args['dtype']


    if props.scale is not None:
        logger.info(f"Adding poisson noise to raw patterns, after scaling by {props.scale:.2e}")
        args['data'].patterns *= props.scale
    else:
        logger.info(f"Adding poisson noise to raw patterns")

    rng = create_rng(args['seed'], 'poisson_noise')

    patterns = rng.poisson(to_numpy(args['data'].patterns)).astype(dtype)

    if props.gaussian is not None:
        patterns += rng.normal(scale=props.gaussian, size=patterns.shape)

    args['data'].patterns = xp.array(patterns)
    return (args['data'], args['state'])


def drop_nan_patterns(args: PreprocessingArgs, props: DropNanProps) -> t.Tuple[Patterns, ReconsState]:
    xp = cast_array_module(args['xp'])

    # flatten scan and patterns
    scan = args['state'].scan.reshape(-1, 2)
    patterns = args['data'].patterns.reshape(-1, *args['data'].patterns.shape[-2:])

    fraction_nan = xp.sum(xp.isnan(args['data'].patterns), axis=(-1, -2)) / xp.prod(patterns.shape[-2:])

    mask = fraction_nan > props.threshold

    if (n := int(xp.sum(mask))):
        logger.info(f"Dropping {n}/{scan.shape[0]} patterns which are at least {props.threshold:.1%} NaN values")

        scan = scan[mask]
        patterns = patterns[mask]

    args['state'].scan = scan
    args['data'].patterns = patterns

    return (args['data'], args['state'])
