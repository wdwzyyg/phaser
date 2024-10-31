import logging

from phaser.utils.optics import make_focused_probe
from ..state import ProbeState
from . import ProbeHookArgs, FocusedProbeProps


def focused_probe(args: ProbeHookArgs, props: FocusedProbeProps) -> ProbeState:
    logger = logging.getLogger(__name__)

    logger.info(f"Making probe, conv_angle {props.conv_angle} mrad, defocus {props.defocus} A")

    sampling = args['sampling']
    ky, kx = sampling.recip_grid(dtype=args['dtype'], xp=args['xp'])
    probe = make_focused_probe(
        ky, kx, args['wavelength'],
        props.conv_angle, defocus=props.defocus
    )
    return ProbeState(sampling, probe)