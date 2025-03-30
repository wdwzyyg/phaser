
import numpy
from numpy.typing import NDArray

from phaser.utils.num import cast_array_module
from phaser.utils.scan import make_raster_scan
from . import ScanHookArgs, RasterScanProps


def raster_scan(args: ScanHookArgs, props: RasterScanProps) -> NDArray[numpy.floating]:
    xp = cast_array_module(args['xp'])
    scan = make_raster_scan(
        props.shape, props.step_size, props.rotation,
        dtype=args['dtype'], xp=xp,
    )

    if props.affine is not None:
        affine = xp.array(props.affine, dtype=scan.dtype)[::-1, ::-1]
        scan = scan @ affine

    return scan