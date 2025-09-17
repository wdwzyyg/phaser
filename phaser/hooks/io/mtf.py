
from pathlib import Path
import warnings
import logging
import typing as t

import numpy

from phaser.utils.num import Sampling
# from phaser.utils.physics import Electron
# from phaser.io.empad import load_4d, EmpadMetadata
from phaser.types import cast_length
from phaser.hooks.mtf import LoadDetectorMtfProps, DetectorMtf

import starfile as sf


def load_mtf(args: None, props: LoadDetectorMtfProps) -> DetectorMtf:
    logger = logging.getLogger(__name__)

    path = Path(props.path).expanduser()

    mtf = DetectorMtf()
    if path.suffix.lower() == '.star':
        df = sf.read(path)

        mtf['inverse_pixels'] = df['rlnResolutionInversePixel'].to_numpy()
        mtf['values'] = df['rlnMtfValue'].to_numpy()
    else:
        mtf = None

    print(mtf)

    return {
       'mtf': mtf
    }