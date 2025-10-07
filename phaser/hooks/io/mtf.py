
from pathlib import Path
import warnings
import logging
import typing as t

import numpy

from phaser.utils.num import Sampling
# from phaser.utils.physics import Electron
# from phaser.io.empad import load_4d, EmpadMetadata
from phaser.types import cast_length
from phaser.hooks.mtf import StarFileMtfProps, GaussianMTFProps, DetectorMtf, MTFHookArgs

import starfile as sf

import matplotlib.pyplot as plt


def load_starfile_mtf(args: MTFHookArgs, props: StarFileMtfProps) -> DetectorMtf:
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

def calc_gaussian_mtf(args: MTFHookArgs, props: GaussianMTFProps):

    shape = args['shape']
    ky = numpy.fft.fftfreq(shape[0])
    kx = numpy.fft.fftfreq(shape[1])
    ky, kx = numpy.meshgrid(ky, kx, indexing='ij')
    k2 = ky**2 + kx**2
    
    mtf = numpy.exp(-props.sigma**2 * k2/2)

    return mtf