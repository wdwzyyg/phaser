
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
from scipy.interpolate import RegularGridInterpolator
import starfile as sf


# import matplotlib.pyplot as plt


def calc_k2_space(shape:tuple ):

    ky = numpy.fft.fftfreq(shape[0])
    kx = numpy.fft.fftfreq(shape[1])
    ky, kx = numpy.meshgrid(ky, kx, indexing='ij')
    k2 = ky**2 + kx**2

    return k2
    

def load_starfile_mtf(args: MTFHookArgs, props: StarFileMtfProps) -> DetectorMtf:
    logger = logging.getLogger(__name__)

    path = Path(props.path).expanduser()

    shape = args['shape']
    N = props.bin

    k2 = calc_k2_space(shape=shape)
    k = numpy.sqrt(k2)

    if path.suffix.lower() == '.star':
        df = sf.read(path)

        kp = df['rlnResolutionInversePixel'].to_numpy()
        mp = df['rlnMtfValue'].to_numpy()

        
        ratio = numpy.sinc(N * kp) / numpy.maximum(numpy.sinc(kp), 1e-12)
        mp *= ratio

        mtf = numpy.interp(k/N, kp, mp)


    else:
        mtf = None

    return mtf

def calc_gaussian_mtf(args: MTFHookArgs, props: GaussianMTFProps):

    shape = args['shape']

    k2 = calc_k2_space(shape=shape)

    mtf = numpy.exp(-props.sigma**2 * k2/2)

    return mtf