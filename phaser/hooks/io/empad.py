
from pathlib import Path
import re
import typing as t

import numpy
from numpy.typing import NDArray

from phaser.utils.num import Sampling
from phaser.utils.physics import Electron
from .. import LoadEmpadProps, RawData


def load_empad(args: None, props: LoadEmpadProps) -> RawData:
    path = Path(props.path).expanduser()

    wavelength = Electron(props.kv * 1e3).wavelength

    a = wavelength / (props.diff_step * 1e-3)  # recip. pixel size -> 1 / real space extent
    sampling = Sampling((128, 128), extent=(a, a))

    # TODO handle metadata here
    patterns = numpy.fft.fftshift(load_4d(path), axes=(-1, -2))
    return {
        'patterns': patterns,
        'mask': numpy.ones_like(patterns, shape=patterns.shape[-2:]),
        'sampling': sampling,
        'wavelength': wavelength,
        'scan': None,
        'probe_options': None,
    }


def load_4d(path: t.Union[str, Path], scan_shape: t.Optional[t.Tuple[int, int]] = None,
            memmap: bool = False) -> NDArray[numpy.float32]:
    """
    Load a raw EMPAD dataset into memory.

    The file is loaded so the dimensions are: (scan_y, scan_x, k_y, k_x), with y decreasing downwards.

    Patterns are not fftshifted or normalized upon loading.

    # Parameters

     - `path`: Path to file to load
     - `scan_shape`: Scan shape of dataset. Will be inferred from the filename if not specified.
     - `memmap`: If specified, memmap the file as opposed to loading it eagerly.

    Returns a numpy array (or `numpy.memmap`)
    """
    path = Path(path)

    if scan_shape is None:
        match = re.search(r"x(\d+)_y(\d+)", path.name)
        if match:
            n_x, n_y = map(int, (match[1], match[2]))
        else:
            raise ValueError(f"Unable to infer probe dimensions from name {path.name}")
    else:
        n_y, n_x = scan_shape

    if memmap:
        a = numpy.memmap(path, dtype=numpy.float32, mode='r')
    else:
        a = numpy.fromfile(path, dtype=numpy.float32)

    if not a.size % (130*128) == 0:
        raise ValueError(f"File not divisible by 130x128 (size={a.size}).")
    a.shape = (-1, 130, 128)
    #a = a[:, :128, :]

    if a.shape[0] != n_x * n_y:
        raise ValueError(f"Got {a.shape[0]} probes, expected {n_x}x{n_y} = {n_x * n_y}.")
    a.shape = (n_y, n_x, *a.shape[1:])
    a = a[..., 127::-1, :]  # flip reciprocal y space, crop junk rows

    return a


@t.overload
def save_4d(arr: NDArray[numpy.float32], *, path: t.Union[str, Path], folder: None = None, name: None = None):
    ...

@t.overload
def save_4d(arr: NDArray[numpy.float32], *, path: None = None, folder: t.Union[str, Path], name: t.Optional[str] = None):
    ...

def save_4d(arr: NDArray[numpy.float32], *, path: t.Union[str, Path, None] = None,
            folder: t.Union[str, Path, None] = None, name: t.Optional[str] = None): #):
    """
    Save a raw EMPAD dataset.

    Either `path` or `folder` can be specified. If `folder` is specified,
    `name` will be used as a format string to determine the filename.
    `path` and `folder` cannot be specified simultaneously.

    Patterns are not fftshifted or normalized upon saving.

    Parameters:
     - `arr`: Array to save
     - `path`: Path to save dataset to.
     - `folder`: Folder to save dataset inside.
     - `name`: When `folder` is specified, format to use to determine filename. Defaults to `"scan_x{x}_y{y}.raw"`.
       Will be formatted using the scan shape `{'x': n_x, 'y': n_y}`.
    """

    try:
        assert len(arr.shape) == 4
        assert arr.shape[2:] == (128, 128)
    except AssertionError as e:
        raise ValueError("Invalid data format") from e

    if folder is not None:
        if path is not None:
            raise ValueError("Cannot specify both 'path' and 'folder'")

        n_y, n_x = arr.shape[:2]
        path = Path(folder) / (name or "scan_x{x}_y{y}.raw").format(x=n_x, y=n_y)
    elif path is not None:
        path = Path(path)
    else:
        raise ValueError("Must specify either 'path' or 'folder'")

    out_shape = list(arr.shape)
    out_shape[2] = 130  # dead rows

    out = numpy.zeros(out_shape, dtype=numpy.float32)
    out[..., 127::-1, :] = arr.astype(numpy.float32)

    with open(path, 'wb') as f:
        out.tofile(f)