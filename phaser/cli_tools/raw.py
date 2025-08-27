import re
from pathlib import Path
import typing as t

import numpy
from numpy.typing import NDArray


def load_4d(path: t.Union[str, Path]) -> NDArray[numpy.float32]:
    """
    Load a raw EMPAD dataset into memory. Scan dimensions are inferred from the filename.
    The file is loaded so the dimensions are: (scan_y, scan_x, k_y, k_x), with y decreasing downwards.
    """
    path = Path(path)
    match = re.search(r"x(\d+)_y(\d+)", path.name)
    if match:
        n_x, n_y = map(int, (match[1], match[2]))
    else:
        raise ValueError(f"Unable to infer probe dimensions from name {path.name}")

    a = numpy.memmap(path, dtype=numpy.float32, mode='r')
    if not a.size % (130*128) == 0:
        raise ValueError(f"File not divisible by 130x128 (size={a.size}).")
    a.shape = (-1, 130, 128)
    #a = a[:, :128, :]

    if a.shape[0] != n_x * n_y:
        raise ValueError(f"Got {a.shape[0]} probes, expected {n_x}x{n_y} = {n_x * n_y}.")
    a.shape = (n_y, n_x, *a.shape[1:])
    #print(a.shape)
    #with open(path.parent / "scan_x128_y64.raw", 'wb') as f:
    #    a[:64, :, :, :].ravel().tofile(f)
    #a = numpy.swapaxes(a, 0, 1)
    a = a[..., 127::-1, :]  # flip reciprocal y space, crop junk rows

    return a


def save_4d(a: NDArray[numpy.float32], folder: t.Union[str, Path, None] = None, name: str = "scan_x{x}_y{y}.raw"):
    """Save a raw EMPAD dataset to `folder`."""
    assert len(a.shape) == 4
    assert a.shape[2:] == (128, 128)
    n_y, n_x = a.shape[:2]

    out_shape = list(a.shape)
    out_shape[2] = 130  # dead rows

    out = numpy.zeros(out_shape, dtype=numpy.float32)
    out[..., 127::-1, :] = a

    path = Path(folder or Path()) / name.format(x=n_x, y=n_y)
    with open(path, 'wb') as f:
        out.tofile(f)
