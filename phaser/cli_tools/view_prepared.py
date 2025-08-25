#!/usr/bin/env python3

import typing as t

import click
import h5py
from scipy.io import loadmat
import numpy
from matplotlib import pyplot
from matplotlib.backend_bases import KeyEvent


def load_prepared_data(path: str, positions_path: t.Optional[str] = None) -> t.Tuple[numpy.ndarray, numpy.ndarray]:
    """
    Shape: (Nprobes, Ny, Nx), (Nprobes, 2)
    """

    print(f"Loading '{path}'...")
    f = t.cast(h5py.Group, h5py.File(path))

    if 'measurement' in f:
        data = t.cast(h5py.Dataset, f['measurement/n0/data'])[()]
        data = numpy.fft.fftshift(data, axes=(-1, -2))
        positions = t.cast(h5py.Dataset, f['measurement/n0/positions'])[()]
        return (data, positions)
    elif 'dp' in f:
        data = t.cast(h5py.Dataset, f['dp'])[()]
        if positions_path is None:
            raise ValueError("Old-style prepared data needs `positions_path` as well.")
        positions = load_probe_positions(positions_path)
        return (data, positions)

    raise ValueError(f"Couldn't find prepared patterns in '{path}'")


def load_probe_positions(path: str) -> numpy.ndarray:
    print(f"Loading '{path}'...")
    try:
        f = loadmat(path)
        if 'outputs' in f:
            return f['outputs'][0, 0]['probe_positions_0']
        raise Exception("Can't find probe positions in .mat file.")
    except (NotImplementedError, ValueError):
        pass

    raise NotImplementedError("Probe positions from HDF5 isn't implemented.")



@click.command()
@click.argument('path', type=click.Path(exists=True, dir_okay=False))
@click.argument('positions_path', type=click.Path(exists=True, dir_okay=False), required=False)
@click.option('--shape', type=(int, int), required=False)
def view_prepared(path: str, positions_path: t.Optional[str] = None, shape: t.Optional[t.Tuple[int, int]] = None):
    """
    View prepared data from PtychoShelves .mat or .h5 files.
    
    PATH should be a prepared HDF5 or .mat file.
    For a .mat file, POSITIONS_PATH should be a Niter.mat file.
    """
    (data, positions) = load_prepared_data(path, None)

    if data.shape[0] != positions.shape[0]:
        raise ValueError("Mismatch in # of probes: {data.shape[0]} in data vs {positions.shape[0]} probe positions.")

    n_probes = data.shape[0]

    stride = int(numpy.sqrt(n_probes)) if shape is None else shape[0]

    pattern_fig, pattern_ax = pyplot.subplots(constrained_layout=True)
    pos_fig, pos_ax = pyplot.subplots(constrained_layout=True)

    i: int = -1

    vmax = float(numpy.nanquantile(data, 0.9999))

    data = numpy.swapaxes(data[..., ::-1], -1, -2)
    img = pattern_ax.imshow(data[i], cmap='inferno', vmin=0.1*vmax, vmax=vmax)

    pos_ax.scatter(positions[:, 0], positions[:, 1], s=2, c='blue')
    pos_ax.set_aspect('equal')
    pos = pos_ax.scatter([positions[i, 0]], [positions[i, 1]], s=4, c='red')

    def update():
        img.set_data(data[i])
        pos.set_offsets(positions[i, None])

        pos_fig.canvas.draw_idle()
        pattern_fig.canvas.draw_idle()

    def key_press(event: KeyEvent):
        nonlocal i
        if event.key == 'right':
            i -= stride
        elif event.key == 'left':
            i += stride
        elif event.key == 'up':
            i -= 1
        elif event.key == 'down':
            i += 1
        else:
            return
        i = i % n_probes
        update()

    pattern_fig.canvas.mpl_connect('key_press_event', key_press)
    pos_fig.canvas.mpl_connect('key_press_event', key_press)
    pyplot.show()


if __name__ == '__main__':
    view_prepared()
