#!/usr/bin/env python3

from pathlib import Path
import typing as t

import click
import numpy
from numpy.typing import NDArray
import h5py
from scipy.io import loadmat


def get_scanrot(probe_pos: NDArray[numpy.floating]):
    pos = probe_pos
    n = int(numpy.sqrt(pos.shape[-1]))
    if not n**2 == pos.shape[-1]:
        raise ValueError("Only works on square probe grid")
    pos = pos.reshape(2, n, n)
    #xdiff, ydiff = pos[:, -1, 0] - pos[:, 0, 0]
    (ydiffs, xdiffs) = (pos[:, 1:, 0].T - pos[:, 0, 0]).T / numpy.arange(1, n)
    angles = -numpy.arctan2(ydiffs, xdiffs)
    scan_rot = numpy.mean(angles[len(angles)//2:])
    return scan_rot


def plot_convergence(probe, conv):
    from matplotlib import pyplot
    from matplotlib.patches import Circle
    fig, ax = pyplot.subplots()
    probe = numpy.fft.fftshift(probe)
    ax.imshow(numpy.abs(probe))
    ax.add_artist(Circle((len(probe)//2, len(probe)//2), radius=conv, color='red', fill=False))
    pyplot.show()


def get_convergence(probe):
    """Return the bright field radius, in px, for the given probe."""
    probe = numpy.fft.fft2(probe)
    #from matplotlib import pyplot
    #pyplot.imshow(numpy.abs(probe))
    #pyplot.show()
    mask = numpy.abs(probe) > 1e-3
    conv = numpy.sqrt(numpy.sum(mask) / numpy.pi)
    print(f"pix_convergence_angle: {conv} (convergence angle in px)")
    #plot_convergence(probe, conv)
    return conv


def from_niter(f: h5py.File):
    print("Loading as output parameters")
    params = t.cast(h5py.Group, f['par/p'])
    pix_size = 1e3/t.cast(h5py.Dataset, params['z'])[()].flat[0]
    print(f"diffraction_pix_size: {pix_size} (mrad, size of each diffraction pixel)")

    probe = t.cast(h5py.Dataset, params['probe_initial'])[()]
    probe = probe['real'] + 1.j * probe['imag']
    conv = get_convergence(probe)
    print(f"mrad_convergence_angle: {pix_size * conv} (convergence angle in mrad)")

    #import pdb; pdb.set_trace()
    orig_probe_pos = t.cast(h5py.Dataset, params['positions_0'])[()]
    final_probe_pos = t.cast(h5py.Dataset, params['positions'])[()]
    #probe_pos = params['positions'][()]  # final probe positions
    print(f"scanrot_det_offset: {get_scanrot(orig_probe_pos) * 180./numpy.pi} (deg, scan rotation)")
    print(f"   final positions: {get_scanrot(final_probe_pos) * 180./numpy.pi} (deg, scan rotation)")
    plot_probe_positions(orig_probe_pos)
    plot_probe_positions(final_probe_pos)


def from_probe_positions(f: h5py.File):
    print("Loading as probe positions")
    probe_pos = t.cast(h5py.Dataset, f['probe_positions_0'])[()]
    print(f"scanrot_det_offset: {get_scanrot(probe_pos) * 180./numpy.pi} (deg, scan rotation)")
    plot_probe_positions(probe_pos)


def plot_probe_positions(probe_pos):
    from matplotlib import pyplot
    pyplot.scatter(probe_pos[0], probe_pos[1], c=numpy.arange(probe_pos.shape[1]))
    pyplot.show()


def plot_probe_diff(probes: h5py.Dataset, i: int, n: int):
    from matplotlib import pyplot
    from matplotlib.backend_bases import KeyEvent

    np = int(numpy.sqrt(probes.shape[0]))
    if not np**2 == probes.shape[0]:
        raise ValueError(f"Only works on square probe grid (n={probes.shape[0]})")

    x = np//2
    y = np//2

    #fig, axs = pyplot.subplots(n, n, sharex=True, sharey=True)
    fig, ax = pyplot.subplots()
    probe = probes[x + y*np][()]
    img = ax.imshow(probe)

    def update():
        nonlocal probe
        print(f"\rpos: ({x}, {y})    ", end='')
        probe = probes[x + y*np][()]
        img.set_data(probe)
        fig.canvas.draw_idle()

    def key_press(event: KeyEvent):
        nonlocal x, y
        #print(f"Key released: {event.key}")
        if event.key == 'left':
            if x > 0:
                x -= 1
        elif event.key == 'right':
            if x < np - 1:
                x += 1
        elif event.key == 'up':
            if y > 0:
                y -= 1
        elif event.key == 'down':
            if y < np - 1:
                y += 1
        else:
            return
        update()

    fig.canvas.mpl_connect('key_press_event', key_press)
    pyplot.show()
    print()

    #i = numpy.arange(n)
    #(x, y) = numpy.meshgrid(i, i)
    #idxs = (x + y*np).reshape(-1)
    #probes = probes[idxs][()]
    #for (i, ax) in enumerate(axs.T.flat):
    #    ax.imshow(probes[i])


@click.command()
@click.argument('path', type=click.Path(exists=True, dir_okay=False))
def extract_params(path: t.Union[str, Path]):
    """Extract/infer reconstruction parameters from PtychoShelves files."""
    path = Path(path)
    print(f"Loading params from '{path}'...")
    f = None
    if path.suffix == '.mat':
        try:
            f = loadmat(path)
        except NotImplementedError:
            pass
    if f is None:
        f = h5py.File(path)

    if 'par' in f:
        assert isinstance(f, h5py.Group)
        from_niter(f)
    elif 'probe_positions_0' in f:
        assert isinstance(f, h5py.Group)
        from_probe_positions(f)
    elif 'probe' in f:
        probe = f['probe']
        get_convergence(probe)
    elif 'dp' in f:
        print("Checking probe movement...")
        probes = f['dp']  # shape: (nprobes, nk, nk)
        assert isinstance(probes, h5py.Dataset)
        plot_probe_diff(probes, 0, 5)
        #raise ValueError("No useful info in 'data_dp.mat'")
    else:
        #import pdb; pdb.set_trace()
        raise ValueError(f"Unknown file structure. Keys: {list(f.keys())}")


if __name__ == '__main__':
    extract_params()
