#!/usr/bin/env python3

from pathlib import Path
import typing as t

import click
import h5py
import numpy
from scipy.io import loadmat
from scipy.linalg import lstsq
from matplotlib import pyplot
from matplotlib.widgets import Slider


def remove_phase_ramp(data: numpy.ndarray) -> numpy.ndarray:
    output = numpy.empty_like(data)

    (yy, xx) = (arr.flatten() for arr in numpy.indices(data.shape[1:], dtype=float))
    pts = numpy.stack((numpy.ones_like(xx), xx, yy), axis=-1)

    for i in range(data.shape[0]):
        layer = data[i]
        p, residues, rank, singular = lstsq(pts, layer.flatten())
        output[i] = layer - (p @ pts.T).reshape(layer.shape)

    return output


@click.command()
@click.argument('path', type=click.Path(exists=True, dir_okay=False))
@click.option('--phase/--no-phase', 'show_phase', default=True, help="Plot error progression.")
@click.option('--error/--no-error', 'show_error', default=False, help="Plot error progression.")
def view_output(path: t.Union[str, Path], show_phase: bool = True, show_error: bool = False):
    """
    View PtychoShelves output from a Niter.mat file.
    """
    print(f"Loading '{path}'...")
    try:
        f = loadmat(path)
        params: numpy.ndarray = t.cast(numpy.ndarray, f['p'])[0, 0]
        roi_y, roi_x = params['object_ROI'].flat
        roi_xx, roi_yy = numpy.meshgrid(roi_x.flatten(), roi_y.flatten(), indexing='ij')

        error: numpy.ndarray = t.cast(numpy.ndarray, f['outputs']['fourier_error_out'])[0, 0]

        obj: numpy.ndarray = numpy.atleast_3d(f['object'])[roi_yy, roi_xx]
        obj = numpy.moveaxis(obj, -1, 0)
        print(f"Loaded from loadmat.")
    except NotImplementedError:
        f = t.cast(h5py.Group, h5py.File(path))
        error = numpy.asarray(f['outputs/fourier_error_out'])

        obj_refs = t.cast(h5py.Dataset, f['outputs/object_roi'])[:, 0]
        obj: numpy.ndarray = numpy.stack([t.cast(numpy.ndarray, t.cast(h5py.Dataset, f[h5py.h5r.get_name(ref, f.id)])[()]) for ref in obj_refs], axis=0)
        obj = obj['real'] + 1.j * obj['imag']
        print(f"Loaded from HDF5.")

    #obj_mag = t.cast(numpy.ndarray, numpy.abs(obj))

    if show_phase:
        obj_phase = t.cast(numpy.ndarray, numpy.angle(obj))
        obj_phase = numpy.unwrap(numpy.unwrap(obj_phase, axis=-1), axis=-2)
        obj_phase = remove_phase_ramp(obj_phase)

        fig, (img_ax, slider_ax) = pyplot.subplots(nrows=2, gridspec_kw={'height_ratios': [12, 1]})
        #vmin, vmax = numpy.min(phase[:, 200:-200, 200:-200]), numpy.max(phase[:, 200:-200, 200:-200])
        vmin, vmax = numpy.nanquantile(obj_phase, 0.02), numpy.nanquantile(obj_phase, 0.98)
        img = img_ax.imshow(obj_phase[0], vmin=float(vmin), vmax=float(vmax), cmap='inferno')
        slider = Slider(ax=slider_ax, label='Frame', valmin=0, valmax=len(obj)-1, valinit=0, valstep=1)

        fig.colorbar(img, ax=img_ax)

        def update(val):
            i = int(numpy.floor(val))
            img.set(data=obj_phase[i])

        slider.on_changed(update)

    if show_error:
        error = error.flatten()
        iterations = numpy.arange(error.size)[~numpy.isnan(error)]
        error = error[~numpy.isnan(error)]

        fig, ax = pyplot.subplots()
        ax.plot(iterations, error, '.-', color='black')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Fourier Error')

    pyplot.show()


if __name__ == '__main__':
    view_output()
