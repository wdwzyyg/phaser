#!/usr/bin/env python3

from pathlib import Path
import typing as t

import numpy
import click
from matplotlib import pyplot
from matplotlib.backend_bases import KeyEvent, MouseEvent, MouseButton
import matplotlib.path as mpath
from matplotlib.transforms import Affine2D
from matplotlib.patches import PathPatch

from .raw import load_4d


def save_hist(path: t.Union[str, Path]):
    a = load_4d(Path(path))

    fig = pyplot.figure(figsize=(16, 8), constrained_layout=True)

    (ax1, ax2) = fig.subplots(nrows=2, sharex=True)
    #fig.suptitle('loghist')
    flat = a.flatten()
    flat = flat[flat < 1e5]

    vmin, vmax = numpy.nanmin(flat), numpy.nanmax(flat)

    nbins = min(1024, numpy.floor(vmax - vmin).astype(int))

    vals, bins = numpy.histogram(flat, bins=nbins, density=True)
    widths = numpy.diff(bins)
    #ax.bar(bins, numpy.log(vals))
    ax1.set_yscale('log')
    ax1.bar(bins[:-1], vals, width=widths, color='black', align='edge')
    ax2.bar(bins[:-1], vals, width=widths, color='black', align='edge')
    ax1.margins(0.)
    ax2.margins(0.)
    ax2.set_xlabel("ADU")
    ax1.set_ylabel("freq")
    ax2.set_ylabel("freq")

    pyplot.show()

    fig.savefig(str(Path(path).with_name('hist.png')), dpi=500)


@click.command()
@click.argument('path', type=click.Path(exists=True, dir_okay=False))
@click.option('--hist/--no-hist', default=False, help="Show histogram")
@click.option('--pacbed/--no-pacbed', default=False, help="Show pacbed")
@click.option('--com/--no-com', default=False, help="Show COM X & Y")
def view_raw(path: t.Union[str, Path], hist: bool = False, pacbed: bool = False, com: bool = False):
    """Visualize raw datasets (off-brand 4D STEM Explorer)."""

    a = load_4d(Path(path))
    (yy, xx) = numpy.indices(a.shape[2:])

    filtered = a.copy()
    filtered[(filtered > 1e6) | (filtered < 0.) | numpy.isnan(filtered)] = 0.

    if pacbed:
        pacbed = numpy.log(filtered.sum(axis=(0, 1), dtype=numpy.float64))
        pacbed_fig, ax = pyplot.subplots()
        ax.imshow(pacbed)

    if com:
        com_x = numpy.sum(xx * filtered, axis=(-1, -2), dtype=numpy.float64) / numpy.sum(filtered, axis=(-1, -2), dtype=numpy.float64)
        fig, ax = pyplot.subplots()
        fig.suptitle('com_x')
        ax.imshow(com_x)
        com_y = numpy.sum(yy * filtered, axis=(-1, -2), dtype=numpy.float64) / numpy.sum(filtered, axis=(-1, -2), dtype=numpy.float64)
        fig, ax = pyplot.subplots()
        fig.suptitle('com_y')
        ax.imshow(com_y)

    if hist:
        fig, (ax1, ax2) = pyplot.subplots(nrows=2, sharex=True, constrained_layout=True)
        #fig.suptitle('loghist')
        flat = a.flatten()
        flat = flat[flat < 1e5]
        vals, bins = numpy.histogram(flat, bins=1024, density=True)
        widths = numpy.diff(bins)
        #ax.bar(bins, numpy.log(vals))
        ax1.set_yscale('log')
        ax1.bar(bins[:-1], vals, width=widths, color='black', align='edge')
        ax2.bar(bins[:-1], vals, width=widths, color='black', align='edge')
        ax1.margins(0.)
        ax2.margins(0.)
        ax2.set_xlabel("ADU")
        ax1.set_ylabel("freq")
        ax2.set_ylabel("freq")
        #ax2.hist(a.flatten(), bins=1024, density=True, color='black', range=(0, 5000))  #range=(0, 16383)

    #fig, axs = pyplot.subplots(n, n, sharex=True, sharey=True)
    recip_fig, recip_ax = pyplot.subplots()
    y = a.shape[0] // 2
    x = a.shape[1] // 2
    probe = a[y, x]
    img = recip_ax.imshow(probe)

    y_c = 64.
    x_c = 64.
    r = 28.
    mask = (xx - x_c)**2 + (yy - y_c)**2 > r**2
    real_space = numpy.sum(a[:, :, mask], axis=-1, dtype=numpy.float64)

    real_fig, real_ax = pyplot.subplots()
    real_ax.imshow(real_space, vmin=float(numpy.quantile(real_space, 0.01)), vmax=float(numpy.quantile(real_space, 0.99)))

    crosshair = mpath.Path(numpy.array([
        [-1.5, -1.5], [1.5, -1.5], [1.5, 1.5], [-1.5, 1.5],
        [-0.5, -0.5], [-0.5, 0.5], [0.5, 0.5], [0.5, -0.5]
    ]), list(map(int, [mpath.Path.MOVETO, 2, 2, 2, mpath.Path.MOVETO, 2, 2, 2])))
    marker = PathPatch(crosshair, fc='red', fill=True, linestyle='None', transform=Affine2D().translate(x, y) + real_ax.transData)
    real_ax.add_patch(marker)
    #marker = Rectangle((x-1., y-1.), 2., 2., ec='red', fill=False, lw=1.5)

    def update():
        nonlocal probe
        print(f"\rpos: ({x}, {y})   ", end='')
        probe = a[y, x]
        img.set_data(probe)
        marker.set_transform(Affine2D().translate(x, y) + real_ax.transData)
        recip_fig.canvas.draw_idle()
        real_fig.canvas.draw_idle()

    def key_press(event: KeyEvent):
        nonlocal x, y
        #print(f"Key released: {event.key}")
        if event.key == 'left':
            if x > 0:
                x -= 1
        elif event.key == 'right':
            if x < a.shape[1] - 1:
                x += 1
        elif event.key == 'up':
            if y > 0:
                y -= 1
        elif event.key == 'down':
            if y < a.shape[0] - 1:
                y += 1
        else:
            print("ignored")
        update()

    def mouse_click(event: MouseEvent):
        nonlocal x, y
        if event.button is MouseButton.LEFT \
            and event.x is not None and event.y is not None:

            (click_x, click_y) = real_ax.transData.inverted().transform(tuple(map(int, (event.x, event.y))))
            (click_x, click_y) = map(int, map(round, (click_x, click_y)))
            if not 0 <= click_x < a.shape[1] or not 0 <= click_y < a.shape[0]:
                return
            x, y = click_x, click_y
            update()

    recip_fig.canvas.mpl_connect('key_press_event', key_press)
    real_fig.canvas.mpl_connect('key_press_event', key_press)
    real_fig.canvas.mpl_connect('button_press_event', mouse_click)

    pyplot.show()
    print("")


if __name__ == '__main__':
    view_raw()