import typing as t

import numpy
from numpy.typing import NDArray
from matplotlib import pyplot
from matplotlib.colors import Colormap, Normalize
from matplotlib.transforms import Affine2DBase

from .num import get_array_module, abs2, to_numpy
from .filter import remove_linear_ramp
from .object import ObjectSampling


if t.TYPE_CHECKING:
    from ..state import ObjectState
    from matplotlib.axes import Axes
    from matplotlib.figure import Figure
    from matplotlib.image import AxesImage


ColormapLike: t.TypeAlias = t.Union[str, Colormap]
NormLike: t.TypeAlias = Normalize


@t.overload
def plot_object_phase(
    data: 'ObjectState', sampling: None = None, *,
    ax: t.Optional['Axes'] = None,
    cmap: t.Optional[ColormapLike] = None, norm: t.Optional[NormLike] = None,
    vmin: t.Optional[float] = None, vmax: t.Optional[float] = None,
    zoom_roi: bool = True,
    **imshow_kwargs: t.Any,
) -> 'AxesImage':
    ...

@t.overload
def plot_object_phase(
    data: NDArray[numpy.complexfloating], sampling: ObjectSampling, *,
    ax: t.Optional['Axes'] = None,
    cmap: t.Optional[ColormapLike] = None, norm: t.Optional[NormLike] = None,
    vmin: t.Optional[float] = None, vmax: t.Optional[float] = None,
    zoom_roi: bool = True,
    **imshow_kwargs: t.Any,
) -> 'AxesImage':
    ...

def plot_object_phase(
    data: t.Union['ObjectState', NDArray[numpy.complexfloating]],
    sampling: t.Optional[ObjectSampling] = None, *,
    ax: t.Optional['Axes'] = None,
    cmap: t.Optional[ColormapLike] = None, norm: t.Optional[NormLike] = None,
    vmin: t.Optional[float] = None, vmax: t.Optional[float] = None,
    zoom_roi: bool = True,
    unwrap: bool = True,
    remove_ramp: bool = True,
    **imshow_kwargs: t.Any,
) -> 'AxesImage':
    if hasattr(data, 'sampling'):
        sampling = t.cast(ObjectSampling, getattr(data, 'sampling'))
        data = t.cast(NDArray[numpy.complexfloating], getattr(data, 'data'))
    elif sampling is None:
        raise ValueError("'sampling' must be specified, or an 'ObjectState' passed")
    else:
        data = t.cast(NDArray[numpy.complexfloating], data)

    xp = get_array_module(data)
    mask = sampling.get_region_mask()
    phase = xp.sum(xp.angle(data), axis=0)

    if unwrap:
        phase = xp.unwrap(xp.unwrap(phase, axis=1), axis=0)
    if remove_ramp:
        phase = remove_linear_ramp(phase, mask)

    return _plot_object_data(
        phase, mask, sampling,
        ax=ax, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax,
        zoom_roi=zoom_roi, **imshow_kwargs
    )


@t.overload
def plot_object_mag(
    data: 'ObjectState', sampling: None = None, *,
    ax: t.Optional['Axes'] = None,
    cmap: t.Optional[ColormapLike] = None, norm: t.Optional[NormLike] = None,
    vmin: t.Optional[float] = None, vmax: t.Optional[float] = None,
    zoom_roi: bool = True,
    **imshow_kwargs: t.Any,
) -> 'AxesImage':
    ...

@t.overload
def plot_object_mag(
    data: NDArray[numpy.complexfloating], sampling: ObjectSampling, *,
    ax: t.Optional['Axes'] = None,
    cmap: t.Optional[ColormapLike] = None, norm: t.Optional[NormLike] = None,
    vmin: t.Optional[float] = None, vmax: t.Optional[float] = None,
    zoom_roi: bool = True,
    **imshow_kwargs: t.Any,
) -> 'AxesImage':
    ...

def plot_object_mag(
    data: t.Union['ObjectState', NDArray[numpy.complexfloating]],
    sampling: t.Optional[ObjectSampling] = None, *,
    ax: t.Optional['Axes'] = None,
    cmap: t.Optional[ColormapLike] = None, norm: t.Optional[NormLike] = None,
    vmin: t.Optional[float] = None, vmax: t.Optional[float] = None,
    zoom_roi: bool = True,
    **imshow_kwargs: t.Any,
) -> 'AxesImage':
    if hasattr(data, 'sampling'):
        sampling = t.cast(ObjectSampling, getattr(data, 'sampling'))
        data = t.cast(NDArray[numpy.complexfloating], getattr(data, 'data'))
    elif sampling is None:
        raise ValueError("'sampling' must be specified, or an 'ObjectState' passed")
    else:
        data = t.cast(NDArray[numpy.complexfloating], data)

    xp = get_array_module(data)
    mag = abs2(xp.prod(data, axis=0))

    return _plot_object_data(
        mag, sampling.get_region_mask(), sampling,
        ax=ax, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax,
        zoom_roi=zoom_roi, **imshow_kwargs
    )


def _plot_object_data(
    data: NDArray[numpy.floating], mask: NDArray[numpy.bool_], sampling: ObjectSampling, *,
    ax: t.Optional['Axes'] = None,
    cmap: t.Optional[ColormapLike] = None, norm: t.Optional[NormLike] = None,
    vmin: t.Optional[float] = None, vmax: t.Optional[float] = None,
    zoom_roi: bool = True,
    **imshow_kwargs: t.Any,
) -> 'AxesImage':
    xp = get_array_module(data)

    if ax is None:
        fig, ax = pyplot.subplots()
        ax = t.cast('Axes', ax)
    else:
        fig = ax.get_figure()

    ax.set_xlabel(r"X [$\mathrm{\AA}$]")
    ax.set_ylabel(r"Y [$\mathrm{\AA}$]")

    if norm is None:
        vmin = vmin if vmin is not None else float(xp.nanmin(data[mask]))
        vmax = vmax if vmax is not None else float(xp.nanmax(data[mask]))
        norm = Normalize(vmin, vmax)

    cmap = t.cast('Colormap', pyplot.get_cmap(cmap))  # type: ignore

    img = ax.imshow(to_numpy(data), cmap=cmap, norm=norm, extent=sampling.mpl_extent(), **imshow_kwargs)

    if zoom_roi:
        min = sampling.region_min if sampling.region_min is not None else sampling.min
        max = sampling.region_max if sampling.region_max is not None else sampling.max
        ax.set_xlim(min[1], max[1])
        ax.set_ylim(max[0], min[0])

    return img


class _ScalebarTransform(Affine2DBase):
    def __init__(self, ax, origin):
        super().__init__("ScalebarTransform")
        self.origin = origin
        self._mtx = None
        self.transAxes = ax.transAxes
        self.transLimits = ax.transLimits
        self.set_children(self.transAxes, self.transLimits)

    def _calc_matrix(self):
        sx, sy = numpy.diag(self.transLimits.get_matrix())[:2]
        m = numpy.diag([sx, 1., 1.]) @ self.transAxes.get_matrix()
        m[0, 2] = (m[0, 2] + m[0, 0] * self.origin[0]) / sx
        m[1, 2] += m[1, 1] * self.origin[1]
        return m

    def get_matrix(self):
        if self._invalid:  # type: ignore
            self._mtx = self._calc_matrix()
        return self._mtx

def add_scalebar(ax: 'Axes', size: float, height: float = 0.05):
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((0.0, 0.0), size, height, fc='white', ec='black', linewidth=2.0, transform=_ScalebarTransform(ax, (0.04, 0.06))))

def add_scalebar_top(ax: 'Axes', size: float, height: float = 0.05):
    from matplotlib.patches import Rectangle
    ax.add_patch(Rectangle((0.0, 0.0), size, height, fc='white', ec='black', linewidth=2.0, transform=_ScalebarTransform(ax, (0.04, 1.0 - 0.06 - height))))


def plot_metrics(metrics: t.Dict[str, float]) -> 'Figure':
    fig, (ax1, ax2) = pyplot.subplots(ncols=2)
    fig.set_size_inches((8, 4))

    _plot_linear_metrics(ax1, metrics)
    _plot_probe_overlap(ax2, metrics)

    return fig


def _plot_linear_metrics(ax: 'Axes', metrics: t.Dict[str, float]):
    # mrad
    diff_scale = metrics['wavelength']*1e3/(2. * metrics['probe_radius'])
    step_scale = 2.*metrics['probe_radius']

    x = metrics['scan_step']
    y = metrics['diff_step']

    ax.set_xlabel("Scan step [$\\mathrm{\\AA/px}$]")
    ax.set_xscale('log')
    ax.set_ylabel("Diff. pixel size [mrad/px]")
    ax.set_yscale('log')

    ylim = (
        min(y/1.5, 0.1*diff_scale),
        max(y*1.5, 10.*diff_scale),
    )
    xlim = (
        min(x/1.5, 0.1*step_scale),
        max(x*1.5, 10.*step_scale),
    )
    yy = numpy.geomspace(*ylim, 201, endpoint=True)
    xx = numpy.geomspace(*xlim, 201, endpoint=True)
    (yy, xx) = numpy.meshgrid(yy, xx, indexing='ij')

    fps = 1.0 / (yy / diff_scale * xx / step_scale)

    ax.axhline(diff_scale, linestyle='dashed', color='#4363d8') # blue
    ax.axvline(step_scale, linestyle='dashdot', color='#f58231') # orange
    ax.contour(xx, yy, fps, [1.0], linestyles=['solid'], colors=['#e6194b']) # red

    ax.scatter([x], [y], marker='x', color='black', s=80)  # type: ignore

    ax.set_ylim(*ylim)
    ax.set_xlim(*xlim)


def _plot_probe_overlap(ax: 'Axes', metrics: t.Dict[str, float]):
    from matplotlib.patches import Circle, Rectangle

    probe_r = metrics['probe_radius']
    scan_r = metrics['scan_step'] / 2.0
    box_r = metrics['scan_step'] * metrics['fund_samp'] / 2.

    ax_r = scan_r + max(probe_r, box_r) * 1.1

    ax.set_xlim(-ax_r, ax_r)
    ax.set_ylim(-ax_r, ax_r)

    ax.set_axis_off()

    theta = 20.0 * numpy.pi / 180.

    pos1 = numpy.array([numpy.cos(theta), numpy.sin(theta)]) * scan_r

    _ = [ax.add_patch(Circle(
        list(pos), probe_r, facecolor='green', alpha=0.6, edgecolor='black',
        linewidth=2.0,
    )) for pos in (pos1, -pos1)]

    ax.plot([pos1[0], -pos1[0]], [pos1[0], -pos1[1]], '.-k', linewidth=2.0)

    _ = [ax.add_patch(Rectangle(
        [pos[0] - box_r, pos[1] - box_r], 2*box_r, 2*box_r,
        edgecolor='black', linewidth=2.0, fill=False,
    )) for pos in (pos1, -pos1)]
