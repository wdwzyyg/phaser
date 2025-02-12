import typing as t

import numpy
from numpy.typing import NDArray
from matplotlib import pyplot
from matplotlib.colors import Colormap, Normalize

from .num import get_array_module, abs2, to_numpy
from .object import ObjectSampling


if t.TYPE_CHECKING:
    from ..state import ObjectState
    from matplotlib.axes import Axes
    from matplotlib.image import AxesImage


ColormapLike: t.TypeAlias = t.Union[str, Colormap]
NormLike: t.TypeAlias = t.Union[str, Normalize]


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
    # TODO subtract phase ramp
    phase = xp.sum(xp.angle(data), axis=0)
    phase_crop = phase[sampling.get_region_crop()]

    return _plot_object_data(
        phase, phase_crop, sampling,
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
    mag_crop = mag[sampling.get_region_crop()]

    return _plot_object_data(
        mag, mag_crop, sampling,
        ax=ax, cmap=cmap, norm=norm, vmin=vmin, vmax=vmax,
        zoom_roi=zoom_roi, **imshow_kwargs
    )


def _plot_object_data(
    data: NDArray[numpy.floating], data_crop: NDArray[numpy.floating], sampling: ObjectSampling, *,
    ax: t.Optional['Axes'] = None,
    cmap: t.Optional[ColormapLike] = None, norm: t.Optional[NormLike] = None,
    vmin: t.Optional[float] = None, vmax: t.Optional[float] = None,
    zoom_roi: bool = True,
    **imshow_kwargs: t.Any,
) -> 'AxesImage':
    xp = get_array_module(data)

    if ax is None:
        fig, ax = pyplot.subplots()
    else:
        fig = ax.get_figure()

    ax.set_xlabel(r"X [$\mathrm{\AA}$]")
    ax.set_ylabel(r"Y [$\mathrm{\AA}$]")

    if norm is None:
        vmin = vmin if vmin is not None else float(xp.nanmin(data_crop))
        vmax = vmax if vmax is not None else float(xp.nanmax(data_crop))
        norm = Normalize(vmin, vmax)

    cmap = pyplot.get_cmap(cmap)

    img = ax.imshow(to_numpy(data), cmap=cmap, norm=norm, extent=sampling.mpl_extent(), **imshow_kwargs)

    if zoom_roi:
        min = sampling.region_min if sampling.region_min is not None else sampling.min
        max = sampling.region_max if sampling.region_max is not None else sampling.max
        ax.set_xlim(min[1], max[1])
        ax.set_ylim(max[0], min[0])

    return img