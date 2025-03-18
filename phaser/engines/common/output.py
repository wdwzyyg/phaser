from functools import partial
from pathlib import Path
import typing as t

import numpy
from numpy.typing import NDArray
import tifffile

from phaser.utils.num import to_numpy, abs2, fft2, get_array_module
from phaser.utils.image import remove_linear_ramp, colorize_complex
from phaser.state import ReconsState, ProbeState, ObjectState
from phaser.plan import SaveOptions


def _scale_to_integral_type(
    arr: NDArray[numpy.floating],
    ty: t.Literal['8bit', '16bit', '32bit', '64bit'],
    mask: t.Optional[NDArray[numpy.bool_]] = None,
    min_range: t.Optional[float] = None,
) -> NDArray[numpy.unsignedinteger]:
    xp = get_array_module(arr)

    dtype = {
        '8bit': numpy.uint8,
        '16bit': numpy.uint16,
        '32bit': numpy.uint32,
        '64bit': numpy.uint64,
    }[ty]

    imax = numpy.iinfo(dtype).max

    arr_crop = arr[..., mask] if mask is not None else arr
    # TODO: cupy doesn't support nanquantile
    vmax = xp.nanquantile(arr_crop, 0.999)
    vmin = xp.nanquantile(arr_crop, 0.001)

    if min_range is not None and (delta := min_range - (vmax - vmin)) > 0:
        # expand max and min to cover min_range
        vmax += delta/2
        vmin -= delta/2

    return (xp.clip((imax + 1) / (vmax - vmin) * (arr - vmin), 0, imax)).astype(dtype)


def output_images(state: ReconsState, out_dir: Path, options: SaveOptions):
    for ty in options.images:
        if ty not in _SAVE_FUNCS:
            raise ValueError(f"Unknown image type '{ty}'")

        try:
            out_name = options.img_fmt.format(
                type=ty, iter=state.iter,
            )
            out_path = out_dir / out_name
        except KeyError as e:
            raise ValueError(f"Invalid format string in 'img_fmt' (unknown key {e})") from None
        except Exception as e:
            raise ValueError("Invalid format string in 'img_fmt'") from e

        _SAVE_FUNCS[ty](state, out_path, options)


def output_state(state: ReconsState, out_dir: Path, options: SaveOptions):
    try:
        out_name = options.hdf5_fmt.format(
            iter=state.iter
        )
    except KeyError as e:
        raise ValueError(f"Invalid format string in 'hdf5_fmt' (unknown key {e})") from None
    except Exception as e:
        raise ValueError("Invalid format string in 'hdf5_fmt'") from e

    state.write_hdf5(out_dir / out_name)


def _probe_write_opts(probe: ProbeState) -> t.Dict[str, t.Any]:
    unit = 'angstrom'
    nprobes = probe.data.shape[0]
    return {
        # 1/angstrom -> 1/centimeter
        'resolution': (1e8/probe.sampling.sampling[1], 1e8/probe.sampling.sampling[0]),
        'resolutionunit': "CENTIMETER",
        'metadata': {
            'OME': {
                'PhysicalSizeX': float(probe.sampling.sampling[1]),
                'PhysicalSizeXUnit': unit,
                'PhysicalSizeY': float(probe.sampling.sampling[0]),
                'PhysicalSizeYUnit': unit,
                'Plane': {
                    'PositionX': [0.0] * nprobes,
                    'PositionXUnit': [unit] * nprobes,
                    'PositionY': [0.0] * nprobes,
                    'PositionYUnit': [unit] * nprobes,
                }
            },
        }
    }


def _probe_recip_write_opts(probe: ProbeState) -> t.Dict[str, t.Any]:
    unit = '1/angstrom'
    nprobes = probe.data.shape[0]
    return {
        'metadata': {
            'OME': {
                'PhysicalSizeX': float(1/probe.sampling.extent[1]),
                'PhysicalSizeXUnit': unit,
                'PhysicalSizeY': float(1/probe.sampling.extent[0]),
                'PhysicalSizeYUnit': unit,
                'Plane': {
                    'PositionX': [0.0] * nprobes,
                    'PositionXUnit': [unit] * nprobes,
                    'PositionY': [0.0] * nprobes,
                    'PositionYUnit': [unit] * nprobes,
                }
            },
        }
    }


def _obj_write_opts(obj: ObjectState, stack: bool, crop: bool) -> t.Dict[str, t.Any]:
    unit = 'angstrom'
    d = {
        # 1/angstrom -> 1/centimeter
        'resolution': (1e8/obj.sampling.sampling[1], 1e8/obj.sampling.sampling[0]),
        'resolutionunit': "CENTIMETER",
        'metadata': {
            'OME': {
                'PhysicalSizeX': float(obj.sampling.sampling[1]),
                'PhysicalSizeXUnit': unit,
                'PhysicalSizeY': float(obj.sampling.sampling[0]),
                'PhysicalSizeYUnit': unit,
            },
        }
    }

    if stack:
        # TODO this isn't quite correct, make a better API in ObjectSapling
        obj_min = obj.sampling.region_min if crop and obj.sampling.region_min is not None else obj.sampling.min
        slices = obj.data.shape[0]
        zs = numpy.cumsum(obj.thicknesses) - obj.thicknesses[0] if len(obj.thicknesses) > 2 else [0.]
        d['metadata']['OME']['Plane'] = {
            'PositionX': [obj_min[1]] * slices,
            'PositionXUnit': [unit] * slices,
            'PositionY': [obj_min[0]] * slices,
            'PositionYUnit': [unit] * slices,
            'PositionZ': list(to_numpy(zs)),
            'PositionZUnit': [unit] * slices,
        }

    return d


def _save_probe(state: ReconsState, out_path: Path, options: SaveOptions):
    write_opts = _probe_write_opts(state.probe)
    probe = to_numpy(state.probe.data)

    if options.img_dtype == 'float':
        # save complex image
        with tifffile.TiffWriter(out_path, ome=True) as w:
            write_opts['metadata']['axes'] = 'CYX'
            w.write(probe, **write_opts)
        return

    img = _scale_to_integral_type(
        colorize_complex(probe), options.img_dtype
    )
    with tifffile.TiffWriter(out_path, ome=True) as w:
        write_opts['metadata']['axes'] = 'CYXS'
        w.write(img, photometric='rgb', **write_opts)


def _save_probe_mag(state: ReconsState, out_path: Path, options: SaveOptions):
    write_opts = _probe_write_opts(state.probe)
    probe_mag = abs2(state.probe.data)

    if options.img_dtype != 'float':
        probe_mag = _scale_to_integral_type(to_numpy(probe_mag), options.img_dtype, min_range=0.2)

    with tifffile.TiffWriter(out_path, ome=True) as w:
        write_opts['metadata']['axes'] = 'CYX'
        w.write(to_numpy(probe_mag), **write_opts)


def _save_probe_recip(state: ReconsState, out_path: Path, options: SaveOptions):
    write_opts = _probe_recip_write_opts(state.probe)
    xp = get_array_module(state.probe.data)
    probe = to_numpy(xp.fft.fftshift(fft2(state.probe.data), axes=(-1, -2)))

    if options.img_dtype == 'float':
        # save complex image
        with tifffile.TiffWriter(out_path, ome=True) as w:
            write_opts['metadata']['axes'] = 'CYX'
            w.write(probe, **write_opts)
        return

    img = _scale_to_integral_type(
        colorize_complex(probe), options.img_dtype
    )
    with tifffile.TiffWriter(out_path, ome=True) as w:
        write_opts['metadata']['axes'] = 'CYXS'
        w.write(img, photometric='rgb', **write_opts)


def _save_probe_recip_mag(state: ReconsState, out_path: Path, options: SaveOptions):
    write_opts = _probe_recip_write_opts(state.probe)
    xp = get_array_module(state.probe.data)
    probe_mag = to_numpy(abs2(xp.fft.fftshift(fft2(state.probe.data), axes=(-1, -2))))

    if options.img_dtype != 'float':
        probe_mag = _scale_to_integral_type(probe_mag, options.img_dtype, min_range=0.2)

    with tifffile.TiffWriter(out_path, ome=True) as w:
        write_opts['metadata']['axes'] = 'CYX'
        w.write(probe_mag, **write_opts)


def _save_object_phase(state: ReconsState, out_path: Path, options: SaveOptions, stack: bool = False):
    crop = options.crop_roi
    write_opts = _obj_write_opts(state.object, stack=stack, crop=crop)

    xp = get_array_module(state.object.data)
    obj_phase = xp.angle(state.object.data)

    if crop:
        obj_phase = obj_phase[..., *state.object.sampling.get_region_crop()]
        mask = xp.ones(obj_phase.shape[-2:], dtype=numpy.bool_)
    else:
        # include whole image, but only scale based on ROI
        mask = state.object.sampling.get_region_mask(xp)

    if options.unwrap_phase:
        obj_phase = xp.unwrap(xp.unwrap(obj_phase, axis=-1), axis=-2)

    if not stack:
        obj_phase = xp.sum(obj_phase, axis=0)

    obj_phase = to_numpy(remove_linear_ramp(obj_phase, mask))
    mask = to_numpy(mask)

    if options.img_dtype != 'float':
        obj_phase = _scale_to_integral_type(obj_phase, options.img_dtype, mask)

    with tifffile.TiffWriter(out_path, ome=True) as w:
        write_opts['metadata']['axes'] = 'ZYX' if stack else 'YX'
        w.write(obj_phase, **write_opts)


def _save_object_mag(state: ReconsState, out_path: Path, options: SaveOptions, stack: bool = False):
    crop = options.crop_roi
    write_opts = _obj_write_opts(state.object, stack=stack, crop=crop)

    xp = get_array_module(state.object.data)
    obj_mag = abs2(state.object.data)
    if crop:
        obj_mag = obj_mag[..., *state.object.sampling.get_region_crop()]
        mask = numpy.ones(obj_mag.shape[-2:], dtype=numpy.bool_)
    else:
        # include whole image, but only scale based on ROI
        mask = state.object.sampling.get_region_mask(numpy)

    if not stack:
        obj_mag = xp.prod(obj_mag, axis=0)

    obj_mag = to_numpy(obj_mag)
    if options.img_dtype != 'float':
        obj_mag = _scale_to_integral_type(obj_mag, options.img_dtype, mask, min_range=0.2)

    with tifffile.TiffWriter(out_path, ome=True) as w:
        write_opts['metadata']['axes'] = 'ZYX' if stack else 'YX'
        w.write(obj_mag, **write_opts)



_SAVE_FUNCS: t.Dict[str, t.Callable[[ReconsState, Path, SaveOptions], t.Any]] = {
    'probe': _save_probe,
    'probe_mag': _save_probe_mag,
    'probe_recip': _save_probe_recip,
    'probe_recip_mag': _save_probe_recip_mag,
    'object_phase_stack': partial(_save_object_phase, stack=True),
    'object_phase_sum': partial(_save_object_phase, stack=False),
    'object_mag_stack': partial(_save_object_mag, stack=True),
    'object_mag_sum': partial(_save_object_mag, stack=False),
}