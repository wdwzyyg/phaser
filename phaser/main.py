import multiprocessing.connection
import base64
import os
import json
from multiprocessing.connection import Connection
import typing as t

import jax
from jax.tree_util import Partial
import numpy
from numpy.typing import NDArray, ArrayLike
import scipy.optimize
import tqdm
from ptycho_lebeau.metadata import AnyMetadata

from phaser.utils.num import to_complex_dtype, Sampling
from phaser.utils.num import fft2, ifft2, abs2, to_numpy
from phaser.utils.misc import create_groupings
from phaser.io.empad import load_4d
from phaser.utils.physics import Electron
from phaser.utils.optics import make_focused_probe, make_hermetian_modes, fourier_shift_filter, fresnel_propagator
from phaser.utils.scan import make_raster_scan
from phaser.utils.object import ObjectSampling, random_phase_object

from phaser.web.util import ConnectionWrapper


def main(connection: t.Optional[Connection] = None):
    xp = jax.numpy
    dtype = numpy.float32
    complex_dtype = to_complex_dtype(dtype)

    if connection is not None:
        # TODO wrap this in a context handler
        conn = ConnectionWrapper(connection)
    else:
        conn = None

    meta = AnyMetadata.parse_file("/Users/colin/Downloads/mos2/1/mos2/mos2_0.00_dstep1.0.json")

    step = 1
    scan_crop = (slice(0, None, step), slice(0, None, step))

    print("Loading raw data...")
    assert meta.path is not None
    assert meta.raw_filename is not None
    raw = load_4d(meta.path / meta.raw_filename)[scan_crop]

    # fftshift patterns to corners
    raw = numpy.fft.fftshift(raw, axes=(-1, -2)).reshape((-1, *raw.shape[-2:]))
    # normalize patterns
    raw /= numpy.sum(raw, axis=(-2, -1))[:, None, None]

    raw = numpy.maximum(raw, 0.)  # clamp to zero (is there a better way to do this?)

    wavelength = Electron(meta.voltage).wavelength
    print(wavelength)

    kstep = meta.diff_step*1e-3 / wavelength
    (b, a) = (1/kstep,) * 2

    grid = Sampling(raw.shape[-2:], extent=(b, a))

    print(f"Initializing probe...")
    init_probe = make_focused_probe(*grid.recip_grid(dtype=dtype, xp=xp), wavelength, meta.conv_angle, defocus=meta.defocus*1e10)
    init_probes = make_hermetian_modes(init_probe, 4, powers=0.1)
    print(f"Initialized probe.")

    if conn is not None:
        conn.send(('init_probe', to_numpy(init_probes)))

    print(f"Initializing scan...")
    (n_x, n_y) = meta.scan_shape
    scan = make_raster_scan((n_y, n_x), numpy.array(meta.scan_step[::-1]) * 1e10, dtype=dtype)[scan_crop].reshape(-1, 2)
    scan = xp.array(scan)

    if meta.scan_correction is not None:
        correction_matrix = numpy.array(meta.scan_correction, dtype=dtype)[::-1, ::-1]
        scan = (correction_matrix @ scan.T).T

    n_slices = 1
    slice_thickness = 10.

    print(f"Initializing object...")
    obj_grid = ObjectSampling.from_scan(scan, grid.sampling, pad=grid.extent / 2. + grid.sampling)
    init_obj = random_phase_object((n_slices, *obj_grid.shape), dtype=complex_dtype, xp=xp)

    if conn is not None:
        conn.send(('init_obj', to_numpy(init_obj)))

    raw = xp.array(raw)
    obj = init_obj.copy()
    probes = init_probes.copy()

    (ky, kx) = grid.recip_grid(dtype=dtype, xp=xp)
    cutout_shape = probes.shape[-2:]

    propagator = fresnel_propagator(ky, kx, wavelength, delta_z=slice_thickness)
    k2 = ky**2 + kx**2
    bandwidth_mask = (k2 < float(numpy.min(grid.k_max) * 2./3.)**2)
    propagator *= bandwidth_mask

    @jax.jit
    def collect_group(group: NDArray[numpy.integer], obj: jax.Array, probes: jax.Array):
        group_scan = scan[*group]
        group_subpx_filters = fourier_shift_filter(ky, kx, obj_grid.get_subpx_shifts(group_scan, cutout_shape))

        # shape (mode, group, y, x)
        group_probes = ifft2(fft2(probes)[:, None, ...] * group_subpx_filters)
        group_objs = obj_grid.get_view_at_pos(obj, group_scan, cutout_shape)
        return (group_probes, group_objs)

    @jax.jit
    def forward_model(group_probes: jax.Array, group_objs: jax.Array):
        for slice_i in range(obj.shape[0]):
            group_probes = group_probes * group_objs[:, slice_i]
            if slice_i + 1 < obj.shape[0]:
                group_probes = ifft2(fft2(group_probes) * propagator)

        # sum over incoherent modes
        intensity = xp.sum(abs2(fft2(group_probes)), axis=0)
        return intensity

    @jax.jit
    def loss(group: NDArray[numpy.integer], obj: jax.Array, probes: jax.Array):
        group_patterns = raw[*group]
        (group_probes, group_objs) = collect_group(group, obj, probes)
        intensity = forward_model(group_probes, group_objs)
        #obj_penalty = 1e-1 * xp.linalg.norm(xp.angle(obj), 2.) / obj.size

        # TV regularization of object phase
        phase = xp.angle(group_objs)
        obj_penalty = xp.sqrt(xp.sum(xp.diff(phase, axis=-2)**2) + xp.sum(xp.diff(phase, axis=-1)**2)) / group_objs.size

        return xp.linalg.norm(intensity - group_patterns) / group.size + 5e-2 * obj_penalty

    grouping = 128
    iterations = 50
    step_size = 1e2

    iter_sses: t.List[float] = []

    step_frac: float = 1.0

    obj_grad_prev: t.Optional[jax.Array] = None
    probe_grad_prev: t.Optional[jax.Array] = None
    step_size_prev: t.Optional[jax.Array] = None
    beta = 0.0

    groups = create_groupings(raw.shape[0], grouping)

    sses: t.List[float] = []
    pattern_intensities: t.List[float] = []

    for (group_i, group) in enumerate(groups):
        patterns = forward_model(*collect_group(group, obj, probes))
        pattern_intensity = xp.sum(patterns) / patterns.shape[0]
        pattern_intensities.append(float(pattern_intensity))
        sse = loss(group, obj, probes)
        sses.append(sse)

    mean_sse = float(numpy.mean(sses))
    iter_sses.append(mean_sse)
    pattern_intensity = numpy.mean(pattern_intensities)
    print(f"Mean pattern intensity: {pattern_intensity}. Scaling probe")
    probes /= pattern_intensity.astype(dtype)

    pbar = tqdm.trange(iterations)
    for i in pbar:
        sum_obj_grad = jax.numpy.zeros_like(obj)
        sum_probe_grad = jax.numpy.zeros_like(probes)
        sses: t.List[float] = []

        groups = create_groupings(raw.shape[0], grouping)

        for (group_i, group) in enumerate(groups):
            sse, (obj_grad, probe_grad) = jax.value_and_grad(loss, argnums=(1, 2))(group, obj, probes)
            # jax returns the conjugate of the gradient/steepest ascent
            obj_grad = obj_grad.conj()
            probe_grad = probe_grad.conj()

            sum_obj_grad += obj_grad
            sum_probe_grad += probe_grad
            sses.append(sse)

        mean_sse = float(numpy.mean(sses))
        iter_sses.append(mean_sse)
        #if len(iter_sses) >= 2 and mean_sse > iter_sses[-2]:
        #    # converged
        #    print(f"Converged in {i+1} iterations")
        #    break

        obj_grad = sum_obj_grad / len(groups)
        probe_grad = sum_probe_grad / len(groups)

        update_dir_obj = -obj_grad
        update_dir_probe = -probe_grad

        # Polak-Ribière
        if probe_grad_prev is not None and obj_grad_prev is not None and step_size_prev is not None:
            beta = ((xp.sum(-obj_grad * (-obj_grad - -obj_grad_prev)).real + xp.sum(-probe_grad * (-probe_grad - -probe_grad_prev)).real) / (
                xp.sum(abs2(obj_grad_prev)) + xp.sum(abs2(probe_grad_prev))
            )).astype(dtype)
            #beta = xp.array(0., dtype=dtype)
            #print(f"beta: {beta}")
            beta = xp.where(beta > 10., 0., xp.maximum(beta, 0.))
            update_dir_obj += beta * obj_update_prev
            update_dir_probe += beta * probe_update_prev

        result = scipy.optimize.minimize_scalar(lambda step_size: loss(group, obj + update_dir_obj * step_size.astype(dtype),
                                                                    probes + update_dir_probe * step_size.astype(dtype)),
                                                method='bounded', bounds=(1e1, 1e8), options={'maxiter': 50, 'xatol': 1e3})
        if beta > 0 and result.fun > sse:
            # redo search along original direction
            update_dir_obj = -obj_grad
            update_dir_probe = -probe_grad
            result = scipy.optimize.minimize_scalar(lambda step_size: loss(group, obj + update_dir_obj * step_size.astype(dtype),
                                                                    probes + update_dir_probe * step_size.astype(dtype)),
                                                    method='bounded', bounds=(1e1, 1e8), options={'maxiter': 50, 'xatol': 1e3})
        step_size = result.x.astype(dtype)
        #step_size = line_search(Partial(loss, group), update_dir_obj, update_dir_probe, start_step_size=1e8, min_step_size=1e1, slope=0.5)

        obj_update_prev = update_dir_obj
        probe_update_prev = update_dir_probe
        obj_grad_prev = obj_grad
        probe_grad_prev = probe_grad
        step_size_prev = step_size

        obj += update_dir_obj * step_size * step_frac
        probes += update_dir_probe * step_size * step_frac

        assert obj.dtype == probes.dtype == numpy.complex64
        # mean error summed across each pattern
        #print(f"RMS: {mean_sse:10.5e}")
        #print(f"step size: {step_size:10.5e}")
        #pbar.set_description(f"RMS: {mean_sse:10.5e}  step size: {step_size:10.5e}  beta: {beta:.3f}")
        pbar.set_postfix({'rms': mean_sse, 'step_size': step_size, 'beta': beta})

        if conn is not None:
            [conn.send(v) for v in [
                ('probe', to_numpy(probes)),
                ('obj', to_numpy(obj)),
                ('progress', (numpy.arange(len(iter_sses)), numpy.array(iter_sses))),
            ]]