import logging
from dataclasses import dataclass
import typing as t

import numpy
from numpy.typing import NDArray, DTypeLike

from phaser.utils.num import cast_array_module, to_complex_dtype, fft2, ifft2
from phaser.utils.misc import FloatKey
from phaser.utils.optics import fresnel_propagator, fourier_shift_filter
from phaser.state import ReconsState
from phaser.hooks.solver import NoiseModel, StateT


@dataclass(init=False)
class SimulationState(t.Generic[StateT]):
    state: ReconsState
    noise_model_state: StateT
    patterns: NDArray[numpy.floating]
    pattern_mask: NDArray[numpy.floating]

    ky: NDArray[numpy.floating]
    kx: NDArray[numpy.floating]

    noise_model: NoiseModel[StateT]
    xp: t.Any
    dtype: DTypeLike
    start_iter: int

    propagators: t.Optional[NDArray[numpy.complexfloating]] = None

    def __init__(
        self, *,
        state: ReconsState,
        noise_model: NoiseModel[StateT],
        patterns: NDArray[numpy.floating],
        pattern_mask: NDArray[numpy.floating],
        xp: t.Any,
        dtype: DTypeLike,
        noise_model_state: t.Optional[StateT] = None,
        start_iter: t.Optional[int] = None
    ):
        self.xp = xp
        self.dtype = dtype
        self.state = state
        self.patterns = patterns
        self.pattern_mask = pattern_mask

        self.noise_model = noise_model
        self.noise_model_state = noise_model_state or noise_model.init_state()
        (self.ky, self.kx) = state.probe.sampling.recip_grid(dtype=dtype, xp=xp)

        self.start_iter = start_iter if start_iter is not None else self.state.iter.total_iter
        self.propagators = None


def make_propagators(sim: SimulationState) -> NDArray[numpy.complexfloating]:
    unique_zs = set(map(FloatKey, sim.state.object.zs))
    complex_dtype = to_complex_dtype(sim.dtype)

    props = {
        z: fresnel_propagator(sim.ky, sim.kx, sim.state.wavelength, z).astype(complex_dtype)
        for z in unique_zs
    }

    return cast_array_module(sim.xp).stack(
        [props[FloatKey(z)] for z in sim.state.object.zs],
        axis = 0
    )

@t.overload
def cutout_group(sim: SimulationState, group: NDArray[numpy.integer], return_filters: t.Literal[False] = False) -> t.Tuple[NDArray[numpy.complexfloating], NDArray[numpy.complexfloating], NDArray[numpy.floating]]:
    ...

@t.overload
def cutout_group(sim: SimulationState, group: NDArray[numpy.integer], return_filters: t.Literal[True]) -> t.Tuple[NDArray[numpy.complexfloating], NDArray[numpy.complexfloating], NDArray[numpy.floating], NDArray[numpy.complexfloating]]:
    ...

def cutout_group(sim: SimulationState, group: NDArray[numpy.integer], return_filters: bool = False):
    """Returns (probe, obj) in the cutout region"""
    probes = sim.state.probe.data

    group_scan = sim.state.scan[*group]
    group_obj = sim.state.object.sampling.get_view_at_pos(sim.state.object.data, group_scan, probes.shape[-2:])
    # group probes in real space
    # shape (len(group), 1, Ny, Nx)
    group_subpx_filters = fourier_shift_filter(sim.ky, sim.kx, sim.state.object.sampling.get_subpx_shifts(group_scan, probes.shape[-2:]))[:, None, ...]
    # shape (len(group), probe modes, Ny, Nx)
    shifted_probes = ifft2(fft2(probes) * group_subpx_filters)

    if return_filters:
        return (shifted_probes, group_obj, group_scan, group_subpx_filters)

    return (shifted_probes, group_obj, group_scan)


try:
    import jax.tree_util
except ImportError:
    pass
else:
    jax.tree_util.register_dataclass(
        SimulationState,
        ('state', 'noise_model_state', 'patterns', 'pattern_mask', 'start_iter'),
        ('xp', 'dtype', 'noise_model'),
        ('ky', 'kx', 'propagators')
    )