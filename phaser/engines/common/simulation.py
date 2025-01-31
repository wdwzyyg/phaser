from dataclasses import dataclass
import logging
import typing as t

import numpy
from numpy.typing import NDArray, DTypeLike

from phaser.utils.num import cast_array_module, to_complex_dtype, fft2, ifft2, is_jax, to_numpy
from phaser.utils.misc import FloatKey
from phaser.utils.optics import fresnel_propagator, fourier_shift_filter
from phaser.state import ReconsState
from phaser.hooks.solver import NoiseModel, ConstraintRegularizer, StateT

logger = logging.getLogger(__name__)


@dataclass(init=False)
class SimulationState:
    state: ReconsState
    noise_model_state: t.Any
    patterns: NDArray[numpy.floating]
    pattern_mask: NDArray[numpy.floating]

    ky: NDArray[numpy.floating]
    kx: NDArray[numpy.floating]

    noise_model: NoiseModel
    regularizers: t.Tuple[ConstraintRegularizer, ...]
    xp: t.Any
    dtype: DTypeLike
    start_iter: int

    propagators: t.Optional[NDArray[numpy.complexfloating]] = None

    def __init__(
        self, *,
        state: ReconsState,
        noise_model: NoiseModel[StateT],
        regularizers: t.Tuple[ConstraintRegularizer, ...],
        patterns: NDArray[numpy.floating],
        pattern_mask: NDArray[numpy.floating],
        xp: t.Any,
        dtype: DTypeLike,
        regularizer_states: t.Optional[t.Tuple[ConstraintRegularizer, ...]] = None,
        noise_model_state: t.Optional[StateT] = None,
        start_iter: t.Optional[int] = None
    ):
        self.xp = xp
        self.dtype = dtype
        self.state = state
        self.patterns = patterns
        self.pattern_mask = xp.array(pattern_mask)

        self.regularizers = regularizers
        self.noise_model = noise_model
        (self.ky, self.kx) = state.probe.sampling.recip_grid(dtype=dtype, xp=xp)

        self.start_iter = start_iter if start_iter is not None else self.state.iter.total_iter
        self.propagators = None

        self.noise_model_state = noise_model_state or noise_model.init_state(self)

        self.regularizer_states = regularizer_states if regularizer_states is not None else tuple(
            reg.init_state(self) for reg in regularizers
        )


def make_propagators(sim: SimulationState) -> t.Optional[NDArray[numpy.complexfloating]]:
    delta_zs = numpy.diff(to_numpy(sim.state.object.zs))
    if len(delta_zs) == 0:
        return None

    unique_zs = set(map(FloatKey, delta_zs))
    complex_dtype = to_complex_dtype(sim.dtype)

    bwlim = numpy.min(sim.state.probe.sampling.k_max) * 0.9 #* 2./3.

    k2 = sim.ky**2 + sim.kx**2
    bwlim_mask = k2 <= bwlim**2
    logger.info(f"Bandwidth limit: {bwlim * sim.state.wavelength * 1e3:6.2f} mrad")

    props = {
        z: fresnel_propagator(sim.ky, sim.kx, sim.state.wavelength, z).astype(complex_dtype) * bwlim_mask
        for z in unique_zs
    }

    return cast_array_module(sim.xp).stack(
        [props[FloatKey(z)] for z in delta_zs],
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


def slice_forwards(
    props: t.Optional[NDArray[numpy.complexfloating]],
    state: StateT,
    f: t.Callable[[int, t.Optional[NDArray[numpy.complexfloating]], StateT], StateT]
) -> StateT:
    if props is None:
        return f(0, None, state)

    n_slices = len(props) + 1

    if is_jax(props):
        import jax
        state = jax.lax.fori_loop(0, n_slices - 1, lambda slice_i, state: f(slice_i, props[slice_i], state), state, unroll=False)
        return f(n_slices - 1, None, state)

    for slice_i in range(n_slices - 1):
        state = f(slice_i, props[slice_i], state)

    return f(n_slices - 1, None, state)


def slice_backwards(
    props: t.Optional[NDArray[numpy.complexfloating]],
    state: StateT,
    f: t.Callable[[int, t.Optional[NDArray[numpy.complexfloating]], StateT], StateT]
) -> StateT:
    if props is None:
        return f(0, None, state)

    n_slices = len(props) + 1

    if is_jax(props):
        import jax
        state = jax.lax.fori_loop(1, n_slices, lambda i, state: f(n_slices - i, props[n_slices - i - 1], state), state, unroll=False)
        return f(0, None, state)

    for slice_i in range(n_slices - 1, 0, -1):
        state = f(slice_i, props[slice_i - 1], state)

    return f(0, None, state)


try:
    import jax.tree_util
except ImportError:
    pass
else:
    jax.tree_util.register_dataclass(
        SimulationState,
        ('state', 'noise_model_state', 'patterns', 'pattern_mask', 'start_iter'),
        ('xp', 'dtype', 'noise_model', 'regularizers'),
        ('ky', 'kx', 'propagators')
    )