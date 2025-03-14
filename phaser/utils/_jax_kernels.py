
from functools import partial
import typing as t

from numpy.typing import ArrayLike
import jax  # pyright: ignore[reportMissingImports]
import jax.numpy as jnp    # pyright: ignore[reportMissingImports]


def to_2d(arr: jax.Array) -> jax.Array:
    arr = arr.reshape(-1, *arr.shape[-1:])
    if arr.ndim == 1:
        return jax.lax.expand_dims(arr, [0])
    return arr


def to_3d(arr: jax.Array) -> jax.Array:
    arr = arr.reshape(-1, *arr.shape[-2:])
    if arr.ndim == 2:
        return jax.lax.expand_dims(arr, [0])
    return arr


@jax.jit
def set_cutouts(obj: jax.Array, cutouts: jax.Array, start_idxs: jax.Array) -> jax.Array:
    for cutout, start_idx in zip(to_3d(cutouts), to_2d(start_idxs)):
        obj = jax.vmap(lambda o: jax.lax.dynamic_update_slice(o, cutout, start_idx))(
            to_3d(obj)
        ).reshape(obj.shape)

    return obj


@jax.jit
def add_cutouts(obj: jax.Array, cutouts: jax.Array, start_idxs: jax.Array) -> jax.Array:
    zero_obj = jax.numpy.zeros_like(obj, shape=obj.shape[-2:])

    for cutout, start_idx in zip(to_3d(cutouts), to_2d(start_idxs)):
        obj += jax.lax.dynamic_update_slice(zero_obj, cutout, start_idx)

    return obj


@partial(jax.jit, static_argnums=2)
def get_cutouts(obj: jax.Array, start_idxs: jax.Array, cutout_shape: t.Tuple[int, int]) -> jax.Array:
    return jax.vmap(jax.vmap(lambda start_idx, obj: jax.lax.dynamic_slice(obj, start_idx, cutout_shape), (None, 0)), (0, None))(
        to_2d(start_idxs), to_3d(obj)
    ).reshape((*start_idxs.shape[:-1], *obj.shape[:-2], *cutout_shape))


@partial(jax.jit, static_argnums=0)
def outer(ufunc: t.Any, x: jax.Array, y: jax.Array) -> jax.Array:
    return jax.vmap(jax.vmap(ufunc, (None, 0)), (0, None))(x, y)


@partial(jax.jit, static_argnames=('output_shape', 'order', 'mode', 'cval'))
def affine_transform(
    input: jax.Array,
    matrix: ArrayLike,
    offset: t.Optional[ArrayLike] = None,
    output_shape: t.Optional[t.Tuple[int, ...]] = None,
    order: int = 1,
    mode: str = 'constant',
    cval: t.Any = 0.0,
) -> jax.Array:
    import jax.scipy.ndimage
    jax_mode = {'grid-constant': 'constant', 'grid-wrap': 'wrap'}.get(mode, mode)

    if output_shape is None:
        output_shape = input.shape

    indices = jnp.indices(output_shape, dtype=float)

    matrix = jnp.array(matrix)
    if matrix.shape == (input.ndim + 1, input.ndim + 1):
        # homogenous transform matrix
        coords = jnp.tensordot(
            matrix, jnp.stack((*indices, jnp.ones_like(indices[0])), axis=0), axes=1
        )[:-1]
    elif matrix.shape == (input.ndim,):
        coords = (indices.T * matrix + jnp.array(offset)).T
    else:
        raise ValueError(f"Expected matrix of shape ({input.ndim + 1}, {input.ndim + 1}) or ({input.ndim},), instead got shape {matrix.shape}")

    coords += jnp.finfo(coords.dtype).eps

    return jax.scipy.ndimage.map_coordinates(input, tuple(coords), order=order, mode=jax_mode, cval=cval)