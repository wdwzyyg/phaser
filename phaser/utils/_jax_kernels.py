
from functools import partial
import typing as t

import jax
from jax.typing import ArrayLike


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


def _multidim_slice(obj: jax.Array, start_idx: jax.Array, cutout_shape: t.Tuple[int, int]):
    return jax.vmap(lambda obj: jax.lax.dynamic_slice(obj, start_idx, cutout_shape))(
        to_3d(obj)
    ).reshape((*obj.shape[:-2], *cutout_shape))


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
    return jax.vmap(lambda start_idx: _multidim_slice(obj, start_idx, cutout_shape))(
        start_idxs.reshape(-1, 2)
    ).reshape((*start_idxs.shape[:-1], *obj.shape[:-2], *cutout_shape))


