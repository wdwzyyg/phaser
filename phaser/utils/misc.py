import dataclasses
import math
import typing as t

import numpy
from numpy.typing import NDArray
from numpy.random import SeedSequence, PCG64, BitGenerator, Generator
from typing_extensions import dataclass_transform


T = t.TypeVar('T')


def _proc_seed(seed: object, entropy: object = None) -> SeedSequence:
    """
    Process a random seed, along with additional entropy to use the same
    seed for multiple applications.
    """
    if seed is None:
        return SeedSequence()
    if isinstance(seed, SeedSequence):
        seed = seed.entropy

    # hash our seed and our extra entropy

    from hashlib import sha256
    import json

    state = sha256()
    state.update(json.dumps(seed).encode('utf-8'))
    if entropy is not None:
        state.update(json.dumps(entropy).encode('utf-8'))
    return SeedSequence(numpy.frombuffer(state.digest(), dtype=numpy.uint32))


def create_rng(seed: object = None, entropy: object = None) -> Generator:
    """
    Create a numpy `PCG64` `Generator` using the initial seed (if specified),
    and some additional entropy.

    If `seed` is an existing `Generator` or `BitGenerator`, it's returned.
    Otherwise, `seed` is used along with `entropy` to construct a high-quality
    initial seed.

    The seed and entropy can be anything JSON-writable.

    If no seed is specified, numpy's default methods are used to construct a high-quality
    seed. With a fixed seed specified, this function is designed to provide deterministic
    behavior across different platforms and for long periods of time.
    """
    if isinstance(seed, Generator):
        return seed
    elif isinstance(seed, BitGenerator):
        return Generator(seed)
    seq = _proc_seed(seed, entropy)
    return Generator(PCG64(seq))


def create_rng_group(n: int, seed: object = None, entropy: object = None) -> t.Tuple[Generator, ...]:
    """
    Create a group of `n` distinct `PCG64` `BitGenerator`s using the initial seed (if specified),
    and some additional entropy.

    If `seed` is an existing `Generator` or `BitGenerator`, its underlying seed
    sequence is used to construct the group. Otherwise, `seed` is used along with
    `entropy` to construct a high-quality initial seed.

    The seed and entropy can be anything JSON-writable.

    If no seed is specified, numpy's default methods are used to construct a high-quality
    seed. With a fixed seed specified, this function is designed to provide deterministic
    behavior across different platforms and for long periods of time.
    """
    if isinstance(seed, Generator):
        seq = seed.bit_generator.seed_seq
    elif isinstance(seed, BitGenerator):
        seq = seed.seed_seq
    else:
        seq = _proc_seed(seed, entropy)

    return tuple(map(Generator, map(PCG64, t.cast(SeedSequence, seq).spawn(n))))


def shuffled(vals: t.Sequence[T], seed: t.Any = None, i: int = 0) -> t.Iterator[T]:
    """
    Return an iterator which gives `vals` in a random order.
    """
    idxs = numpy.arange(len(vals))
    rng = create_rng(seed, f"shuffle_{i}")
    rng.shuffle(idxs)

    for idx in idxs:
        yield vals[int(idx)]


def create_sparse_groupings(shape: t.Union[int, t.Iterable[int], NDArray[numpy.floating]], grouping: int = 8,
                            seed: t.Any = None, i: int = 0) -> list[NDArray[numpy.int64]]:
    """
    Randomly partition the indices of `shape` into groups of maximum size `grouping`.

    Returns a list of groups. Each group can be used to index an array `arr` of shape `shape`:
    `arr[*group]`
    """
    if isinstance(shape, int):
        shape = (shape,)
    if not isinstance(shape, (tuple, list)):
        # assume `shape` is a list of positions
        shape = shape.shape[:-1]  # type: ignore

    idxs = numpy.indices(shape)  # type: ignore
    idxs = idxs.reshape(idxs.shape[0], -1).T

    rng = create_rng(seed, f'groupings_{i}' if i != 0 else 'groupings')
    rng.shuffle(idxs)
    return numpy.array_split(idxs.T, numpy.ceil(idxs.shape[0] / grouping).astype(numpy.int64), axis=-1)


def create_compact_groupings(positions: NDArray[numpy.floating], grouping: int = 8,
                             seed: t.Any = None, i: int = 0) -> list[NDArray[numpy.int64]]:
    """
    Partition the indices of `positions` into groups of maximum size `grouping`, such that each group is spatially compact.

    Uses k-means clustering to ensure groups are spatially compact.

    Returns a list of groups. Each group can be used to index an array `arr` of shape `shape`:
    `arr[*group]`
    """
    from k_means_constrained import KMeansConstrained

    rng = create_rng(seed, f'groupings_{i}' if i != 0 else 'groupings')
    random_state = numpy.random.RandomState(rng.bit_generator)

    idxs = numpy.indices(positions.shape[:-1])
    idxs = idxs.reshape(idxs.shape[0], -1)
    n_groups = numpy.ceil(idxs.shape[-1] / grouping).astype(numpy.int64)

    kmeans = KMeansConstrained(n_groups, size_max=grouping, init='random', n_init=1, random_state=random_state)
    labels = kmeans.fit_predict(positions.reshape(-1, positions.shape[-1]))
    #_, labels = kmeans2(
    #    positions.reshape(-1, positions.shape[-1]),
    #    n_groups, iter=20, minit='points', missing='raise', seed=rng
    #)

    return [
        idxs[..., labels == i]
        for i in range(n_groups)
    ]


def mask_fraction_of_groups(n_groups: int, fraction: float) -> NDArray[numpy.bool_]:
    n_required = max(1, math.ceil(n_groups * fraction))
    if n_required >= n_groups:
        return numpy.ones(n_groups, dtype=numpy.bool_)

    every = n_groups // n_required  # guaranteed > 1
    mask = numpy.zeros(n_groups, dtype=numpy.bool_)
    mask[::every] = 1

    return mask


class FloatKey(float):
    def __hash__(self):
        return float.__hash__(round(self, 5))

    def __eq__(self, other: t.Any) -> bool:
        return isinstance(other, float) and \
            round(self, 5) == round(other, 5)


@t.overload
@dataclass_transform(kw_only_default=False, frozen_default=False)
def jax_dataclass(cls: t.Type[T], /, *,
    init: bool = True, kw_only: bool = False, frozen: bool = False,
    static_fields: t.Sequence[str] = (), drop_fields: t.Sequence[str] = (),
) -> t.Type[T]:
    ...

@t.overload
@dataclass_transform(kw_only_default=False, frozen_default=False)
def jax_dataclass(*,
    init: bool = True, kw_only: bool = False, frozen: bool = False,
    static_fields: t.Sequence[str] = (), drop_fields: t.Sequence[str] = (),
) -> t.Callable[[t.Type[T]], t.Type[T]]:
    ...

def jax_dataclass(cls: t.Optional[t.Type[T]] = None, /, *,
    init: bool = True, kw_only: bool = False, frozen: bool = False,
    static_fields: t.Sequence[str] = (), drop_fields: t.Sequence[str] = (),
) -> t.Union[t.Type[T], t.Callable[[t.Type[T]], t.Type[T]]]:
    if cls is None:
        return lambda cls: jax_dataclass(cls, init=init, kw_only=kw_only, frozen=frozen,
                                         static_fields=static_fields, drop_fields=drop_fields)

    cls = dataclasses.dataclass(init=init, kw_only=kw_only, frozen=frozen)(cls)
    _register_dataclass(cls, static_fields=static_fields, drop_fields=drop_fields)
    return cls


def _register_dataclass(cls: type, static_fields: t.Sequence[str], drop_fields: t.Sequence[str]):
    try:
        from jax.tree_util import register_pytree_with_keys
    except ImportError:
        return

    fields = dataclasses.fields(cls)
    field_names = {field.name for field in fields}

    if (extra := set(static_fields).difference(field_names)):
        raise ValueError(f"Unknown field(s) passed to 'static_fields': {', '.join(map(repr, extra))}")
    if (extra := set(drop_fields).difference(field_names)):
        raise ValueError(f"Unknown field(s) passed to 'drop_fields': {', '.join(map(repr, extra))}")

    data_fields = tuple(field_names.difference(static_fields).difference(drop_fields))

    def flatten_with_keys(x: t.Any, /) -> tuple[t.Iterable[tuple[str, t.Any]], t.Hashable]:
        meta = tuple(getattr(x, name) for name in static_fields)
        trees = tuple((name, getattr(x, name)) for name in data_fields)
        return trees, meta

    def unflatten(meta: t.Hashable, trees: t.Iterable[t.Any], /) -> t.Any:
        if not isinstance(meta, tuple):
            raise TypeError
        static_args = dict(zip(static_fields, meta, strict=True))
        data_args = dict(zip(data_fields, trees, strict=True))
        return cls(**static_args, **data_args)

    def flatten(x: t.Any, /) -> tuple[t.Iterable[t.Any], t.Hashable]:
        hashed = tuple(getattr(x, name) for name in static_fields)
        trees = tuple(getattr(x, name) for name in data_fields)
        return trees, hashed

    register_pytree_with_keys(cls, flatten_with_keys, unflatten, flatten)


def unwrap(val: t.Optional[T]) -> T:
    assert val is not None
    return val


__all__ = [
    'create_rng', 'create_rng_group',
    'create_sparse_groupings', 'create_compact_groupings',
    'mask_fraction_of_groups', 'FloatKey',
    'jax_dataclass', 'unwrap',
]
