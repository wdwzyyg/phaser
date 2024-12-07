import numpy
from numpy.typing import NDArray
from numpy.random import SeedSequence, PCG64, BitGenerator, Generator

import typing as t


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


def create_sparse_groupings(shape: t.Union[int, t.Iterable[int], NDArray[numpy.floating]], grouping: int = 8, seed: t.Any = None) -> list[NDArray[numpy.int64]]:
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

    rng = create_rng(seed, 'groupings')
    rng.shuffle(idxs)
    return numpy.array_split(idxs.T, numpy.ceil(idxs.shape[0] / grouping).astype(numpy.int64), axis=-1)


def create_compact_groupings(positions: NDArray[numpy.floating], grouping: int = 8, seed: t.Any = None) -> list[NDArray[numpy.int64]]:
    """
    Partition the indices of `positions` into groups of maximum size `grouping`, such that each group is spatially compact.

    Uses k-means clustering to ensure groups are spatially compact.

    Returns a list of groups. Each group can be used to index an array `arr` of shape `shape`:
    `arr[*group]`
    """
    from k_means_constrained import KMeansConstrained

    rng = create_rng(seed, 'groupings')
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