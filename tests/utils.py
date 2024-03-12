from itertools import chain
import inspect
import typing as t

import pytest

CallableT = t.TypeVar('CallableT', bound=t.Callable)
P = t.ParamSpec('P')
T = t.TypeVar('T')


def _wrap_pytest(wrapper: CallableT, wrapped: t.Callable,
                 mod_params: t.Optional[t.Callable[[t.Sequence[inspect.Parameter]], t.Sequence[inspect.Parameter]]] = None
) -> CallableT:
    # hacks to allow pytest to find fixtures in wrapped functions
    old_sig = inspect.signature(wrapped)
    params = tuple(old_sig.parameters.values())
    if mod_params is not None:
        params = mod_params(params)
    new_sig = old_sig.replace(parameters=params)

    testmark = getattr(wrapped, "pytestmark", []) + getattr(wrapper, "pytestmark", [])
    if len(testmark) > 0:
        wrapper.pytestmark = testmark  # type: ignore
    wrapper.__name__ = wrapped.__name__
    wrapper.__doc__ = wrapped.__doc__
    wrapper.__signature__ = new_sig  # type: ignore
    return wrapper


def with_backends(
    *backends: t.Union[str, t.Iterable[str]]
) -> t.Callable[[t.Callable[P, T]], t.Callable[P, T]]:
    """Run a test on the specified compute backends"""
    backends = t.cast(t.Tuple[str, ...],
        tuple(chain.from_iterable((b,) if isinstance(b, str) else b for b in backends))
    )

    def decorator(f: t.Callable[P, T]) -> t.Callable[P, T]:
        return pytest.mark.parametrize('backend', [
            pytest.param(backend, marks=getattr(pytest.mark, backend))
            for backend in backends
        ])(f)

    return decorator


def get_backend_module(backend: str):
    """Get the module `xp` associated with a compute backend"""
    backend = backend.lower()
    if backend not in ('cuda', 'cpu'):
        raise ValueError(f"Unknown backend '{backend}'")

    if backend == 'cuda' and not t.TYPE_CHECKING:
        import cupy
        return cupy

    import numpy
    return numpy