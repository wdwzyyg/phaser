import builtins
from contextlib import contextmanager
from itertools import chain
import inspect
import sys
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


def get_backend_scipy(backend: str):
    """Get the scipy module associated with a compute backend"""
    backend = backend.lower()
    if backend not in ('cuda', 'cpu'):
        raise ValueError(f"Unknown backend '{backend}'")

    if backend == 'cuda' and not t.TYPE_CHECKING:
        import cupyx.scipy
        return cupyx.scipy

    import scipy
    return scipy


_import = builtins.__import__


class Importer:
    def __init__(self, prevented: t.Iterable[str] = ()):
        self.prevented: set[str] = set(prevented)

    def __call__(self, name, globals, locals, fromlist, level):
        if name in self.prevented:
            raise ImportError(f"Mocked ImportError for '{name}'")
        return _import(name, globals, locals, fromlist, level)


@contextmanager
def mock_importerror(*modulenames: t.Iterable[str]):
    modules = set(chain.from_iterable((n,) if isinstance(n, str) else n for n in modulenames))

    if not len(modules):
        # just run the body and return
        yield
        return

    # remove modules if they're already imported
    for module in modules:
        sys.modules.pop(module, None)

    # replace __import__ with our copy
    if not isinstance(builtins.__import__, Importer):
        builtins.__import__ = Importer()
    # and specify which modules to prevent
    builtins.__import__.prevented.update(modules)

    # run the context manager's body
    yield

    # un-prevent modules
    builtins.__import__.prevented.difference_update(modules)

    # if we're the last ones, change __import__ back
    if not len(builtins.__import__.prevented):
        builtins.__import__ = _import