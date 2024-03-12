
import pytest

BACKENDS: set[str] = set(('cpu', 'cuda'))
AVAILABLE_BACKENDS: set[str] = set(('cpu',))

try:
    import cupy
    if cupy.cuda.runtime.getDeviceCount() > 0:
        AVAILABLE_BACKENDS.add('cuda')
except ImportError:
    pass


def pytest_runtest_setup(item):
    required_backends = BACKENDS.intersection(mark.name for mark in item.iter_markers())
    missing = required_backends - AVAILABLE_BACKENDS
    if len(missing):
        missing = ', '.join(missing)
        pytest.skip(f"Backend {missing} is not available")

