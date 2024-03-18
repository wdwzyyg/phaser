
import pytest
import numpy

BACKENDS: set[str] = set(('cpu', 'cuda'))
AVAILABLE_BACKENDS: set[str] = set(('cpu',))

try:
    import cupy
    if cupy.cuda.runtime.getDeviceCount() > 0:
        AVAILABLE_BACKENDS.add('cuda')
except ImportError:
    pass


def pytest_addoption(parser: pytest.Parser):
    parser.addoption("--save-expected", action="store_true", dest='save-result', default=False,
                     help="Overwrite expected files with the results of tests.")


def pytest_runtest_setup(item):
    required_backends = BACKENDS.intersection(mark.name for mark in item.iter_markers())
    missing = required_backends - AVAILABLE_BACKENDS
    if len(missing):
        missing = ', '.join(missing)
        pytest.skip(f"Backend {missing} is not available")


@pytest.fixture(scope='function')
def file_contents_array(request: pytest.FixtureRequest, pytestconfig: pytest.Config) -> numpy.ndarray:
    from tests.utils import read_array, EXPECTED_PATH, OVERWRITE_EXPECTED

    marker = request.node.get_closest_marker('expected_filename')
    assert marker is not None
    name = str(marker.args[0])

    if pytestconfig.getoption("save-expected"):
        return OVERWRITE_EXPECTED  # type: ignore

    try:
        return read_array(EXPECTED_PATH / name)
    except Exception as e:
        raise RuntimeError("Failed to load expected result") from e