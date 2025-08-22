
import sys
from pathlib import Path
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import typing as t

import click
from click.exceptions import Exit
from rich.prompt import Prompt
from rich.theme import Theme
from rich.console import Console

from .metadata import EmpadMetadata, UnscannedMetadata


T = t.TypeVar('T')


def default_console() -> Console:
    return Console(theme=Theme({
        'warning': 'bold yellow',
        'error': 'bold red',
        'info': 'blue',
        'prompt.invalid': 'bold red',
    }))


def prompt_ask(name, unit=None, default=None, default_str: str = "", console=None, validate=float, err="Please enter a number") -> t.Tuple[str, t.Any]:
    if not console:
        console = default_console()
    while True:
        str_val = Prompt.ask(f"{name} \\[{unit}]", default=default_str, console=console)
        if str_val == default_str:
            return (default_str, default)
        try:
            val = validate(str_val)
            return (str_val, val)
        except Exception as e:
            console.print(f"[prompt.invalid]{err}")


@dataclass
class Parameter(t.Generic[T], ABC):
    name: str
    default_val: t.Optional[T]

    @abstractmethod
    def ask(self, metadata: EmpadMetadata, console: t.Optional[Console] = None) -> T:
        ...


@dataclass
class OptStringParam(Parameter[t.Optional[str]]):
    def ask(self, metadata: EmpadMetadata, console: t.Optional[Console] = None) -> t.Optional[str]:
        val = Prompt.ask(self.name, default=self.default_val, console=console)
        if val is not None:
            self.default_val = val
        return val


@dataclass
class FloatParam(Parameter[float]):
    unit: t.Optional[str]
    default_str: str
    conv_factor: t.Optional[float] = None

    def ask(self, metadata: EmpadMetadata, console: t.Optional[Console] = None) -> float:
        (val_str, val) = prompt_ask(self.name, self.unit, self.default_val, self.default_str, console, validate=float)
        self.default_str = val_str
        self.default_val = val
        if self.conv_factor:
            return val * self.conv_factor
        return val


def default_params() -> t.Sequence[t.Tuple[str, Parameter]]:
    return [
        ('conv_angle', FloatParam("Convergence angle", 18.9, "mrad", "18.9")),
        ('defocus', FloatParam("Defocus (CW is +)", 0.0, "nm", "0.0", 1e-9)),             # nm to m
        ('beam_current', FloatParam("Approx. beam current", 30.0, "pA", "30.0", 1e-12)),  # pA to A
        ('diff_step', CameraParam(18.8 / 24., 0.4575)),
        ('adu', AduParam(578, 200000.)),  # calibrated 2022-12-14
        ('author', OptStringParam("Author", None)),
    ]


@dataclass(init=False)
class CameraParam(Parameter[float]):
    """Maps f"{camera_length:0.3f}" to diff_step values"""
    used: t.Dict[str, float] = field(default_factory=dict)
    """ Camera length associated with default_val (m)"""
    default_camera_length: float = 0.4575

    def __init__(self, diff_step: float, camera_length: float):
        self.name = 'Diffraction pixel spacing'
        self.default_val = diff_step
        self.used = {}
        self.default_camera_length = camera_length

    def ask(self, metadata: EmpadMetadata, console: t.Optional[Console] = None) -> float:
        if not console:
            console = default_console()
        camera_length = metadata.camera_length
        camera_length_str = f"{camera_length:0.3f}"
        default = self.used.get(camera_length_str, None) 
        if default is None and self.default_val is not None:
            # diff_step scales inversely with camera length
            default = self.default_val * self.default_camera_length / camera_length
        default_str = f"{default:.03f}" if default is not None else ""

        console.print(f"Camera length: {metadata.camera_length * 1e3 :.0f} mm")
        (val_str, val) = prompt_ask(self.name, "mrad/px", default, default_str, console)
        self.used[camera_length_str] = val
        return val


@dataclass(init=False)
class AduParam(Parameter[float]):
    """Maps f"{voltage:.0f}" to ADU values"""
    used: t.Dict[str, float]
    """Voltage associated with default_val (V)"""
    default_voltage: float

    def __init__(self, default: float, default_voltage: float):
        self.name = 'Single-electron intensity'
        self.default_val = default
        self.used = {}
        self.default_voltage = default_voltage

    def ask(self, metadata: EmpadMetadata, console: t.Optional[Console] = None) -> float:
        voltage = metadata.voltage
        voltage_str = f"{voltage:.0f}"
        default = self.used.get(voltage_str, None) 
        if default is None and self.default_val is not None:
            # ADU should scale linearly with voltage
            default = self.default_val * voltage / self.default_voltage
        default_str = f"{default:.0f}" if default is not None else ""

        (val_str, val) = prompt_ask(self.name, "ADU", default, default_str, console)
        self.used[voltage_str] = val
        return val


def process_dir(path, params: t.Optional[t.Sequence[t.Tuple[str, Parameter]]] = None,
                console: t.Optional[Console] = None, prompt: bool = True):
    path = Path(path)

    if console is None:
        console = default_console()
    if params is None:
        params = default_params()

    for raw_file in path.rglob('**/*.xml'):
        raw_dir = raw_file.parent
        try:
            [metadata_path, *extra] = list(filter(lambda f: not f.name.startswith('.'), raw_dir.glob('*.xml')))
        except ValueError:
            continue
        if len(extra):
            console.print(f"[warning]Skipping dir with multiple metadata files: '{raw_dir}'.")
            continue

        output_path = metadata_path.with_suffix('.json')
        if output_path.exists():
            # skip already created
            continue

        try:
            metadata = EmpadMetadata.from_xml(metadata_path)
            console.print(f"Loaded '{metadata_path}'.")
        except UnscannedMetadata:
            console.print(f"[info]Skipping metadata '{metadata_path}' (not a raster scan dataset).")
            continue
        except Exception:
            console.print(f"[error]Couldn't parse XML metadata '{metadata_path}'. Skipping.")
            console.print_exception()
            continue

        if prompt:
            for (k, param) in params:
                val = param.ask(metadata, console)
                setattr(metadata, k, val)

            metadata.notes = Prompt.ask(f"Notes", default=None, console=console)

        with open(output_path, 'w') as f:
            f.write(metadata.json(indent=4, exclude=set('path')))

        console.print(f"Wrote metadata to '{output_path}'.")

    console.print(f"Processed all files.")


@click.command()
@click.option('--watch/--no-watch', default=False, help="Watch folder for changes (Linux only)")
@click.option('--prompt/--no-prompt', default=True, help="Prompt for additional metadata.")
@click.argument('folder', type=click.Path(exists=True, file_okay=False), required=False)
def process_metadata(folder: t.Union[str, Path, None], watch: bool = False, prompt: bool = True):
    """
    Process metadata for all the raw datasets contained in FOLDER.
    """
    console = default_console()
    params = default_params()

    if folder is None:
        folder = Path('.')
    path = Path(folder).resolve()

    console.print(f"Processing dir {path}")
    process_dir(path, params, console, prompt)

    if not watch:
        return

    if not 'linux' in sys.platform:
        console.print(f"[error]--watch is supported on Linux only.")
        raise Exit(1)
    try:
        from inotify.adapters import InotifyTree
        from inotify.constants import IN_CLOSE
    except ImportError:
        console.print(f"[error]Couldn't import inotify.\nInstall it for --watch support.")
        raise Exit(1)

    notifier = InotifyTree(str(path), mask=IN_CLOSE)
    console.print(f"Watching for experiments...")

    gen = notifier.event_gen(yield_nones=False)
    for (_, event_types, _path, filename) in t.cast(t.Iterator[t.Tuple[t.Any, t.List[str], t.Any, str]], gen):
        if not (any(t in event_types for t in ('IN_CLOSE_WRITE', 'IN_CLOSE_NOWRITE'))
                and Path(filename).match('scan*.raw')):
            continue

        print("Found changes, reprocessing.")
        process_dir(path, params, console)


if __name__ == '__main__':
    process_metadata()