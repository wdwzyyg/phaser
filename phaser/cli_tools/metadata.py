from __future__ import annotations

import abc
from datetime import datetime, timedelta
from pathlib import Path
import csv
import typing as t

from pydantic import BaseModel, Field, validator, root_validator
import numpy
from lxml import etree


def parse_version(s: str) -> t.Tuple[int, ...]:
    def to_int(seg: str) -> int:
        if not seg.isdigit():
            raise ValueError()
        return int(seg)

    try:
        return tuple(map(to_int, s.split('.')))
    except ValueError:
        raise ValueError(f"Invalid version string '{s}'") from None


def _convert_to_si(cls, values: t.Dict[str, t.Any]) -> t.Dict[str, t.Any]:
    """
    Convert units from old metadata format to new metadata format (all SI units).
    """

    if values.get('defocus') is not None:
        values['defocus'] *= 1e-9         # nm to m

    values['scan_step'] = tuple(v * 1e-10 for v in values['scan_step'])  # A to m

    if values.get('beam_current') is not None:
        values['beam_current'] *= 1e-12   # pA to A

    if values.get('exposure_time') is not None:
        values['exposure_time'] *= 1e-3        # ms to s
    if values.get('post_exposure_time') is not None:
        values['post_exposure_time'] *= 1e-3   # ms to s

    # update version
    values['version'] = "2.0"

    return values



if not t.TYPE_CHECKING:
    class Metadata(abc.ABC):
        ...
else:
    class Metadata(abc.ABC):
        @property
        @abc.abstractmethod
        def file_type(self) -> str:
            ...

        @property
        @abc.abstractmethod
        def name(self) -> str:
            ...

        @property
        @abc.abstractmethod
        def path(self) -> t.Optional[Path]:
            ...

        @property
        @abc.abstractmethod
        def voltage(self) -> float:
            ...

        @property
        @abc.abstractmethod
        def conv_angle(self) -> float:
            ...

        @property
        @abc.abstractmethod
        def defocus(self) -> float:
            ...

        @property
        @abc.abstractmethod
        def diff_step(self) -> float:
            ...

        @property
        @abc.abstractmethod
        def scan_rotation(self) -> float:
            ...

        @property
        @abc.abstractmethod
        def scan_shape(self) -> t.Tuple[int, int]:
            ...

        @property
        @abc.abstractmethod
        def scan_step(self) -> t.Tuple[float, float]:
            ...

        @property
        @abc.abstractmethod
        def scan_correction(self) -> t.Optional[t.Tuple[t.Tuple[float, float], t.Tuple[float, float]]]:
            ...

        @property
        @abc.abstractmethod
        def scan_positions(self) -> t.Optional[t.List[t.Tuple[float, float]]]:
            ...

        @property
        @abc.abstractmethod
        def raw_filename(self) -> t.Optional[str]:
            ...

        @property
        @abc.abstractmethod
        def crop(self) -> t.Optional[t.Tuple[int, int, int, int]]:
            ...

        @abc.abstractmethod
        def is_simulated(self) -> bool:
            ...


class PyMultislicerMetadata(BaseModel, Metadata):
    file_type: t.Literal['pyMultislicer_metadata'] = 'pyMultislicer_metadata'

    name: str
    """Experiment name."""
    version: str = "2.0"
    """Metadata version"""

    @root_validator(pre=False)
    def _validate_version(cls, values):
        version = parse_version(values['version'])
        if version > (2, 0):
            raise ValueError(f"Unsupported metadata version '{values['version']}'")
        if version < (2, 0):
            return _convert_to_si(cls, values)
        return values

    path: t.Optional[Path] = Field(default=None, exclude=True)
    """Current path to experimental folder."""

    @classmethod
    def parse_obj(cls, obj: t.Any, path: t.Union[str, Path, None] = None) -> 'PyMultislicerMetadata':
        # ugly hack
        meta: PyMultislicerMetadata = BaseModel.parse_obj.__func__(cls, obj)  # type: ignore
        meta.path = Path(path).parent if path is not None else None
        return meta

    @classmethod
    def parse_file(cls, path: t.Union[str, Path], *, content_type: t.Optional[str] = None,
                   encoding: str = 'utf8', proto: t.Optional[str] = None, allow_pickle: bool = False) -> PyMultislicerMetadata:
        # ugly hack
        meta: PyMultislicerMetadata = BaseModel.parse_file.__func__(cls, path, content_type=content_type, encoding=encoding,  # type: ignore
                                                                    proto=proto, allow_pickle=allow_pickle)
        meta.path = Path(path).parent
        return meta

    raw_filename: t.Optional[str]
    """Raw 4DSTEM data filename."""

    voltage: float
    """Accelerating voltage (V)."""

    conv_angle: t.Optional[float] = None
    """Convergence angle (mrad)."""
    defocus: t.Optional[float] = None
    """Defocus (m). Positive is overfocus."""
    diff_step: t.Optional[float] = None
    """Diffraction pixel size (mrad/px)."""

    scan_rotation: float
    """Scan rotation (degrees)."""
    scan_shape: t.Tuple[int, int]
    """Scan shape (x, y)."""
    scan_fov: t.Tuple[float, float]
    """Scan field of view (m)."""
    scan_step: t.Tuple[float, float]
    """Scan step (m/px)."""

    scan_correction: t.Optional[t.Tuple[t.Tuple[float, float], t.Tuple[float, float]]] = None
    """Scan correction matrix, [x', y'] = scan_correction @ [x, y]"""

    scan_positions: t.Optional[t.List[t.Tuple[float, float]]] = None
    """
    Scan position override (m).
    Should be specified as a 1d list of (x, y) positions, in scan order. `scan_correction` is applied to these positions (if present).
    """

    def is_simulated(self) -> t.Literal[True]:
        return True

    @property
    def crop(self) -> None:
        """Region scan is valid within, (min_x, max_x, min_y, max_y). Matlab-style slicing (1-indexed, inclusive)."""
        return None


class UnscannedMetadata(BaseException):
    ...


class EmpadMetadata(Metadata, BaseModel):
    file_type: t.Literal['empad_metadata'] = 'empad_metadata'

    name: str
    """Experiment name."""
    version: str = "2.0"
    """Metadata version"""

    @root_validator(pre=False)
    def _validate_version(cls, values):
        version = parse_version(values['version'])
        if version > (2, 0):
            raise ValueError(f"Unsupported metadata version '{values['version']}'")
        if version < (2, 0):
            return _convert_to_si(cls, values)
        return values

    raw_filename: t.Optional[str]
    """Raw 4DSTEM data filename."""
    orig_path: t.Optional[Path] = None
    """Original path to experimental folder."""
    path: t.Optional[Path] = Field(default=None, exclude=True)
    """Current path to experimental folder."""
    author: t.Optional[str] = None
    """Dataset author"""
    time: str
    """Acquisition time, formatted according to ISO 8061."""
    time_unix: float
    """Acquisition time, in Unix time (seconds since epoch)."""

    def get_datetime(self) -> datetime:
        return datetime.fromtimestamp(self.time_unix)

    bg_unix: t.Optional[float] = None
    """Background acquisition time, in Unix time (seconds since epoch)."""

    has_bg: bool = False
    """True if the background acquisition is <6 hours out of date."""

    voltage: float
    """Accelerating voltage (V)."""
    conv_angle: t.Optional[float] = None
    """Convergence angle (mrad)."""
    defocus: t.Optional[float] = None
    """Defocus (m). Positive is overfocus."""

    camera_length: float
    """Camera length (m)."""
    diff_step: t.Optional[float] = None
    """Est. diffraction pixel size (mrad/px)."""

    scan_rotation: float
    """Scan rotation (degrees)."""
    scan_shape: t.Tuple[int, int]
    """Scan shape (x, y)."""
    scan_fov: t.Tuple[float, float]
    """Scan field of view (m)."""
    scan_step: t.Tuple[float, float]
    """Scan step (Angstrom/px)."""

    exposure_time: float
    """Pixel exposure time (s)."""
    post_exposure_time: float
    """Pixel post-exposure time (s)."""
    beam_current: t.Optional[float] = None
    """Approx. beam current (A)."""
    adu: t.Optional[float] = None
    """Single-electron intensity (data units)."""

    scan_correction: t.Optional[t.Tuple[t.Tuple[float, float], t.Tuple[float, float]]] = None
    """Scan correction matrix, [x', y'] = scan_correction @ [x, y]"""

    scan_positions: t.Optional[t.List[t.Tuple[float, float]]] = None
    """
    Scan position override (m).
    Should be specified as a 1d list of (x, y) positions, in scan order. `scan_correction` is applied to these positions (if present).
    """

    notes: t.Optional[str] = None

    crop: t.Optional[t.Tuple[int, int, int, int]] = None
    """Region scan is valid within, (min_x, max_x, min_y, max_y). Matlab-style slicing (1-indexed, inclusive)."""

    def is_simulated(self) -> t.Literal[False]:
        return False

    @classmethod
    def parse_obj(cls, obj: t.Any, path: t.Union[str, Path, None] = None) -> 'EmpadMetadata':
        # ugly hack
        meta: EmpadMetadata = BaseModel.parse_obj.__func__(cls, obj)  # type: ignore
        meta.path = Path(path).parent if path is not None else None
        return meta

    @classmethod
    def parse_file(cls, path: t.Union[str, Path], *, content_type: t.Optional[str] = None,
                   encoding: str = 'utf8', proto: t.Optional[str] = None, allow_pickle: bool = False) -> EmpadMetadata:
        # ugly hack
        meta: EmpadMetadata = BaseModel.parse_file.__func__(cls, path, content_type=content_type, encoding=encoding,  # type: ignore
                                                            proto=proto, allow_pickle=allow_pickle)
        meta.path = Path(path).parent
        return meta

    class Config:
        allow_population_by_field_name = True
        extra = 'forbid'

        json_encoders = {
            # encode empty paths as empty string (for passing through to matlab)
            Path: lambda p: "" if p == Path("") else str(p)
        }

    @staticmethod
    def from_xml(xml_path: t.Union[str, Path]) -> EmpadMetadata:
        orig_path = Path(xml_path).parent
        xml = t.cast(etree._ElementTree, etree.parse(str(xml_path)))  # type: ignore

        def get(root, tag: str) -> etree._Element:
            elem = root.find(tag)  # type: ignore
            if elem is None:
                raise ValueError(f"Couldn't find tag '{tag}' in XML metadata.")
            return elem

        def try_get(root, tag: str) -> t.Optional[etree._Element]:
            return root.find(tag)  # type: ignore

        root: etree._Element = xml.getroot()
        timestamp = get(root, 'timestamp')
        raw_filename = get(root, 'raw_file').attrib['filename']
        time_unix = float(timestamp.attrib['timestamp'])

        scan_params = get(root, 'scan_parameters')
        if try_get(scan_params, 'series_count') is not None:
            # not a scanned dataset
            raise UnscannedMetadata()

        background = try_get(root, 'background_image')
        exposure_time = float(get(root, 'exposure_time').text) * 1e-3            # ms to s
        post_exposure_time = float(get(root, 'post_exposure_time').text) * 1e-3  # ms to s

        bg_unix = None if background is None else background.attrib.get('timestamp', None)
        bg_unix = None if bg_unix is None else float(bg_unix)
        # True if background is <6 hours out of date
        has_bg = False if bg_unix is None else abs(time_unix - bg_unix) < 60*60*12

        iom = get(root, 'iom_measurements')
        scan_rotation = float(get(iom, 'scan_rotation').text) * 180. / numpy.pi
        camera_length = float(get(iom, 'nominal_camera_length').text)  # m
        voltage = float(get(iom, 'high_voltage').text)                 # V

        scan_params = get(root, "scan_parameters[@mode='acquire']")
        scan_shape = (get(scan_params, 'scan_resolution_x'), get(scan_params, 'scan_resolution_y'))
        scan_shape = tuple(map(lambda elem: int(elem.text), scan_shape))
        scan_size = float(get(scan_params, 'scan_size').text)

        if abs(scan_size - 1.) > 1e-5:
            print("Warning: scan_size != 1 has not been tested.")

        # all in m
        fov = get(iom, 'full_scan_field_of_view')
        scale_factor = float(get(fov, 'scale_factor').text)
        fov = tuple(map(lambda elem: float(elem.text) * scan_size / scale_factor, (get(fov, 'x'), get(fov, 'y'))))
        scan_step = tuple(map(lambda v: v / max(scan_shape), fov))

        return EmpadMetadata(
            name=root.attrib['name'],
            orig_path=orig_path,
            raw_filename=raw_filename,
            time=timestamp.attrib['isoformat'],
            time_unix=time_unix,
            bg_unix= bg_unix,
            has_bg=has_bg,
            voltage=voltage,
            camera_length=camera_length,
            scan_rotation=scan_rotation,
            scan_shape=t.cast(t.Tuple[int, int], scan_shape),
            scan_fov=t.cast(t.Tuple[float, float], fov),
            scan_step=t.cast(t.Tuple[float, float], scan_step),
            exposure_time=exposure_time,
            post_exposure_time=post_exposure_time,
        )


class AnyMetadata(BaseModel, Metadata):
    __root__: t.Union[EmpadMetadata, PyMultislicerMetadata] = Field(discriminator='file_type')

    @validator('__root__', pre=True)
    def _default_keys(cls, value: t.Any) -> t.Any:
        # hack to specify default file_type
        if isinstance(value, dict) and 'file_type' not in value:
            value['file_type'] = 'empad_metadata'
        # when parsing from a file, if version not found, assume it's 0.1
        if isinstance(value, dict) and 'version' not in value:
            value['version'] = '0.1'
        return value

    @classmethod
    def parse_file(cls, path: t.Union[str, Path], *, content_type: t.Optional[str] = None,
                   encoding: str = 'utf8', proto: t.Optional[str] = None, allow_pickle: bool = False) -> AnyMetadata:
        # ugly hack
        meta: AnyMetadata = BaseModel.parse_file.__func__(cls, path, content_type=content_type, encoding=encoding,  # type: ignore
                                                          proto=proto, allow_pickle=allow_pickle)
        meta.__root__.path = Path(path).parent
        return meta

    @classmethod
    def parse_obj(cls, obj: t.Any, path: t.Union[str, Path, None] = None) -> 'Metadata':
        # ugly hack
        meta: AnyMetadata = BaseModel.parse_obj.__func__(cls, obj)  # type: ignore
        meta.path = Path(path).parent if path is not None else None  # type: ignore
        return meta

    def __getattr__(self, name):
        return getattr(self.__root__, name)

    def __setattr__(self, name, value):
        return setattr(self.__root__, name, value)


def datetime_to_excel(date: datetime) -> t.Tuple[int, float]:
    offset = date - datetime(1900, 1, 1, 0, 0, 0, 0)

    # split into days and time
    days, rem = divmod(offset, timedelta(days=1))
    # return number of days and fraction of day
    return (days, rem / timedelta(days=1))


_CSV_FMTS: t.Sequence[t.Tuple[str, t.Union[str, t.Callable[[EmpadMetadata], t.Any]]]] = [
    ('Date', lambda meta: datetime_to_excel(meta.get_datetime())[0]),
    ('Time', lambda meta: datetime_to_excel(meta.get_datetime())[1]),
    ('Name', 'name'),
    ('Material', ''),
    ('Taken By', 'author'),
    ('Has bg', 'has_bg'),
    ('Voltage (kV)', lambda meta: meta.voltage / 1e3),  #f"{meta.voltage / 1e3:.1f}"),
    ('Scan size x (px)', lambda meta: meta.scan_shape[0]),
    ('Scan size y (px)', lambda meta: meta.scan_shape[1]),
    ('Scan FOV x (m)', lambda meta: meta.scan_fov[0]),
    ('Scan FOV y (m)', lambda meta: meta.scan_fov[1]),
    ('Scan step x (A/px)', lambda meta: meta.scan_step[0]*1e10),  # m to A
    ('Scan step y (A/px)', lambda meta: meta.scan_step[1]*1e10),  # m to A
    ('Scan rot (deg)', 'scan_rotation'),
    ('Mag (x)', ''),
    ('Overfocus (CW, nm)', lambda meta: meta.defocus*1e9 if meta.defocus is not None else ""),  # m to nm
    ('Conv. (mrad)', 'conv_angle'),
    ('Camera length (mm)', lambda meta: meta.camera_length * 1e3),
    ('Diff. pixel size (mrad/px)', 'diff_step'),
    ('Path', lambda meta: str(meta.path.absolute()) if meta.path is not None else ""),
    ('Notes', 'notes'),
]

def to_csv(path: t.Union[str, Path, t.TextIO], metadata: t.Iterable[EmpadMetadata]):
    if isinstance(path, (str, Path)):
        close = True
        f = open(path, 'w', encoding='utf-8')
    else:
        close = False
        f = path

    try:
        writer = csv.writer(f, dialect='excel', delimiter=',', quoting=csv.QUOTE_NONNUMERIC)
        writer.writerow(name for (name, fmt) in _CSV_FMTS)
        writer.writerows((
                getattr(meta, fmt, "") if isinstance(fmt, str) else fmt(meta)
                for (name, fmt) in _CSV_FMTS
            ) for meta in metadata
        )
    finally:
        if close:
            f.close()


# out object key, column to search for, parse
_CSV_PARSE_COLS: t.Dict[str, t.Union[str, t.Tuple[str, t.Callable[[str], t.Any]]]] = {
    'name': 'name',
}


def from_csv(f: t.TextIO) -> t.Sequence[Metadata]:
    raise NotImplementedError()
