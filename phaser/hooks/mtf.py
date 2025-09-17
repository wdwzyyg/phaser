from pathlib import Path
import typing as t

import numpy
from numpy.typing import NDArray, DTypeLike
import pane.annotations as annotations
from typing_extensions import NotRequired

from ..types import Dataclass, Slices
from .hook import Hook

if t.TYPE_CHECKING:
    from phaser.utils.num import Sampling
    from phaser.utils.object import ObjectSampling
    from ..state import ObjectState, ProbeState, ReconsState, Patterns
    from ..execute import Observer



class LoadDetectorMtfProps(Dataclass):
    path: Path
    bin: t.Optional[int] = 1

class DetectorMtf(t.TypedDict):
    inverse_pixel: NDArray[numpy.floating]
    values: NDArray[numpy.floating]

class DetectorMtfHook(Hook[None, DetectorMtf]):
    known = {
        'custom': ('phaser.hooks.io.mtf:load_mtf', LoadDetectorMtfProps, ('starfile',)),
    }