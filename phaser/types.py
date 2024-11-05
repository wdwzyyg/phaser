import typing as t

import numpy
import pane


class Cancelled(BaseException):
    ...


class Dataclass(pane.PaneBase, kw_only=True, allow_extra=False):
    ...


class SliceList(Dataclass):
    zs: t.List[float]

class SliceStep(Dataclass):
    n: int
    slice_thickness: float

    @property
    def zs(self) -> t.List[float]:
        return [float(z) for z in numpy.arange(self.n) * self.slice_thickness]

class SliceTotal(Dataclass):
    n: int
    total_thickness: float

    @property
    def zs(self) -> t.List[float]:
        return [float(z) for z in numpy.arange(self.n) * self.total_thickness / self.n]

Slices: t.TypeAlias = t.Union[SliceList, SliceStep, SliceTotal]


__all__ = [
    'Dataclass',
]