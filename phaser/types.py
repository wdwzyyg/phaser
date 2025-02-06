import typing as t

import numpy
import pane

if t.TYPE_CHECKING:
    from phaser.hooks import FlagArgs


BackendName: t.TypeAlias = t.Literal['cuda', 'cupy', 'jax', 'cpu', 'numpy']


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


class Flag(Dataclass):
    after: int = 0
    every: int = 1
    before: t.Optional[int] = None

    def __call__(self, args: 'FlagArgs') -> bool:
        i = args['state'].iter.engine_iter
        return (
            (i < self.before if self.before is not None else True)
            and i > self.after
            and (i - self.after) % self.every == 0
        )

    def resolve(self) -> t.Self:
        return self


__all__ = [
    'BackendName', 'Dataclass', 'Slices', 'Flag',
]