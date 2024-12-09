import abc

from ..hooks import Hook
from ..state import ReconsState

class Updater(abc.ABC):
    @abc.abstractmethod
    def update_for_group():
        ...

    @abc.abstractmethod
    def update_for_iteration():
        ...


class UpdateHook(Hook[ReconsState, Updater]):
    known = {}