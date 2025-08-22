from __future__ import annotations

import functools
import sys
import math
import traceback
import typing as t

import click

if t.TYPE_CHECKING:
    P = t.ParamSpec('P')
    U = t.TypeVar('U')

def handle_exception(f: t.Callable[P, U]) -> t.Callable[P, U]:
    """Catch any exception, print a stack trace up to the current frame, and exits."""
    @functools.wraps(f)
    def wrapper(*args: P.args, **kwargs: P.kwargs) -> U:
        try:
            return f(*args, **kwargs)
        except Exception:
            # get parent traceback
            (ty, e, tb) = sys.exc_info()
            # get parent frame
            tb = tb.tb_next  # type: ignore
            traceback.print_exception(ty, e, tb)
            raise click.exceptions.Exit(1)

    return wrapper


def electron_wavelength(kv: float = 200.) -> float:
    """Return the wavelength (in angstroms) of a electron with the given kinetic energy (in keV)."""
    # relativistic total energy (pc) (E^2 = KE^2 + 2*KE*RE)
    rest_energy = 510.99895000  # keV
    hc = 12.3984197  # keV-angstrom
    return hc / math.sqrt(kv**2 + 2*kv*rest_energy)