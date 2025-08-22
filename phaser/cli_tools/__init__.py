import click
from .process_metadata import process_metadata
from .prepare import prepare

from .calc_tilt import calc_tilt
from .calc_drift import calc_drift

@click.group()
def tools():
    """Toolbox of utilities."""
    pass

# register subcommands here
tools.add_command(process_metadata)
tools.add_command(prepare)
tools.add_command(calc_tilt)
tools.add_command(calc_drift)
