import click
from .process_metadata import process_metadata
from .prepare import prepare

from .calc_tilt import calc_tilt
from .calc_drift import calc_drift
from .view_raw import view_raw
from .view_prepared import view_prepared
from .view_output   import view_output
from .extract_params import extract_params
from .to_csv import to_csv

@click.group()
def tools():
    """Toolbox of utilities."""
    pass

# register subcommands here
tools.add_command(process_metadata)
tools.add_command(prepare)
tools.add_command(calc_tilt)
tools.add_command(calc_drift)
tools.add_command(view_raw)
tools.add_command(view_prepared)
tools.add_command(view_output)
tools.add_command(extract_params)
tools.add_command(to_csv)



# @click.command(cls=MainCommand, commands=dict((v, v) for v in 
#     ('prepare', 'run', 'view_raw', 'view_prepared', 'view_output',
#      'process_metadata', 'extract_params', 'to_csv', 'calc_drift', 'calc_tilt')

