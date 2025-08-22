import click
from .process_metadata import process_metadata

@click.group()
def tools():
    """Toolbox of utilities."""
    pass

# register subcommands here
tools.add_command(process_metadata)