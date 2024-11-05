from pathlib import Path
import typing as t

import click

from .plan import ReconsPlan
from .execute import execute_plan


@click.group()
def cli():
    pass


@cli.command('run')
@click.argument('path', type=click.Path(exists=True, dir_okay=False))
def run(path: t.Union[str, Path]):
    plans = ReconsPlan.from_yaml_all(path)

    for plan in plans:
        execute_plan(plan)


@cli.command('serve')
@click.argument('path', type=click.Path(exists=True, dir_okay=False))
def serve(path: t.Union[str, Path]):
    from phaser.web.server import server

    plan = ReconsPlan.from_yaml(path)
    server.run(plan)


@cli.command('worker')
@click.argument('url', type=str, required=True)
def worker(url: str):
    from phaser.web.worker import run_worker

    run_worker(url)


if __name__ == '__main__':
    cli()