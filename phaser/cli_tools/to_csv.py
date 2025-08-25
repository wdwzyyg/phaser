
import sys
from pathlib import Path
import typing as t

import click

from .metadata import EmpadMetadata
from .metadata import to_csv as _to_csv


@click.command()
@click.argument('path', type=click.Path(), nargs=-1, required=False)
@click.option('--out', '-o', type=click.Path(allow_dash=True))
def to_csv(path: t.Union[str, Path, t.Sequence[t.Union[str, Path]]], out: t.Union[str, Path, None] = None):
    """
    Export ptychography metadata to a CSV file.

    `path` may either be a list of directories to search within, or a list of metadata JSON files.
    If unspecified, the current directory will be searched.

    The resulting CSV file is written to the `out` file, or to stdout.
    """
    if isinstance(path, (str, Path)):
        paths = [path]
    elif path is None or len(path) == 0:
        paths = ['']
    else:
        paths = path

    def resolve_paths(paths: t.Iterable[t.Union[str, Path]]) -> t.Iterator[Path]:
        for path in map(Path, paths):
            if path.is_dir():
                yield from path.rglob('*.json')
            else:
                yield path

    def parse_metadata(paths: t.Iterable[Path]) -> t.Iterator[EmpadMetadata]:
        for path in paths:
            print(f"Parsing '{path}'", file=sys.stderr)
            try:
                yield EmpadMetadata.parse_file(path)
            except Exception as e:
                print(f"Couldn't parse '{path}', may not be a metadata file. Skipping.", file=sys.stderr)

    f = sys.stdout if out in (None, '-') else out
    _to_csv(f, parse_metadata(resolve_paths(paths)))
