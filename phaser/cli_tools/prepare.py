from pathlib import Path
from itertools import chain
from glob import glob as _glob
import sys
import typing as t

import click
import json
import yaml
from pydantic import ValidationError

try:
    from yaml import CLoader as Loader
except ImportError:
    from yaml import Loader

from ptycho_lebeau.params import ParamMetaSet, SaveRecord
from ptycho_lebeau.metadata import Metadata, AnyMetadata
from ptycho_lebeau.util import handle_exception


def _try_parse_metadata(path: Path) -> t.Union[Metadata, ValidationError, None]:
    """
    Try and parse a metadata file. Raises for an invalid file.
    Returns None if `path` might be reconstruction parameters instead.
    """
    with open(path) as f:
        raw_meta = json.load(f)

    try:
        return AnyMetadata.parse_obj(raw_meta, path)
    except ValidationError as e:
        if 'metadata' in raw_meta.get('file_type', '') or 'time_unix' in raw_meta:
            # definitely a metadata file. Raise an exception
            raise
        # we're not sure yet, so return an error we can decide to throw later
        return e



@click.command()
@click.argument('files', type=click.Path(allow_dash=True, dir_okay=False), nargs=-1)
@click.option('--glob/--no-glob', default=True, help="Enable globbing of arguments.")
@click.option('--sparse/--dense', default=None, help="Sparse parameter set.")
@click.option('--out-file', type=click.Path(dir_okay=False), help="Output file to write list of prepared files into")
@handle_exception
def prepare(files: t.Union[t.Sequence[t.Union[str, Path]], str, Path], *, glob: bool = True,
            sparse: t.Optional[bool] = None, out_file: t.Union[str, Path, None] = None):
    """
    Process reconstruction parameters (and acquisition metadata) into PtychoShelves-ready JSON files.
    """

    if isinstance(files, (str, Path)):
        files = (files,)

    param_paths: t.List[Path] = []
    param_sets: t.List[ParamMetaSet] = []
    metadatas: t.List[Metadata] = []

    # process supplied files, sorting them into parameters and metadata
    it = chain.from_iterable(_glob(str(f), recursive=True) for f in files) if glob else files

    def _process_filelist(path: t.Union[str, Path]) -> t.Iterable[Path]:
        path = Path(path)
        if path.suffix.lower() != '.txt':
            yield path
            return
        print(f"Loading filelist '{path}'...")
        with open(path, 'r') as f:
            lines = list(f)

        yield from (Path(line.strip()) for line in lines)

    # .txt files are treated as a list of files to load
    it = chain.from_iterable(map(_process_filelist, it))

    for path in it:
        print(f"Loading '{path}'...")
        meta = None
        # json, might be metadata file
        if path.suffix.lower() == '.json':
            meta = _try_parse_metadata(path)
            if isinstance(meta, Metadata):
                metadatas.append(meta)
                print(f"Loaded '{path}' as '{meta.file_type}'.")
                continue

        with open(path, 'r') as f:
            objs = list(yaml.load_all(f, Loader))
            print(f"Loaded '{path}' as reconstruction(s).")
        try:
            # ugly, fix later
            sets = list(map(ParamMetaSet.parse_obj, objs))
            param_sets.extend(sets)
            param_paths.extend((path.parent,) * len(sets))
        except ValidationError as e:
            if len(objs) == 1 and meta is not None:
                print(f"Could not parse '{path}' as metadata or reconstruction params.", file=sys.stderr)
                print(f"As metadata:\n{meta}", file=sys.stderr)
                print(f"\nAs reconstruction:\n{e}", file=sys.stderr)
            else:
                print(f"Could not parse file '{path}' as reconstruction params:", file=sys.stderr)
                print(e, file=sys.stderr)
            sys.exit(1)

    # keep track of what we've saved
    save_record = SaveRecord()
    saved_paths = []

    if len(param_sets) == 0:
        print("No parameter sets to process.")
    else:
        print(f"Processing {len(param_sets)} parameter set(s).")
    if len(metadatas) > 1:
        print(f"{len(metadatas)} dataset(s) per parameter set")

    for (param_set, path) in zip(param_sets, param_paths):
        if len(metadatas) > 0:
            param_set = param_set.with_metadata(metadatas)

        i = -1
        for (i, reconstruction) in enumerate(param_set.iter(save_record, path=path, sparse=sparse)):
            path = reconstruction.name + '.json'
            with open(path, 'w') as f:
                print(f"Saving reconstruction #{i+1} to '{path}'...")
                f.write(reconstruction.json(indent=4, exclude={'engines': {'__all__': {'slices', 'init_slices'}}}))
                saved_paths.append(path)

        print(f"Saved {i+1} reconstruction(s)")

    if out_file is not None:
        print(f"Writing list of reconstructions to '{out_file}'")
        with open(out_file, 'w') as f:
            for path in saved_paths:
                print(path, file=f)


if __name__ == '__main__':
    prepare()
