"""
Utilities for configuration
"""

from pathlib import Path
import functools
import io
import logging
import textwrap
import typing as t

import pane
from pane.field import _MISSING  # bad

PaneClassT = t.TypeVar('PaneClassT', bound=pane.PaneBase)


@functools.cache
def get_config_dir() -> Path:
    import platformdirs
    return platformdirs.user_config_path(
        'phaser', roaming=True, use_site_for_root=True,
    )


def get_class_docstrings(cls: type) -> t.Dict[str, str]:
    """Extract attribute docstrings from class `cls`"""
    # TODO: can probably replace this with griffe or something
    import ast
    import inspect

    try:
        source = inspect.getsource(cls)
    except OSError:
        return {}

    classdef = t.cast(ast.ClassDef, ast.parse(source).body[0])
    assert classdef.name == cls.__name__

    d: t.Dict[str, str] = {}
    last_field: t.Optional[str] = None

    for stmt in classdef.body:
        if isinstance(stmt, ast.AnnAssign) and stmt.simple:
            last_field = t.cast(ast.Name, stmt.target).id
            continue
        if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Constant):
            const = stmt.value.value
            if isinstance(const, str) and last_field:
                d[last_field] = const
        last_field = None
    return d


def _format_type(ty: t.Any) -> str:
    from types import UnionType

    origin = t.get_origin(ty)

    if origin is None:
        if ty is type(None) or ty is None:
            return "None"
        if isinstance(ty, type):
            return ty.__name__
        if isinstance(ty, str):
            return ty
        return repr(ty)

    args = t.get_args(ty)

    if origin is UnionType or origin is t.Union:
        return ' | '.join(map(_format_type, args))

    args = ', '.join(map(_format_type, args))
    return f'{_format_type(origin)}[{args}]'


class Config(t.Generic[PaneClassT]):
    def __init__(self, name: str, ty: t.Type[PaneClassT]):
        self.config_name = name
        self.ty = ty

        if not issubclass(ty, pane.PaneBase):
            raise TypeError(f"Config type '{self.ty.__name__}' must be a pane class.")

    def path(self) -> Path:
        return get_config_dir() / f'{self.config_name}.yaml'

    def default(self) -> PaneClassT:
        try:
            return pane.from_data({}, self.ty)
        except pane.ConvertError as e:
            raise TypeError(
                f"Config type '{self.ty.__name__}' must be default constructible."
                " This is a bug with phaser."
            ) from e

    def get(self) -> PaneClassT:
        logger = logging.getLogger()
        path = self.path()
        logger.info(f"Configuration path: '{path}'")

        if not path.exists():
            logger.info("Configuration file not found, using default")
            return self.default()

        try:
            config = pane.from_yaml_all(path, self.ty)
        except pane.ConvertError as e:
            e.add_note("Invalid configuration file")
            raise
        except Exception as e:
            e.add_note("Failed to read configuration file")
            raise
        if not len(config):
            return self.default()
        if len(config) > 1:
            raise ValueError("Invalid configuration file (multiple YAML documents)")
        return config[0]

    def write_default(self) -> bool:
        """
        Write a default configuration file.

        Does nothing and returns `False` if the file already exists
        """
        logger = logging.getLogger()
        path = self.path()
        if path.exists():
            return False
        logger.info(f"Creating default config file at '{path}'")

        import yaml
        try:
            from yaml import CSafeDumper as Dumper
        except ImportError:
            from yaml import SafeDumper as Dumper

        buf = io.StringIO('\n')

        docstrings = get_class_docstrings(self.ty)

        for field in self.ty.__pane_info__.fields:
            if not field.init or not field.has_default():
                continue

            if field.default is _MISSING:
                if field.default_factory is None:
                    continue
                default = field.default_factory()
            else:
                default = field.default

            # write expression
            expr = t.cast(str, yaml.dump(
                {field.in_names[0]: default},
                Dumper=Dumper, explicit_start=False,
                allow_unicode=True, default_flow_style=None,
            )).strip('\n')
            if expr[0] == '{' and expr[-1] == '}':
                expr = expr[1:-1]
            buf.write(expr)
            buf.write('\n')
            # and docstring
            if (docstring := docstrings.get(field.name)):
                buf.write(textwrap.dedent(docstring).strip('\n'))
                buf.write('\n')
            # and type
            buf.write('type: ' + _format_type(field.type))
            buf.write('\n\n')

        buf.seek(0)
        try:
            path.parent.mkdir(parents=False, exist_ok=True)
            with open(path, 'w') as f:
                f.writelines(
                    '# ' + line if line.strip() else '\n'
                    for line in buf
                )
        except Exception as e:
            e.add_note("Failed to write config file")
            raise
        return True


__all__ = [
    'get_config_dir', 'Config',
]