import typing as t

import click


class MainCommand(click.MultiCommand):
    def __init__(self, commands: t.Union[t.Iterable[str], t.Dict[str, t.Union[str, t.Tuple[str, str]]]], **kwargs):
        super().__init__(**kwargs)
        self.commands: t.Dict[str, t.Union[str, t.Tuple[str, str]]]
        if isinstance(commands, dict):
            self.commands = commands
        else:
            self.commands = dict((v, v) for v in commands)

    def list_commands(self, ctx: click.Context):
        return list(self.commands.keys())

    def get_command(self, ctx: click.Context, cmd_name: str) -> t.Optional[click.Command]:
        name = cmd_name.lower()
        val = (self.commands.get(name) or 
               self.commands.get(name.replace('-', '_')))
        if val is None:
            return None
        if isinstance(val, tuple):
            (module, func) = val
        else:
            module = val
            func = val
        mod = __import__(f"{__package__}.{module}", None, None, [func])
        return getattr(mod, func)


@click.command(cls=MainCommand, commands=dict((v, v) for v in 
    ('prepare', 'run', 'view_raw', 'view_prepared', 'view_output',
     'process_metadata', 'extract_params', 'to_csv', 'calc_drift', 'calc_tilt')
))
def main():
    """LeBeau group ptychography utilities."""
    ...


if __name__ == '__main__':
    main()
