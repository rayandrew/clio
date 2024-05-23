import typer as base_typer

from clio.utils.timer import timeit


class Typer(base_typer.Typer):
    @timeit
    def __call__(self, *args, **kwargs):
        res = super().__call__(*args, **kwargs)
        return res


typer = base_typer


__all__ = ["Typer", "typer"]
