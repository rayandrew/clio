import csv
import gzip
from itertools import dropwhile, takewhile
from pathlib import Path
from typing import Callable, Iterable


def wrap_gen(gen, *others):
    for i in gen:
        yield i, *others


def _read_csv(buffer: Iterable[str], contains_header: bool = True, filter: Callable[[list[str]], bool] | None = None):
    if not filter:
        filter = lambda _: True

    reader = csv.reader(buffer)
    if contains_header:
        yield next(reader), True

    yield from takewhile(lambda x: filter(x[0]), dropwhile(lambda x: not filter(x[0]), wrap_gen(reader, False)))


def read_csv(file: Path | str, contains_header: bool = True, filter: Callable[[list[str]], bool] | None = None):
    with open(file, "r") as f:
        yield from _read_csv(f, contains_header, filter)


def read_csv_gz(file: Path | str, contains_header: bool = True, filter: Callable[[list[str]], bool] | None = None):
    with gzip.open(file, "rt") as f:
        yield from _read_csv(f, contains_header, filter)


__all__ = ["read_csv", "read_csv_gz"]
