from collections.abc import Callable
from dataclasses import InitVar, dataclass
from pathlib import Path

from clio.utils.logging import log_get

log = log_get(name=__name__)


@dataclass(kw_only=True)
class TraceEntry:
    # following MSFT format
    ts_record: float
    disk_id: str
    offset: int
    io_size: int
    read: bool

    def as_dict(self):
        return {
            "ts_record": self.ts_record,
            "disk_id": self.disk_id,
            "offset": self.offset,
            "io_size": self.io_size,
            "read": self.read,
        }


class TraceWriter:
    def __init__(self, file_path: str | Path, sep: str = " ", write_header: bool = False):
        self.file_path = file_path
        self.sep = sep
        self.write_header = write_header
        self.opened = False
        self.open()

    def __enter__(self):
        self.open()
        return self

    def write(self, entry: TraceEntry):
        entry_l = [entry.ts_record, entry.disk_id, entry.offset, entry.io_size, int(entry.read)]
        self.file.write(self.sep.join(map(str, entry_l)) + "\n")

    def __exit__(self, *args, **kwargs):
        self.close()

    def close(self):
        self.file.close()
        return self

    def open(self):
        if self.opened:
            return self

        self.file = open(self.file_path, "w")
        if self.write_header:
            entry_l = ["ts_record", "disk_id", "offset", "io_size", "read"]
            self.file.write(self.sep.join(entry_l) + "\n")
        self.opened = True

        return self


class TraceReader:
    def __init__(self, file_path: str | Path, sep: str = " "):
        self.file_path = file_path
        self.sep = sep
        self.opened = False
        self.open()

    def __enter__(self):
        self.open()
        return self

    def __iter__(self):
        return self

    def __len__(self):
        return sum(1 for _ in self)

    def __next__(self):
        line = self.file.readline()
        if not line:
            raise StopIteration
        entries = line.strip().split(self.sep)
        assert len(entries) == 5, f'Expected 5 entries ["ts_record", "disk_id", "offset", "io_size", "read"] in 1 row, got {len(entries)}'
        return TraceEntry(
            ts_record=float(entries[0]),
            disk_id=entries[1],
            offset=int(entries[2]),
            io_size=int(entries[3]),
            read=bool(int(entries[4])),
        )

    def iter_filter(self, filter: Callable[[TraceEntry], bool]):
        for entry in self:
            if filter(entry):
                yield entry

    def __exit__(self, *args, **kwargs):
        self.close()

    def close(self):
        self.file.close()
        return self

    def open(self):
        if self.opened:
            return self

        self.file = open(self.file_path, "r")
        if self.file.readline() == "":
            raise ValueError("File is empty")
        self.opened = True

        return self


__all__ = ["TraceEntry", "TraceWriter", "TraceReader"]
