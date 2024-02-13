from dataclasses import dataclass
from pathlib import Path


@dataclass(kw_only=True)
class MSFT_Entry:
    ts_record: float
    disk_id: str
    offset: int
    io_size: float
    write: bool


class MSFT_CSVWriter:
    def __init__(self, file_path: str | Path):
        self.file_path = file_path
        self.opened = False
        self.open()

    def __enter__(self):
        self.open()
        return self

    def write(self, entry: MSFT_Entry):
        self.file.write(f"{entry.ts_record},{entry.disk_id},{entry.offset},{entry.io_size},{entry.write}\n")

    def __exit__(self, *args, **kwargs):
        self.close()

    def close(self):
        self.file.close()
        return self

    def open(self):
        if self.opened:
            return self

        self.file = open(self.file_path, "w")
        self.file.write("ts_record,disk_id,offset,io_size,write\n")
        self.opened = True

        return self


def msft_entries_to_csv(entries: list[MSFT_Entry], file_path: str | Path):
    with MSFT_CSVWriter(file_path) as writer:
        for entry in entries:
            writer.write(entry)


__all__ = ["MSFT_Entry", "msft_entries_to_csv", "MSFT_CSVWriter"]
