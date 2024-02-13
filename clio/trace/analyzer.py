from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import typer

from clio.utils.csv import read_csv_gz
from clio.utils.logging import log_global_setup
from clio.utils.trace import TraceEntry, TraceReader

app = typer.Typer(name="Analyzer")


@app.command()
def quick(
    file: Annotated[Path, "The file to analyze"],
):
    """
    Quickly analyze a file.

    :param file (Path): The file to analyze
    """

    log = log_global_setup()
    log.info("Analyzing %s", file, tab=0)

    # Read the file
    data = TraceReader(file)
    stats = TraceStatistic()

    for entry in data:
        stats.disks.add(entry.disk_id)
        if entry.read:
            stats.read_count += 1
            stats.read_size += entry.io_size
        else:
            stats.write_count += 1
            stats.write_size += entry.io_size
        stats.offset += entry.offset

    log.info("Statistics", tab=0)
    log.info("%s", stats)


@dataclass
class TraceStatistic:
    disks: set[str] = field(default_factory=set)
    read_count: int = 0
    write_count: int = 0
    read_size: int = 0
    write_size: int = 0
    offset: int = 0

    @property
    def total_count(self) -> int:
        return self.read_count + self.write_count

    @property
    def total_size(self) -> int:
        return self.read_size + self.write_size

    @property
    def avg_size(self) -> float:
        return self.total_size / self.total_count

    @property
    def num_disks(self) -> float:
        return len(self.disks)

    @property
    def avg_offset(self) -> float:
        return self.offset / self.total_count

    def __str__(self) -> str:
        stats = [
            f"disks: {self.num_disks} [{', '.join(self.disks)}]",
            f"num_reads: {self.read_count}",
            f"num_writes: {self.write_count}",
            f"num_io: {self.total_count}",
            f"size_read: {self.read_size}",
            f"size_write: {self.write_size}",
            f"size_total: {self.total_size}",
            f"size_avg: {self.avg_size}",
            f"offset_total: {self.offset}",
            f"offset_avg: {self.avg_offset}",
        ]
        return "\n".join(stats)


@app.command()
def full(
    file: Annotated[Path, "The file to analyze"],
    output: Annotated[Path, typer.Option(help="The output path to write the results to")],
):
    """
    Fully analyze a file.

    :param file (Path): The file to analyze
    :param output (Path): The output path to write the results to
    """
    output.mkdir(parents=True, exist_ok=True)
    log = log_global_setup(output / "log.txt")
    log.info("Analyzing %s", file, tab=0)

    # Read the file
    data = TraceReader(file)
    stats = TraceStatistic()

    for entry in data:
        stats.disks.add(entry.disk_id)
        if entry.read:
            stats.read_count += 1
            stats.read_size += entry.io_size
        else:
            stats.write_count += 1
            stats.write_size += entry.io_size
        stats.offset += entry.offset

    log.info("Statistics", tab=0)
    log.info("%s", stats)


if __name__ == "__main__":
    app()
