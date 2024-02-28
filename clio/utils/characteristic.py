import statistics
from dataclasses import dataclass, field

import numpy as np
import numpy.typing as npt

from clio.utils.indented_file import IndentedFile
from clio.utils.stats import Stats


@dataclass
class Statistic:
    avg: float = 0.0
    min: float = 0.0
    max: float = 0.0
    mode: float = 0.0
    median: float = 0.0
    variance: float = 0.0
    count: int = 0
    std: float = 0.0
    total: float = 0.0
    p10: float = 0.0
    p20: float = 0.0
    p30: float = 0.0
    p40: float = 0.0
    p50: float = 0.0
    p60: float = 0.0
    p70: float = 0.0
    p80: float = 0.0
    p90: float = 0.0
    p95: float = 0.0
    p99: float = 0.0
    p999: float = 0.0
    p100: float = 0.0

    def __init__(self, data: list[float] | npt.ArrayLike | None = None):
        if data is not None:
            if len(data) > 0:  # type: ignore
                self.update(data)

    def update(self, data: list[float] | npt.ArrayLike):
        data = np.array(data)
        self.count = len(data)
        self.total = np.sum(data)
        self.avg = np.mean(data).item()
        self.min = np.min(data).item()
        self.max = np.max(data).item()
        self.mode = statistics.mode(data)
        self.median = np.median(data).item()
        self.variance = np.var(data).item()
        self.std = np.std(data).item()
        for p in [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 100]:
            setattr(self, f"p{p}", np.percentile(data, p).item())
        self.p999 = np.percentile(data, 99.9).item()

    def to_lines(self) -> list[str]:
        return [
            f"avg: {self.avg}",
            f"min: {self.min}",
            f"max: {self.max}",
            f"mode: {self.mode}",
            f"median: {self.median}",
            f"variance: {self.variance}",
            f"std: {self.std}",
            f"total: {self.total}",
            f"p10: {self.p10}",
            f"p20: {self.p20}",
            f"p30: {self.p30}",
            f"p40: {self.p40}",
            f"p50: {self.p50}",
            f"p60: {self.p60}",
            f"p70: {self.p70}",
            f"p80: {self.p80}",
            f"p90: {self.p90}",
            f"p95: {self.p95}",
            f"p99: {self.p99}",
            f"p999: {self.p999}",
            f"p100: {self.p100}",
        ]

    def __str__(self) -> str:
        return "{" + ", ".join(self.to_lines()) + "}"

    def to_indented_file(self, file: IndentedFile):
        for line in self.to_lines():
            file.writeln(line)


@dataclass
class Characteristic:
    disks: set[str] = field(default_factory=set)
    start_ts: int = 0
    end_ts: int = 0
    duration: int = 0
    read_count: int = 0
    write_count: int = 0
    read_size: Statistic = field(default_factory=Statistic)
    write_size: Statistic = field(default_factory=Statistic)
    offset: Statistic = field(default_factory=Statistic)

    @property
    def num_io(self) -> int:
        return self.read_count + self.write_count

    @property
    def size(self) -> float:
        return self.read_size.total + self.write_size.total

    @property
    def num_disks(self) -> float:
        return len(self.disks)

    def __str__(self) -> str:
        stats = [
            f"disks: {self.num_disks} [{', '.join(sorted(self.disks))}]",
            f"num_reads: {self.read_count}",
            f"num_writes: {self.write_count}",
            f"num_io: {self.num_io}",
            f"size_read: {self.read_size}",
            f"size_write: {self.write_size}",
            f"size_total: {self.size}",
            f"offset_total: {self.offset}",
        ]
        return "\n".join(stats)

    def write_stats(self, stats: Stats):
        stats.add_kv("disks", f"{self.num_disks} [{', '.join(sorted(self.disks))}]")
        stats.add_kv("num_reads", self.read_count)
        stats.add_kv("num_writes", self.write_count)
        stats.add_kv("num_io", self.num_io)
        stats.add_kv("size_read", self.read_size)
        stats.add_kv("size_write", self.write_size)
        stats.add_kv("size_total", self.size)
        stats.add_kv("offset_total", self.offset)

    def to_indented_file(self, file: IndentedFile):
        file.writeln("Disks")
        file.inc_indent()
        file.writeln("Count: %d", self.num_disks)
        file.writeln("ID: %s", ", ".join(sorted(self.disks)))
        file.dec_indent()
        file.writeln("IOs")
        file.inc_indent()
        file.writeln("Total: %d", self.num_io)
        file.writeln("Reads: %d", self.read_count)
        file.writeln("Writes: %d", self.write_count)
        file.dec_indent()
        file.writeln("Size")
        file.inc_indent()
        file.writeln("Total: %f", self.size)
        file.writeln("Read")
        file.inc_indent()
        self.read_size.to_indented_file(file)
        file.dec_indent()
        file.writeln("Write")
        file.inc_indent()
        self.write_size.to_indented_file(file)
        file.dec_indent()
        file.writeln("Offset")
        file.inc_indent()
        self.offset.to_indented_file(file)
        file.dec_indent()
        file.dec_indent()


__all__ = ["Characteristic", "Statistic"]
