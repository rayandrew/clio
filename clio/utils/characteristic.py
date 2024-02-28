import io
import statistics
from collections import UserList
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import numpy.typing as npt
from serde import serde
from serde.msgpack import from_msgpack, to_msgpack

from clio.utils.indented_file import IndentedFile


@serde
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

    @staticmethod
    def generate(data: list[float] | npt.ArrayLike | None = None) -> "Statistic":
        self = Statistic()
        if data is not None:
            self.update(data)
        return self

    def update(self, data: list[float] | npt.ArrayLike | None = None):
        if data is None or len(data) == 0:  # type: ignore
            return

        data = np.array(data)
        self.count = len(data)
        self.total = float(np.sum(data))
        self.avg = float(np.mean(data).item())
        self.min = float(np.min(data).item())
        self.max = float(np.max(data).item())
        self.mode = float(statistics.mode(data))
        self.median = float(np.median(data).item())
        self.variance = float(np.var(data).item())
        self.std = float(np.std(data).item())
        for p in [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99, 100]:
            setattr(self, f"p{p}", float(np.percentile(data, p).item()))
        self.p999 = float(np.percentile(data, 99.9).item())

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

    def from_indented_file(self, file_or_path_or_str: io.TextIOWrapper | str | Path):
        if isinstance(file_or_path_or_str, (str, Path)):
            with open(file_or_path_or_str, "r") as file:
                self.from_indented_file(file)
            return

        with file_or_path_or_str as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                key, value = line.split(":")
                key = key.strip()
                value = value.strip()
                setattr(self, key, float(value))

    def from_str(self, data: str):
        for line in data.split("\n"):
            line = line.strip()
            if not line:
                continue
            key, value = line.split(":")
            key = key.strip()
            value = value.strip()
            setattr(self, key, float(value))

    def to_msgpack(self, path: Path | str | io.BufferedIOBase):
        if isinstance(path, (str, Path)):
            with open(path, "wb") as file:
                file.write(to_msgpack(self))
            return

        path.write(to_msgpack(self))

    def from_msgpack(self, data: bytes):
        self = from_msgpack(Statistic, data)


@serde
@dataclass
class Characteristic:
    disks: set[str] = field(default_factory=set)
    start_ts: int = 0
    end_ts: int = 0
    ts_unit: str = "ms"
    size_unit: str = "B"
    duration: int = 0
    read_count: int = 0
    write_count: int = 0
    iat: Statistic = field(default_factory=Statistic)
    read_size: Statistic = field(default_factory=Statistic)
    write_size: Statistic = field(default_factory=Statistic)
    offset: Statistic = field(default_factory=Statistic)

    @property
    def num_io(self) -> int:
        return self.read_count + self.write_count

    @property
    def size(self) -> float:
        return float(self.read_size.total + self.write_size.total)

    @property
    def read_ratio(self) -> float:
        return self.read_count / self.num_io

    @property
    def write_ratio(self) -> float:
        return self.write_count / self.num_io

    @property
    def rw_ratio(self) -> float:
        return self.read_count / self.write_count

    @property
    def num_disks(self) -> int:
        return len(self.disks)

    @property
    def throughput(self) -> float:
        return self.size / self.duration

    @property
    def iops(self) -> float:
        if self.ts_unit == "ms":
            return self.num_io / (self.duration / 1000)
        if self.ts_unit == "us":
            return self.num_io / (self.duration * 1000)
        if self.ts_unit == "ns":
            return self.num_io / (self.duration * 1000000)
        if self.ts_unit == "m" or "min" in self.ts_unit:
            return self.num_io / (self.duration * 60)
        # duration assumed to be in seconds
        return self.num_io / self.duration

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

    def to_indented_file(self, file: IndentedFile):
        with file.section("General"):
            file.writeln("IOPS: %f", self.iops)
            file.writeln("Throughput: %f", self.throughput)

        with file.section("Timestamp"):
            file.writeln("Unit: %s", self.ts_unit)
            file.writeln("Start: %d", self.start_ts)
            file.writeln("End: %d", self.end_ts)
            file.writeln("Duration: %d", self.duration)

        with file.section("Disks"):
            file.writeln("Count: %d", self.num_disks)
            file.writeln("ID: [%s]", ", ".join(sorted(self.disks)))

        with file.section("IO Ratio"):
            file.writeln("Read: %f", self.read_ratio)
            file.writeln("Write: %f", self.write_ratio)
            file.writeln("RW: %f", self.rw_ratio)

        with file.section("IOs"):
            file.writeln("Total: %d", self.num_io)
            file.writeln("Reads: %d", self.read_count)
            file.writeln("Writes: %d", self.write_count)

        with file.section("Size"):
            file.writeln("Unit: %s", self.size_unit)
            file.writeln("Total: %f", self.size)
            with file.section("Read"):
                self.read_size.to_indented_file(file)
            with file.section("Write"):
                self.write_size.to_indented_file(file)
        with file.section("Offset"):
            self.offset.to_indented_file(file)

        with file.section("IAT"):
            self.iat.to_indented_file(file)

    def to_msgpack(self, path: Path | str | io.BufferedIOBase):
        if isinstance(path, (str, Path)):
            with open(path, "wb") as file:
                file.write(to_msgpack(self))
            return

        path.write(to_msgpack(self))

    def from_msgpack(self, data: bytes):
        self = from_msgpack(Characteristic, data)


@serde
@dataclass
class Characteristics(UserList[Characteristic]):
    data: list[Characteristic] = field(default_factory=list)

    def to_indented_file(self, file: IndentedFile, section_prefix: str = ""):
        for i, char in enumerate(self):
            with file.section(f"{i}" if section_prefix == "" else f"{section_prefix} {i}"):
                char.to_indented_file(file)

    def to_msgpack(self, path: Path | str | io.BufferedIOBase):
        if isinstance(path, (str, Path)):
            with open(path, "wb") as file:
                file.write(to_msgpack(self))
            return

        path.write(to_msgpack(self))

    def from_msgpack(self, data: bytes):
        self = from_msgpack(Characteristics, data)


__all__ = ["Statistic", "Characteristic", "Characteristics"]
