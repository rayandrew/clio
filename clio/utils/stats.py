import statistics
from pathlib import Path
from typing import Any, Sequence

import numpy as np


def write_stats(stats: list[str], output: Path | str, mode: str = "w"):
    with open(output, mode) as f:
        f.write("\n".join(stats) + "\n")


class Stats:
    def __init__(self) -> None:
        self.buffer: list[str] = []

    def add(self, stats: list[str]):
        self.buffer += stats

    def add_line(self, line: str):
        self.buffer.append(line)

    def add_separator(self):
        self.buffer.append("========================================")

    def add_stats(self, name: str, data: Sequence[float]):
        self.add(generate_stats(data, name))
        self.add_separator()

    def add_kv(self, key: str, value: Any):
        self.buffer.append(f"{key}: {value}")

    def add_new_line(self):
        self.buffer.append("")

    def save(self, output: Path | str, mode: str = "w"):
        write_stats(self.buffer, output, mode=mode)


class StatsParser:
    def __init__(self, path_or_buffer: Path | str | list[str]) -> None:
        self.results = {}

        if isinstance(path_or_buffer, Path) or isinstance(path_or_buffer, str):
            path = Path(path_or_buffer)
            with open(path, "r") as f:
                stats_content = f.readlines()
                self.results = self._parse_stats(stats_content)
        elif isinstance(path_or_buffer, list):
            self.results = self._parse_stats(path_or_buffer)

        self.keys = list(self.results.keys())

    def _normalize_value(self, value: Any):
        if value == "-":
            return 0
        try:
            return float(value)
        except ValueError:
            return value

    def _parse_stats(self, stats_content: list[str]):
        stats = {}
        prefix = ""
        for line in stats_content:
            trimmed = line.strip()
            if "=" in trimmed:
                prefix = ""
                continue
            if ":" in trimmed:
                key, value = trimmed.split(":")
                key = key.strip()
                key = "_".join(key.split(" "))
                if prefix != "":
                    key = prefix + "_" + key
                key = key.lower()
                value = self._normalize_value(value.strip())
                stats[key] = value
            else:
                if trimmed == "":
                    continue

                prefix = trimmed

        return stats

    def attrs(self) -> list[str]:
        return self.keys

    def __getattr__(self, name: str, throw: bool = False):
        if name not in self.keys:
            if throw:
                raise AttributeError(f"Attribute {name} not found")
            return None
        return self.results.get(name, None)


def generate_stats(data: Sequence[float], name: str) -> list[str]:
    stats = []
    stats.append(f"{name}")
    stats.append(f"\tMin: {min(data)}")
    stats.append(f"\tMax: {max(data)}")
    stats.append(f"\tMean: {sum(data) / len(data)}")
    stats.append(f"\tMedian: {sorted(data)[len(data) // 2]}")
    stats.append(f"\tMode: {statistics.mode(data)}")
    stats.append(f"\tStd: {statistics.stdev(data)}")
    stats.append(f"\tVar: {statistics.variance(data)}")
    # percentiles
    for p in range(10, 101, 10):
        stats.append(f"\tP{str(p)}: {np.percentile(data, p)}")

    return stats


def exp_done(output: Path | str):
    output = Path(output)
    with open(output / "done", "w") as f:
        f.write("done\n")


def check_exp_done(output: Path | str):
    output = Path(output)
    return (output / "done").exists()


__all__ = ["Stats", "generate_stats", "write_stats", "StatsParser", "exp_done", "check_exp_done"]
