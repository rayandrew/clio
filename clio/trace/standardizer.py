import sys
from pathlib import Path
from typing import Annotated

import typer

from clio.utils.csv import read_csv_gz
from clio.utils.logging import log_global_setup
from clio.utils.trace import TraceEntry, TraceWriter

app = typer.Typer(name="Standardizer")


@app.command()
def msrc(
    files: Annotated[list[Path], "The file(s) to analyze"],
    output: Annotated[Path, typer.Option(help="The output path to write the results to")],
):
    """
    Standardize the MSRC trace format to MSFT format.

    :param files (list[Path]): The file(s) to analyze
    :param output (Path): The output path to write the results to
    """
    assert len(files) > 0, "No files to analyze"

    output.mkdir(parents=True, exist_ok=True)
    log = log_global_setup(output / "log.txt", console_width=None)
    min_time = 0

    writer = TraceWriter(output / f"{output.stem}.trace", sep=" ", write_header=False)
    for i, file in enumerate(files):
        log.info("Standardizing %s", file, tab=0)
        for j, (row, _) in enumerate(read_csv_gz(file, contains_header=False)):
            if i == 0 and j == 0:
                min_time = float(row[0])

            entry = TraceEntry(
                ts_record=(float(row[0]) - min_time) * 0.00001,
                disk_id=row[2],
                offset=int(row[4]),
                io_size=int(row[5]),
                read=row[3] == "Read",
            )
            writer.write(entry)
    writer.close()


@app.command(name="analyze")
def analyze(): ...


if __name__ == "__main__":
    app()
