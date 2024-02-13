from pathlib import Path
from typing import Annotated

import typer

from clio.utils.csv import read_csv_gz
from clio.utils.logging import log_global_setup
from clio.utils.msft import MSFT_CSVWriter, MSFT_Entry

app = typer.Typer()


@app.command()
def msrc(
    files: Annotated[list[Path], "The file(s) to analyze"],
    output: Annotated[Path, typer.Option(help="The output path to write the results to")],
):
    """
    Quickly analyze a file.

    :param file (Path): The file to analyze
    :param verbose (bool): Whether to show extra information
    """
    assert len(files) > 0, "No files to analyze"

    output.mkdir(parents=True, exist_ok=True)
    log = log_global_setup("log.txt", console_width=None)
    min_time = 0

    writer = MSFT_CSVWriter(output / "hm.csv")
    for i, file in enumerate(files):
        log.info("Analyzing %s", file, tab=0)
        for j, (row, _) in enumerate(read_csv_gz(file, contains_header=False, filter=lambda x: x[3] == "Read")):
            if i == 0 and j == 0:
                min_time = float(row[0])

            entry = MSFT_Entry(
                ts_record=(float(row[0]) - min_time) * 0.00001,
                disk_id=row[2],
                offset=int(row[4]),
                io_size=float(row[5]),
                write=row[6] == "Write",
            )
            writer.write(entry)
    writer.close()


@app.command(name="analyze")
def analyze(): ...


if __name__ == "__main__":
    app()
