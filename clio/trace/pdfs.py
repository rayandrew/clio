import io
import sys
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from clio.utils.characteristic import Characteristic, Characteristics, Statistic
from clio.utils.indented_file import IndentedFile
from clio.utils.logging import LogLevel, log_global_setup
from clio.utils.query import QueryExecutionException, get_query
from clio.utils.trace_pd import TraceWindowGeneratorContext, read_raw_trace_as_df, trace_time_window_generator

app = typer.Typer(name="PDF", pretty_exceptions_enable=False)


@app.command()
def characteristics(
    dir: Annotated[Path, typer.Argument(help="The characteristics file to plot", exists=True, file_okay=False, dir_okay=True, resolve_path=True)],
    output: Annotated[Path, typer.Option(help="The output path to write the results to")],
    query: Annotated[str, typer.Option(help="The query to filter the data")] = "",
    log_level: Annotated[LogLevel, typer.Option(help="The log level to use")] = LogLevel.INFO,
): ...


@app.command()
def temp(): ...


if __name__ == "__main__":
    app()
