import warnings
from pathlib import Path
from typing import Annotated

warnings.filterwarnings("ignore")

import pandas as pd

import typer

from clio.utils.dataframe import append_to_df
from clio.utils.general import parse_time
from clio.utils.logging import LogLevel, log_global_setup
from clio.utils.query import QueryExecutionException, get_query
from clio.utils.timer import default_timer

app = typer.Typer(name="Trace -- Combine", pretty_exceptions_enable=False)


@app.command()
def combine(
    trace_dir: Annotated[Path, typer.Argument(help="The data directories to use", exists=True, file_okay=False, dir_okay=True, resolve_path=True)],
    output: Annotated[Path, typer.Option(help="The output path to write the results to")],
    window_size: Annotated[str, typer.Option(help="The window size to use (in minute(s))", show_default=True)] = "1m",
    log_level: Annotated[LogLevel, typer.Option(help="The log level to use")] = LogLevel.INFO,
    delimiter: Annotated[str, typer.Option(help="The delimiter to use", show_default=True)] = ",",
    filter_path: Annotated[str, typer.Option(help="The path filter to use", show_default=True)] = "",
):
    args = locals()

    global_start_time = default_timer()

    output_dir = output.parent
    output_dir.mkdir(parents=True, exist_ok=True)
    log = log_global_setup(output_dir / "log.txt", level=log_level)

    window_size = parse_time(window_size)

    log.info("Args", tab=0)
    for arg in args:
        log.info("%s: %s", arg, args[arg], tab=1)

    trace_paths = list(trace_dir.glob("*.trace"))
    trace_paths.sort(key=lambda x: int(x.stem.split("_")[1]))

    try:
        q = get_query(filter_path)
        if q:

            def filter_fn(x):
                return q({"path": x})

            trace_paths = list(filter(filter_fn, trace_paths))
    except QueryExecutionException as e:
        log.error("Error while executing query: %s", e, tab=0)
        return

    if len(trace_paths) == 0:
        log.warn("No trace files found", tab=0)
        return

    log.info("Processing %s trace files", len(trace_paths), tab=0)

    def reader(path: Path):
        return pd.read_csv(path, names=["ts_record", "latency", "io_type", "size", "offset", "ts_submit", "size_after_replay"], header=None)

    data = reader(trace_paths[0])
    data["ts_record"] = data["ts_record"] - data["ts_record"].min()
    for path in trace_paths[1:]:
        new_data = reader(path)
        new_data["ts_record"] += data["ts_record"].max()
        new_data["ts_submit"] += data["ts_submit"].max()
        data = pd.concat(
            [
                data,
                new_data,
            ]
        )

    data.to_csv(output, index=False, header=False)

    global_end_time = default_timer()
    log.info("Total elapsed time: %s", global_end_time - global_start_time, tab=0)


if __name__ == "__main__":
    app()
