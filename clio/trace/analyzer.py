import sys
from pathlib import Path
from typing import Annotated

import pandas as pd
import typer

from clio.utils.characteristic import Characteristic, Statistic
from clio.utils.indented_file import IndentedFile
from clio.utils.logging import LogLevel, log_global_setup
from clio.utils.query import QueryExecutionException, get_query
from clio.utils.stats import Stats
from clio.utils.trace import TraceReader
from clio.utils.trace_pd import TraceWindowGeneratorContext, read_raw_trace_as_df, trace_time_window_generator

app = typer.Typer(name="Analyzer", pretty_exceptions_enable=False)


@app.command()
def quick(
    file: Annotated[Path, "The file to analyze"],
    output: Annotated[Path, typer.Option(help="The output path to write the results to")],
    query: Annotated[str, typer.Option(help="The query to filter the data")] = "",
):
    """
    Quickly analyze a file.

    :param file (Path): The file to analyze
    """
    output.mkdir(parents=True, exist_ok=True)
    log = log_global_setup(output / "log.txt")
    log.info("Analyzing %s", file, tab=0)

    # Read the file
    data = TraceReader(file)
    stats = Characteristic()

    try:
        q = get_query(query)
        iter = data.iter_filter(lambda e: q(e.as_dict())) if q else data
        for entry in iter:
            stats.disks.add(entry.disk_id)
            if entry.read:
                stats.read_count += 1
                # stats.read_size += entry.io_size
            else:
                stats.write_count += 1
                # stats.write_size += entry.io_size
            # stats.offset += entry.offset
    except QueryExecutionException as e:
        log.error("Failed to execute expression: %s", e)
        sys.exit(1)

    log.info("Statistics", tab=0)
    log.info("%s", stats)

    stats_file = Stats()
    stats.write_stats(stats_file)
    log.info("Saving stats to %s", output / "stats.stats", tab=0)
    stats_file.save(output / "stats.stats")


@app.command()
def window(
    file: Annotated[Path, "The file to analyze"],
    output: Annotated[Path, typer.Option(help="The output path to write the results to")],
    window: Annotated[float, typer.Option(help="The window size to use for the analysis in seconds")] = 60,
    query: Annotated[str, typer.Option(help="The query to filter the data")] = "",
    log_level: Annotated[LogLevel, typer.Option(help="The log level to use")] = LogLevel.INFO,
):
    """
    Fully analyze a file

    :param file (Path): The file to analyze
    :param output (Path): The output path to write the results to
    """
    output.mkdir(parents=True, exist_ok=True)
    log = log_global_setup(output / "log.txt", level=log_level)
    log.info("Analyzing %s with %s-seconds window", file, window, tab=0)

    try:
        q = get_query(query)
        data = read_raw_trace_as_df(file)
        if q:
            data: pd.DataFrame = data[q({"data": data})]  # type: ignore
        reference_data = pd.DataFrame()
        window_data = pd.DataFrame()

        trace_ctx = TraceWindowGeneratorContext()
        ts_offset = 0.0
        stats_file = IndentedFile(output / "stats.stats")

        for i, trace_ctx, reference_data, window_data, is_interval_valid, is_last in trace_time_window_generator(
            ctx=trace_ctx,
            window_size=window,
            current_trace_df=data,
            trace_paths=[file],
            n_dfs=1,
            return_last_remaining_data=True,
            curr_df_count=0,
            curr_ts_record=ts_offset,
            reference_df=reference_data,
            query=(lambda df: q({"data": df})) if q else lambda df: df,
        ):
            if not is_interval_valid:
                log.debug("Interval is not valid", tab=1)
                continue

            log.info("Processing window %d (reference: %d, window: %d)", i, len(reference_data), len(window_data), tab=1)
            n_data = len(window_data)
            read_count = window_data["read"].sum()
            write_count = n_data - read_count
            characteristic = Characteristic(
                disks=set(window_data["disk_id"].unique()),
                start_ts=int(window_data["ts_record"].min()),
                end_ts=int(window_data["ts_record"].max()),
                duration=int(window_data["ts_record"].max() - window_data["ts_record"].min()),
                read_count=read_count,
                write_count=write_count,
                read_size=Statistic(window_data[window_data["read"]]["io_size"].values),  # type: ignore
                write_size=Statistic(window_data[~window_data["read"]]["io_size"].values),  # type: ignore
                offset=Statistic(window_data["offset"].values),  # type: ignore
            )
            stats_file.writeln("Window %d", i)
            stats_file.inc_indent()
            stats_file.writeln("Reference data size: %d", len(reference_data))
            stats_file.writeln("Window data size: %d", len(window_data))
            stats_file.writeln("Characteristic")
            stats_file.inc_indent()
            characteristic.to_indented_file(stats_file)
            stats_file.dec_indent()
            stats_file.dec_indent()
    except QueryExecutionException as e:
        log.error("Failed to execute expression: %s", e)
        sys.exit(1)

    stats_file.close()


if __name__ == "__main__":
    app()
