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
from clio.utils.trace_pd import TraceWindowGeneratorContext, trace_time_window_generator

app = typer.Typer(name="Analyzer", pretty_exceptions_enable=False)


@app.command()
def whole(
    file: Annotated[Path, typer.Argument(help="The file to analyze", exists=True, file_okay=True, dir_okay=False, resolve_path=True)],
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

    try:
        q = get_query(query)
        data = pd.read_csv(file)
        if q:
            data: pd.DataFrame = data[q({"data": data})]  # type: ignore

        stats_file = IndentedFile(output / "stats.stats")
        stats_file_str = io.StringIO()
        stats_file_str_ifile = IndentedFile(stats_file_str, indent=2)

        characteristic = Characteristic(
            num_io=len(data),
            disks=set(data["disk_id"].unique()),
            start_ts=int(data["ts_record"].min()),
            end_ts=int(data["ts_record"].max()),
            duration=int(data["ts_record"].max() - data["ts_record"].min()),
            ts_unit="ms",
            read_count=int(data["read"].sum()),
            write_count=len(data) - int(data["read"].sum()),
            size=Statistic.generate(data["io_size"].values),  # type: ignore
            read_size=Statistic.generate(data[data["read"]]["io_size"].values),  # type: ignore
            write_size=Statistic.generate(data[~data["read"]]["io_size"].values),  # type: ignore
            offset=Statistic.generate(data["offset"].values),  # type: ignore
            iat=Statistic.generate(data["ts_record"].diff().dropna().values),  # type: ignore
        )

        characteristic.to_indented_file(stats_file)
        characteristic.to_indented_file(stats_file_str_ifile)
        characteristic.to_msgpack(output / "stats.msgpack")
        stats_file.close()

        log.info("Stats:\n%s", stats_file_str.getvalue(), tab=1)
    except QueryExecutionException as e:
        log.error("Failed to execute expression: %s", e)
        sys.exit(1)


@app.command()
def window(
    file: Annotated[Path, typer.Argument(help="The file to analyze", exists=True, file_okay=True, dir_okay=False, resolve_path=True)],
    output: Annotated[Path, typer.Option(help="The output path to write the results to")],
    window: Annotated[float, typer.Option(help="The window size to use for the analysis in seconds")] = 60,
    query: Annotated[str, typer.Option(help="The query to filter the data")] = "",
    log_level: Annotated[LogLevel, typer.Option(help="The log level to use")] = LogLevel.INFO,
):
    """
    Fully analyze a file

    :param file (Path): The file to analyze
    :param output (Path): The output path to write the results to
    :param query (str): The query to filter the data
    :param log_level (LogLevel): The log level to use
    """
    output.mkdir(parents=True, exist_ok=True)
    log = log_global_setup(output / "log.txt", level=log_level)
    log.info("Analyzing %s with %s-seconds window", file, window, tab=0)

    try:
        q = get_query(query)
        data = pd.read_csv(file)
        if q:
            data: pd.DataFrame = data[q({"data": data})]  # type: ignore
        reference_data = pd.DataFrame()
        window_data = pd.DataFrame()

        trace_ctx = TraceWindowGeneratorContext()
        ts_offset = 0.0
        stats_file = IndentedFile(output / "stats.stats")
        characteristics = Characteristics()

        for i, trace_ctx, curr_path, reference_data, window_data, is_interval_valid, is_last in trace_time_window_generator(
            ctx=trace_ctx,
            window_size=window,
            current_trace=data,
            trace_paths=[file],
            n_data=1,
            return_last_remaining_data=True,
            curr_count=0,
            curr_ts_record=ts_offset,
            reference=reference_data,
            query=(lambda df: q({"data": df})) if q else lambda df: df,
        ):
            if not is_interval_valid:
                log.debug("Interval is not valid", tab=1)
                continue

            log.info("Processing window %d (reference: %d, window: %d)", i, len(reference_data), len(window_data), tab=1)
            n_data = len(window_data)
            read_count = int(window_data["read"].sum())
            write_count = n_data - read_count
            min_ts_record = int(window_data["ts_record"].min())
            max_ts_record = int(window_data["ts_record"].max())
            duration = max_ts_record - min_ts_record
            characteristic = Characteristic(
                num_io=n_data,
                disks=set(window_data["disk_id"].unique()),
                start_ts=min_ts_record,
                end_ts=max_ts_record,
                duration=duration,
                ts_unit="ms",
                read_count=read_count,
                write_count=write_count,
                size=Statistic.generate(window_data["io_size"].values),  # type: ignore
                read_size=Statistic.generate(window_data[window_data["read"]]["io_size"].values),  # type: ignore
                write_size=Statistic.generate(window_data[~window_data["read"]]["io_size"].values),  # type: ignore
                offset=Statistic.generate(window_data["offset"].values),  # type: ignore
                iat=Statistic.generate(window_data["ts_record"].diff().dropna().values),  # type: ignore
            )
            characteristics.append(characteristic)

        characteristics.to_indented_file(stats_file, section_prefix="Window")
        characteristics.to_msgpack(output / "stats.msgpack")
    except QueryExecutionException as e:
        log.error("Failed to execute expression: %s", e)
        sys.exit(1)

    stats_file.close()


if __name__ == "__main__":
    app()
