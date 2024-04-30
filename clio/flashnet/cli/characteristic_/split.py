import warnings
from pathlib import Path
from typing import Annotated

warnings.filterwarnings("ignore")

import pandas as pd

import typer

from clio.utils.general import general_set_seed, parse_time
from clio.utils.logging import LogLevel, log_get, log_global_setup
from clio.utils.timer import default_timer
from clio.utils.trace_pd import TraceWindowGeneratorContext, trace_get_labeled_paths, trace_time_window_generator

app = typer.Typer(name="Trace -- Split", pretty_exceptions_enable=False)


@app.command()
def split(
    data_dir: Annotated[list[Path], typer.Argument(help="The data directories to use", exists=True, file_okay=False, dir_okay=True, resolve_path=True)],
    output: Annotated[Path, typer.Option(help="The output path to write the results to")],
    window_size: Annotated[str, typer.Option(help="The window size to use (in minute(s))", show_default=True)] = "1m",
    profile_name: Annotated[str, typer.Option(help="The profile name to use", show_default=True)] = "profile_v1",
    log_level: Annotated[LogLevel, typer.Option(help="The log level to use")] = LogLevel.INFO,
):
    args = locals()

    global_start_time = default_timer()

    output.mkdir(parents=True, exist_ok=True)
    log = log_global_setup(output / "log.txt", level=log_level)

    window_size = parse_time(window_size)

    log.info("Args", tab=0)
    for arg in args:
        log.info("%s: %s", arg, args[arg], tab=1)

    trace_paths: dict[str, list[Path]] = {}
    trace_count: int = 0
    ts_offset = 0.0

    for trace_dir in data_dir:
        name = trace_dir.name
        traces = trace_get_labeled_paths(
            trace_dir,
            profile_name=profile_name,
        )
        trace_paths[name] = traces
        trace_count += len(traces)

    if trace_count == 0:
        raise FileNotFoundError(f"No labeled files found in {data_dir}")

    ctx = TraceWindowGeneratorContext()
    prev_data: pd.DataFrame = pd.DataFrame()
    window_dir = output / "window"
    window_dir.mkdir(parents=True, exist_ok=True)
    for i, (trace_name, trace_paths_list) in enumerate(trace_paths.items()):
        is_last_trace = i == len(trace_paths) - 1
        log.info("Trace name: %s, is last trace: %s", trace_name, is_last_trace, tab=1)
        initial_trace_path = trace_paths_list[0]

        next_initial_df = pd.read_csv(initial_trace_path)
        next_initial_df["original_ts_record"] = next_initial_df["ts_record"]
        if prev_data.empty:
            next_initial_df["ts_record"] += ts_offset
        else:
            log.info("Concatenating previous data with length %s", len(prev_data), tab=2)
            # get the last ts_record from prev_data
            ts_offset = prev_data["ts_record"].iloc[-1]
            next_initial_df["ts_record"] += ts_offset

        initial_df = pd.concat([prev_data, next_initial_df], ignore_index=True)
        reference = pd.DataFrame()
        window = pd.DataFrame()

        for i, ctx, reference, window, is_interval_valid, is_last in trace_time_window_generator(
            ctx=ctx,
            window_size=window_size * 60,
            trace_paths=trace_paths_list,
            n_data=len(trace_paths_list),
            current_trace=initial_df,
            reference=reference,
            return_last_remaining_data=not is_last_trace,
            curr_count=0,
            curr_ts_record=0,
            end_ts=-1,
        ):
            if not is_interval_valid:
                continue

            name = f"{trace_name}.idx_{i}"
            log.info("Processing window %s", name, tab=1)
            window.to_csv(window_dir / f"{name}.csv", index=False)

    global_end_time = default_timer()
    log.info("Total elapsed time: %s", global_end_time - global_start_time, tab=0)
