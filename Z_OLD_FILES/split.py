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
    trace: Annotated[Path, typer.Argument(help="The data directories to use", exists=True, file_okay=True, dir_okay=False, resolve_path=True)],
    output: Annotated[Path, typer.Option(help="The output path to write the results to")],
    window_size: Annotated[str, typer.Option(help="The window size to use (in minute(s))", show_default=True)] = "1m",
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

    trace_paths = [trace]

    ctx = TraceWindowGeneratorContext()

    def reader(path: Path):
        df = pd.read_csv(path, names=["ts_record", "latency", "io_type", "size", "offset", "ts_submit", "size_after_replay"], header=None)
        return df

    initial_df = reader(trace_paths[0])
    reference = pd.DataFrame()
    window = pd.DataFrame()

    for i, ctx, curr_path, reference, window, is_interval_valid, is_last in trace_time_window_generator(
        ctx=ctx,
        window_size=window_size * 60,
        trace_paths=trace_paths,
        n_data=len(trace_paths),
        current_trace=initial_df,
        reference=reference,
        return_last_remaining_data=True,
        curr_count=0,
        curr_ts_record=0,
        end_ts=-1,
        reader=reader,
    ):
        if i == 0:
            window.to_csv(output / f"trace_{i}.trace", index=False, header=False)

        break

    global_end_time = default_timer()
    log.info("Total elapsed time: %s", global_end_time - global_start_time, tab=0)


if __name__ == "__main__":
    app()