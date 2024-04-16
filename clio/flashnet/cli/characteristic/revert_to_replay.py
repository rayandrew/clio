import json
import warnings
from pathlib import Path
from typing import Annotated

warnings.filterwarnings("ignore")

import pandas as pd

import typer

from clio.flashnet.cli.characteristic.utils import parse_list_file

from clio.utils.general import general_set_seed, parse_time
from clio.utils.logging import LogLevel, log_global_setup
from clio.utils.timer import default_timer
from clio.utils.trace_pd import normalize_df_ts_record

app = typer.Typer(name="Trace Characteristics -- Revert to Replay", pretty_exceptions_enable=False)


@app.command()
def heimdall(
    data_dir: Annotated[Path, typer.Argument(help="The data directory", exists=True, file_okay=False, dir_okay=True, resolve_path=True)],
    list_of_window: Annotated[
        Path, typer.Option("--list-file", help="The list of window files", exists=True, file_okay=True, dir_okay=False, resolve_path=True)
    ],
    output: Annotated[Path, typer.Option(help="The output path to write the results to")],
    window_size: Annotated[str, typer.Option(help="The window size to use (in minute(s))", show_default=True)] = "1m",
    # feat_name: Annotated[str, typer.Option(help="The feature name", show_default=True)] = "feat_v6_ts",
    # window_agg_size: Annotated[int, typer.Option(help="The window aggregation size (in number of I/Os)", show_default=True)] = 10,
    log_level: Annotated[LogLevel, typer.Option(help="The log level to use")] = LogLevel.INFO,
    seed: Annotated[int, typer.Option(help="The seed to use", show_default=True)] = 3003,
):
    args = locals()

    global_start_time = default_timer()

    general_set_seed(seed)

    window_size = parse_time(window_size)

    output.mkdir(parents=True, exist_ok=True)
    log = log_global_setup(output / "log.txt", level=log_level)

    log.info("Args", tab=0)
    for arg in args:
        log.info("%s: %s", arg, args[arg], tab=1)

    # raw_data_dir = output / "raw"
    replay_data_dir = output  # / "heimdall"
    traces = parse_list_file(list_of_window)

    for trace_group, trace_dict in traces.items():
        log.info("Trace group: %s", trace_group, tab=0)

        replay_trace_group_dir = replay_data_dir / trace_group

        if replay_trace_group_dir.exists():
            log.warning("Replay trace group dir exists: %s", replay_trace_group_dir, tab=1)
            log.warning("Delete the directory and re-run the command if you want to regenerate the data", tab=1)
            continue

        replay_trace_group_dir.mkdir(parents=True, exist_ok=True)

        with open(replay_trace_group_dir / "trace_dict.json", "w") as f:
            json.dump(trace_dict, f)

        for i, (trace_name, trace) in enumerate(trace_dict.items()):
            p = data_dir / f"{trace}.csv"
            df = pd.read_csv(p)
            df = normalize_df_ts_record(df)
            df = df.drop(columns=["ts_submit", "original_ts_record", "size_after_replay", "latency", "size_after_replay", "reject"], errors="ignore")
            df["dummy"] = 0
            # reorder
            df = df[["ts_record", "dummy", "offset", "size", "io_type"]]
            # save without header and use " " as separator
            df.to_csv(replay_trace_group_dir / f"{i}.{trace}.trace", index=False, header=False, sep=" ")

    global_end_time = default_timer()
    log.info("Total elapsed time: %s", global_end_time - global_start_time, tab=0)


@app.command()
def linnos(
    data_dir: Annotated[Path, typer.Argument(help="The data directory", exists=True, file_okay=False, dir_okay=True, resolve_path=True)],
    list_of_window: Annotated[
        Path, typer.Option("--list-file", help="The list of window files", exists=True, file_okay=True, dir_okay=False, resolve_path=True)
    ],
    output: Annotated[Path, typer.Option(help="The output path to write the results to")],
    window_size: Annotated[str, typer.Option(help="The window size to use (in minute(s))", show_default=True)] = "1m",
    # feat_name: Annotated[str, typer.Option(help="The feature name", show_default=True)] = "feat_v6_ts",
    # window_agg_size: Annotated[int, typer.Option(help="The window aggregation size (in number of I/Os)", show_default=True)] = 10,
    log_level: Annotated[LogLevel, typer.Option(help="The log level to use")] = LogLevel.INFO,
    seed: Annotated[int, typer.Option(help="The seed to use", show_default=True)] = 3003,
):
    #### NO DUMMY COLUMN HERE ####

    args = locals()

    global_start_time = default_timer()

    general_set_seed(seed)

    window_size = parse_time(window_size)

    output.mkdir(parents=True, exist_ok=True)
    log = log_global_setup(output / "log.txt", level=log_level)

    log.info("Args", tab=0)
    for arg in args:
        log.info("%s: %s", arg, args[arg], tab=1)

    # raw_data_dir = output / "raw"
    replay_data_dir = output  # / "linnos"
    traces = parse_list_file(list_of_window)

    for trace_group, trace_dict in traces.items():
        log.info("Trace group: %s", trace_group, tab=0)

        replay_trace_group_dir = replay_data_dir / trace_group

        if replay_trace_group_dir.exists():
            log.warning("Replay trace group dir exists: %s", replay_trace_group_dir, tab=1)
            log.warning("Delete the directory and re-run the command if you want to regenerate the data", tab=1)
            continue

        replay_trace_group_dir.mkdir(parents=True, exist_ok=True)

        with open(replay_trace_group_dir / "trace_dict.json", "w") as f:
            json.dump(trace_dict, f)

        for i, (trace_name, trace) in enumerate(trace_dict.items()):
            p = data_dir / f"{trace}.csv"
            df = pd.read_csv(p)
            df = normalize_df_ts_record(df)
            df = df.drop(columns=["ts_submit", "original_ts_record", "size_after_replay", "latency", "size_after_replay", "reject"], errors="ignore")
            # reorder
            df = df[["ts_record", "offset", "size", "io_type"]]
            # save without header and use " " as separator
            df.to_csv(replay_trace_group_dir / f"{i}.{trace}.trace", index=False, header=False, sep=" ")

    global_end_time = default_timer()
    log.info("Total elapsed time: %s", global_end_time - global_start_time, tab=0)


@app.command()
def heimdall_to_linnos(
    trace_dir: Annotated[Path, typer.Argument(help="The data directory", exists=True, file_okay=False, dir_okay=True, resolve_path=True)],
    output: Annotated[Path, typer.Option(help="The output path to write the results to")],
    log_level: Annotated[LogLevel, typer.Option(help="The log level to use")] = LogLevel.INFO,
):
    args = locals()

    global_start_time = default_timer()

    output.mkdir(parents=True, exist_ok=True)
    log = log_global_setup(output / "log.txt", level=log_level)

    log.info("Args", tab=0)
    for arg in args:
        log.info("%s: %s", arg, args[arg], tab=1)

    replay_data_dir = output  # / "heimdall-to-linnos"
    replay_data_dir.mkdir(parents=True, exist_ok=True)

    for i, trace_file in enumerate(trace_dir.iterdir()):
        df = pd.read_csv(trace_file, sep=" ", header=None, names=["ts_record", "dummy", "offset", "size", "io_type"])
        # reorder
        df = df[["ts_record", "offset", "size", "io_type"]]
        # save without header and use " " as separator
        df.to_csv(replay_data_dir / f"{i}.{trace_file.name}", index=False, header=False, sep=" ")

    global_end_time = default_timer()
    log.info("Total elapsed time: %s", global_end_time - global_start_time, tab=0)


@app.command()
def linnos_to_heimdall(
    trace_dir: Annotated[Path, typer.Argument(help="The data directory", exists=True, file_okay=False, dir_okay=True, resolve_path=True)],
    output: Annotated[Path, typer.Option(help="The output path to write the results to")],
    log_level: Annotated[LogLevel, typer.Option(help="The log level to use")] = LogLevel.INFO,
):
    args = locals()

    global_start_time = default_timer()

    output.mkdir(parents=True, exist_ok=True)
    log = log_global_setup(output / "log.txt", level=log_level)

    log.info("Args", tab=0)
    for arg in args:
        log.info("%s: %s", arg, args[arg], tab=1)

    replay_data_dir = output  # / "linnos-to-heimdall"
    replay_data_dir.mkdir(parents=True, exist_ok=True)

    for i, trace_file in enumerate(trace_dir.iterdir()):
        df = pd.read_csv(trace_file, sep=" ", header=None, names=["ts_record", "offset", "size", "io_type"])
        df["dummy"] = 0
        df = df[["ts_record", "dummy", "offset", "size", "io_type"]]
        # save without header and use " " as separator
        df.to_csv(replay_data_dir / f"{i}.{trace_file.name}", index=False, header=False, sep=" ")

    global_end_time = default_timer()
    log.info("Total elapsed time: %s", global_end_time - global_start_time, tab=0)


if __name__ == "__main__":
    app()
