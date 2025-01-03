import json
import warnings
from pathlib import Path
from typing import Annotated

warnings.filterwarnings("ignore")


import pandas as pd

import typer

from clio.flashnet.cli.characteristic_.utils import parse_list_file
from clio.flashnet.preprocessing.add_filter import add_filter_v2
from clio.flashnet.preprocessing.feature_engineering import feature_engineering
from clio.flashnet.preprocessing.labeling import labeling

from clio.utils.general import general_set_seed, parse_time
from clio.utils.logging import LogLevel, log_global_setup
from clio.utils.timer import Timer, default_timer
from clio.utils.trace_pd import normalize_df_ts_record
from natsort import natsorted

app = typer.Typer(name="Trace Characteristics -- Generate", pretty_exceptions_enable=False)


@app.command()
def generate(
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
    static_prev_df: Annotated[bool, typer.Option(help="Use static prev_df", show_default=True)] = False,
    relabel: Annotated[bool, typer.Option(help="Relabel the trace", show_default=True)] = False,
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
    preprocessed_data_dir = output / "preprocessed"
    traces = parse_list_file(list_of_window)

    for trace_group, trace_dict in traces.items():
        log.info("Trace group: %s", trace_group, tab=0)
        # raw_trace_group_dir = raw_data_dir / trace_group
        # raw_trace_group_dir.mkdir(parents=True, exist_ok=True)
        # for trace in trace_dict.values():
        #     log.info("Trace: %s", trace, tab=1)
        #     src_path = data_dir / f"{trace}.csv"
        #     dst_path = raw_trace_group_dir / f"{trace}.csv"
        #     shutil.copy(src_path, dst_path)

        # trace_list_p = [data_dir / f"{t}.csv" for t in trace_list]
        preprocessed_trace_group_dir = preprocessed_data_dir / trace_group

        if preprocessed_trace_group_dir.exists():
            log.warning("Preprocessed trace group dir exists: %s", preprocessed_trace_group_dir, tab=1)
            log.warning("Delete the directory and re-run the command if you want to regenerate the data", tab=1)
            continue

        preprocessed_trace_group_dir.mkdir(parents=True, exist_ok=True)

        with open(preprocessed_trace_group_dir / "trace_dict.json", "w") as f:
            json.dump(trace_dict, f)

        prev_df_is_chosen = False
        prev_df = None
        for i, (trace_name, trace) in enumerate(trace_dict.items()):
            # name, idx = name.split(".idx_")
            # idx = int(idx)
            if relabel:
                p = data_dir / f"{i}.{trace}.trace"
                ## columns are ts_record,latency,io_type,size,offset,ts_submit,size_after_replay,reject,original_ts_record
                df = pd.read_csv(p, names=["ts_record", "latency", "io_type", "size", "offset", "ts_submit", "size_after_replay"])
            else:
                p = data_dir / f"{trace}.csv"
                df = pd.read_csv(p)
            # relabeling
            if relabel:
                df = labeling(df)
            df = normalize_df_ts_record(df)
            with Timer("Feature Engineering") as t:
                df, readonly_df = feature_engineering(df, prev_data=prev_df)
            log.info("Feature engineering took %s s", t.elapsed, tab=1)
            df.to_csv(preprocessed_trace_group_dir / f"{i}.{trace}.profile_v1.feat_v6_ts.dataset", index=False)
            readonly_df.to_csv(preprocessed_trace_group_dir / f"{i}.{trace}.profile_v1.feat_v6_ts.readonly.dataset", index=False)

            if not static_prev_df:
                log.info("Choosing previous df", tab=1)
                prev_df = df.copy()
            else:
                if not prev_df_is_chosen:
                    log.info("Choosing previous df", tab=1)
                    prev_df = df.copy()
                    prev_df_is_chosen = True

            with Timer("Filtering") as t:
                filtered_df = add_filter_v2(df)
            log.info("Filtering took %s s", t.elapsed, tab=1)
            filtered_df.to_csv(preprocessed_trace_group_dir / f"{i}.{trace}.profile_v1_filter.feat_v6_ts.dataset", index=False)
            readonly_filtered_df = filtered_df[filtered_df["io_type"] == 1]
            readonly_filtered_df.to_csv(preprocessed_trace_group_dir / f"{i}.{trace}.profile_v1_filter.feat_v6_ts.readonly.dataset", index=False)

    global_end_time = default_timer()
    log.info("Total elapsed time: %s", global_end_time - global_start_time, tab=0)


# Generate by glob by folder dir -> no list file
@app.command()
def generate_v2(
    data_dir: Annotated[Path, typer.Argument(help="The data directory", exists=True, file_okay=False, dir_okay=True, resolve_path=True)],
    output: Annotated[Path, typer.Option(help="The output path to write the results to")],
    window_size: Annotated[str, typer.Option(help="The window size to use (in minute(s))", show_default=True)] = "1m",
    # feat_name: Annotated[str, typer.Option(help="The feature name", show_default=True)] = "feat_v6_ts",
    # window_agg_size: Annotated[int, typer.Option(help="The window aggregation size (in number of I/Os)", show_default=True)] = 10,
    log_level: Annotated[LogLevel, typer.Option(help="The log level to use")] = LogLevel.INFO,
    seed: Annotated[int, typer.Option(help="The seed to use", show_default=True)] = 3003,
    static_prev_df: Annotated[bool, typer.Option(help="Use static prev_df", show_default=True)] = False,
    relabel: Annotated[bool, typer.Option(help="Relabel the trace", show_default=True)] = False,
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

    preprocessed_data_dir = output
    # make dir
    preprocessed_data_dir.mkdir(parents=True, exist_ok=True)
    # get all files in data_dir, csvs only
    traces = [f.stem for f in data_dir.iterdir() if f.suffix == ".csv"]
    # natsort traces
    traces = natsorted(traces)

    prev_df_is_chosen = False
    prev_df = None
    for i, trace in enumerate(traces):
        csv_path = data_dir / f"{trace}.csv"

        log.info("Trace: %s", csv_path, tab=0)

        df = pd.read_csv(csv_path, names=["ts_record", "latency", "io_type", "size", "offset", "ts_submit", "size_after_replay"])

        # relabeling
        if relabel:
            df = labeling(df)
        df = normalize_df_ts_record(df)

        with Timer("Feature Engineering") as t:
            df, readonly_df = feature_engineering(df, prev_data=prev_df)

        log.info("Feature engineering took %s s", t.elapsed, tab=1)
        df.to_csv(preprocessed_data_dir / f"{i}.{trace}.profile_v1.feat_v6_ts.dataset", index=False)
        readonly_df.to_csv(preprocessed_data_dir / f"{i}.{trace}.profile_v1.feat_v6_ts.readonly.dataset", index=False)
        if not static_prev_df:
            log.info("Choosing previous df", tab=1)
            prev_df = df.copy()
        else:
            if not prev_df_is_chosen:
                log.info("Choosing previous df", tab=1)
                prev_df = df.copy()
                prev_df_is_chosen = True
        with Timer("Filtering") as t:
            filtered_df = add_filter_v2(df)
        log.info("Filtering took %s s", t.elapsed, tab=1)
        filtered_df.to_csv(preprocessed_data_dir / f"{i}.{trace}.profile_v1_filter.feat_v6_ts.dataset", index=False)
        readonly_filtered_df = filtered_df[filtered_df["io_type"] == 1]
        readonly_filtered_df.to_csv(preprocessed_data_dir / f"{i}.{trace}.profile_v1_filter.feat_v6_ts.readonly.dataset", index=False)

    global_end_time = default_timer()
    log.info("Total elapsed time: %s", global_end_time - global_start_time, tab=0)


def rescale(
    data_dir: Annotated[Path, typer.Argument(help="The data directory", exists=True, file_okay=False, dir_okay=True, resolve_path=True)],
    output_dir: Annotated[Path, typer.Argument(help="The output directory", exists=False, file_okay=False, dir_okay=True, resolve_path=True)],
    metric: Annotated[str, typer.Option(help="The metric to use", show_default=True)] = "iops",
    multiplier: Annotated[float, typer.Option(help="The multiplier to use", show_default=True)] = 1.0,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get all files in data_dir, csvs, .tar.gz, .tar.xz only
    traces = [f for f in data_dir.iterdir() if (f.suffix == ".csv" or f.suffix == ".gz")]
    traces = natsorted(traces)

    if multiplier == 1.0:
        print("No rescaling needed! Exiting...")
        return
    import tarfile

    rw_props = pd.DataFrame(columns=["chunk", "num_io_ori", "num_io_rescaled", "read_ori", "write_ori", "read_rescaled", "write_rescaled"])
    for i, trace_path in enumerate(traces):
        if i % 100 == 0:
            print(f"Processing {i+1}/{len(traces)}")
            rw_props.to_csv(output_dir / "a_rw_props.csv", index=False)
        csv_path = trace_path

        df = None
        if trace_path.suffixes == [".tar", ".gz"]:
            with tarfile.open(csv_path, "r:*") as tar:
                csv_path = tar.getnames()[0]
                df = pd.read_csv(tar.extractfile(csv_path), names=["ts_record", "dummy", "offset", "size", "io_type"], sep=" ")
        elif trace_path.suffixes == [".csv"]:
            df = pd.read_csv(csv_path, names=["ts_record", "dummy", "offset", "size", "io_type"], sep=" ")
        length_ori = len(df)
        io_type_ori = df["io_type"].value_counts()
        df = handle_rescale(df, metric, multiplier)
        df = df.sort_values(by="ts_record")

        length_rescaled = len(df)
        io_type_rescaled = df["io_type"].value_counts()
        rw_props = rw_props._append(
            {
                "chunk": i,
                "num_io_ori": length_ori,
                "num_io_rescaled": length_rescaled,
                "read_ori": io_type_ori.get(0, 0),
                "write_ori": io_type_ori.get(1, 0),
                "read_rescaled": io_type_rescaled.get(0, 0),
                "write_rescaled": io_type_rescaled.get(1, 0),
            },
            ignore_index=True,
        )
        df.to_csv(output_dir / f"chunk_{i}.csv", index=False, header=False, sep=" ")
    rw_props.to_csv(output_dir / "a_rw_props.csv", index=False)


import numpy as np


def handle_rescale(df, metric, multiplier):
    if "iops" in metric.lower():
        if multiplier > 1:
            df_new = df.groupby("io_type", group_keys=False).apply(lambda x: x.sample(frac=multiplier, replace=True, random_state=42))
            duplicates = df_new["ts_record"].duplicated(keep=False)
            if duplicates.any():
                random_offsets = np.random.uniform(-100, 100, duplicates.sum())
                df_new.loc[duplicates, "ts_record"] += random_offsets
        else:
            df_new = df.groupby("io_type", group_keys=False).apply(lambda x: x.sample(frac=multiplier, random_state=42))

        return df_new
    else:
        raise ValueError(f"Unknown metric: {metric}")


if __name__ == "__main__":
    app()
