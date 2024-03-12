import json
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated

warnings.filterwarnings("ignore")


import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from matplotlib.gridspec import GridSpec

import torch

import typer
from scipy.stats import kruskal, mannwhitneyu

from clio.flashnet.aggregate import calculate_agg
from clio.utils.characteristic import Characteristic, CharacteristicDict, Statistic
from clio.utils.general import parse_time
from clio.utils.indented_file import IndentedFile
from clio.utils.logging import LogLevel, log_get, log_global_setup
from clio.utils.timer import Timer, default_timer
from clio.utils.trace_pd import TraceWindowGeneratorContext, read_dataset_as_df, trace_get_labeled_paths, trace_time_window_generator

app = typer.Typer(name="Trace Characteristics", pretty_exceptions_enable=False)
_log = log_get(__name__)


@app.command()
def analyze(
    data_dir: Annotated[list[Path], typer.Argument(help="The data directories to use", exists=True, file_okay=False, dir_okay=True, resolve_path=True)],
    output: Annotated[Path, typer.Option(help="The output path to write the results to")],
    window_size: Annotated[str, typer.Option(help="The window size to use (in minute(s))", show_default=True)] = "10",
    profile_name: Annotated[str, typer.Option(help="The profile name to use", show_default=True)] = "profile_v1",
    # feat_name: Annotated[str, typer.Option(help="The feature name to use for prediction", show_default=True)] = "feat_v6_ts",
    # window_agg_size: Annotated[int, typer.Option(help="The window aggregation size to use for prediction (in number of I/Os)", show_default=True)] = 10,
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
    characteristics: CharacteristicDict = CharacteristicDict()
    trace_counter = 0
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
            reader=read_dataset_as_df,
            end_ts=-1,
        ):
            if not is_interval_valid:
                continue

            trace_counter += 1
            log.info("Processing window %d", trace_counter, tab=1)
            log.info("Window columns: %s", list(window.columns), tab=2)
            n_data = len(window)
            read_count = int((window["io_type"] == 1).sum())
            write_count = n_data - read_count
            min_ts_record = int(window["ts_record"].min())
            max_ts_record = int(window["ts_record"].max())
            duration = max_ts_record - min_ts_record
            log.debug("Generating size...")
            size = Statistic.generate(window["size"].values)
            log.debug("Generating read size...")
            read_size = Statistic.generate(window[window["io_type"] == 1]["size"].values)
            log.debug("Generating write size...")
            write_size = Statistic.generate(window[window["io_type"] == 0]["size"].values)
            log.debug("Generating offset...")
            offset = Statistic.generate(window["offset"].values)
            log.debug("Generating iat...")
            iat = window["ts_record"].diff().dropna()
            iat[iat < 0] = 0
            iat = Statistic.generate(iat.values)
            log.debug("Generating throughput...")
            throughput = Statistic.generate((window["size"] / window["latency"]).values)
            log.debug("Generating latency...")
            latency = Statistic.generate(window["latency"].values)
            characteristic = Characteristic(
                num_io=n_data,
                disks=set([0]),
                start_ts=min_ts_record,
                end_ts=max_ts_record,
                duration=duration,
                ts_unit="ms",
                read_count=read_count,
                write_count=write_count,
                size=size,
                read_size=read_size,
                write_size=write_size,
                offset=offset,
                iat=iat,
                throughput=throughput,
                latency=latency,
            )
            name = f"{trace_name}.idx_{i}"
            characteristics[name] = characteristic
            characteristics.to_msgpack(output / "characteristics.msgpack")
            characteristics.to_dataframe().to_csv(output / "characteristics.csv", index=False)

            # window_agg = window.copy()
            # window_agg["norm_ts_record"] = window_agg["ts_record"] - window_agg["ts_record"].min()
            # window_agg = calculate_agg(window_agg, group_col="norm_ts_record", window_size=window_agg_size)
            # log.info("Window agg columns: %s", list(window_agg.columns), tab=1)

            if is_last and not reference:
                prev_data = window.copy()
                if prev_data.empty:
                    ts_offset = max_ts_record
                log.info("End of current trace, saving remaining data with length %s", len(prev_data), tab=2)

    global_end_time = default_timer()
    log.info("Total elapsed time: %s", global_end_time - global_start_time, tab=0)


@app.command()
def split(
    data_dir: Annotated[list[Path], typer.Argument(help="The data directories to use", exists=True, file_okay=False, dir_okay=True, resolve_path=True)],
    output: Annotated[Path, typer.Option(help="The output path to write the results to")],
    window_size: Annotated[str, typer.Option(help="The window size to use (in minute(s))", show_default=True)] = "10",
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
            reader=read_dataset_as_df,
            end_ts=-1,
        ):
            if not is_interval_valid:
                continue

            name = f"{trace_name}.idx_{i}"
            log.info("Processing window %s", name, tab=1)
            window.to_csv(window_dir / f"{name}.csv", index=False)

    global_end_time = default_timer()
    log.info("Total elapsed time: %s", global_end_time - global_start_time, tab=0)


def mult_normalize(
    df: pd.DataFrame,
    ignore_columns: list[str] = [],
) -> pd.DataFrame:
    ignore_cols = ignore_columns + [
        "name",
        # "num_io",
        "disks",
        "start_ts",
        "end_ts",
        "ts_unit",
        "size_unit",
        "duration",
    ]
    # normalize columns below using min-max normalization, treat min as 1 and other values are values / min
    norm_df = df.copy()
    for column in df.columns.difference(ignore_cols):
        # min_val = np.max([float(norm_df[column].min()), 1e-7])
        min_val = norm_df[column].min()
        # _log.info("Column: %s, min val %s", column, min_val, tab=1)
        norm_df[column] = norm_df[column] / min_val if min_val > 0 else 1
    return norm_df


@app.command()
def calculate(
    # data_dir: Annotated[list[Path], typer.Argument(help="The data directories", exists=True, file_okay=False, dir_okay=True, resolve_path=True)],
    data_dir: Annotated[Path, typer.Argument(help="The data directory", exists=True, file_okay=False, dir_okay=True, resolve_path=True)],
    characteristic_path: Annotated[
        Path, typer.Option("--characteristic", help="The characteristic file", exists=True, file_okay=True, dir_okay=False, resolve_path=True)
    ],
    output: Annotated[Path, typer.Option(help="The output path to write the results to")],
    # feat_name: Annotated[str, typer.Option(help="The feature name", show_default=True)] = "feat_v6_ts",
    # window_agg_size: Annotated[int, typer.Option(help="The window aggregation size (in number of I/Os)", show_default=True)] = 10,
    log_level: Annotated[LogLevel, typer.Option(help="The log level to use")] = LogLevel.INFO,
):
    args = locals()

    global_start_time = default_timer()

    output.mkdir(parents=True, exist_ok=True)
    log = log_global_setup(output / "log.txt", level=log_level)

    log.info("Args", tab=0)
    for arg in args:
        log.info("%s: %s", arg, args[arg], tab=1)

    characteristics = CharacteristicDict.from_msgpack(characteristic_path)
    log.info("Loaded characteristics from %s", characteristic_path, tab=0)

    # for name, value in characteristics.items():
    #     log.info("Name: %s", name, tab=1)

    characteristics_df = characteristics.to_dataframe()
    log.info("Characteristics dataframe shape: %s", characteristics_df.shape, tab=0)
    log.info("Characteristics dataframe columns: %s", list(characteristics_df.columns), tab=0)
    characteristics_df = mult_normalize(characteristics_df)

    # find pairwise characteristics multiplication

    base_column_dir = output / "column"
    base_column_dir.mkdir(parents=True, exist_ok=True)

    data_dict: dict[int, dict[int, pd.DataFrame]] = {}
    names: set[str] = set()

    # pairwise window that has size vs 2*size vs 3*size and so on
    for column in ["size_avg", "read_size_avg", "iat_avg", "num_io", "latency_avg"]:
        char_df = characteristics_df.copy()
        base_df = char_df[char_df[column] == 1]

        mult_dict: dict[int, pd.DataFrame] = {
            1: base_df,
        }
        for mult in [1.2, 1.5, 2, 3, 4, 5, 6, 7, 8, 9, 10]:
            # find the window that has roughly equal to size * mult
            mult_df = char_df[np.isclose(char_df[column], mult, atol=0.1, rtol=0.0)]
            if mult_df.empty:
                log.info("No window found for %s_mult_%s", column, mult, tab=1)
                continue
            mult_dict[mult] = mult_df
            names.update(mult_df["name"])

        # check if the mult_dict contains only the base
        if len(mult_dict) == 1:
            continue

        # log.info("Names: %s", names, tab=0)

        column_dir = base_column_dir / column
        column_dir.mkdir(parents=True, exist_ok=True)

        log.info("Column: %s", column, tab=1)
        for k, v in mult_dict.items():
            log.info("Key: %s, shape: %s", k, v.shape, tab=2)
            v.to_csv(column_dir / f"{k}.csv", index=False)

        data_dict[column] = mult_dict

    names = sorted(names)

    N_COL = 2
    # create a heatmap for each column
    # for column, v in data_dict.items():
    #     column_dir = base_column_dir / column
    #     log.info("Column: %s", column, tab=0)
    #     len_data = len(v)
    #     n_col = N_COL
    #     n_row = (len_data // n_col) + 1
    #     # if n_row == 1:
    #     #     n_row = 2
    #     log.info("n_row: %s, n_col: %s", n_row, n_col, tab=1)
    #     fig = plt.figure(figsize=(n_col * 6, n_row * 6))
    #     gs = GridSpec(n_row, n_col, figure=fig)
    #     fig.suptitle(f"Column: {column}")
    #     for i, (mult, v2) in enumerate(v.items()):
    #         # fig, ax = plt.subplots(figsize=(6, 6))
    #         # ax = axs[i // n_col, i % n_col]
    #         ax = fig.add_subplot(gs[i // n_col, i % n_col])
    #         v2_cleaned = v2.drop(columns=["name", "disks", "ts_unit", "size_unit"])  # , "start_ts", "end_ts","duration"])
    #         # pick only _avg columns and num_io, iops
    #         cols = [col for col in v2_cleaned.columns if "_avg" in col]
    #         cols.append("num_io")
    #         cols.append("iops")
    #         v2_cleaned = v2_cleaned[cols]
    #         # v2_cleaned = v2_cleaned.filter(like="_avg")
    #         # v2_cleaned = v2_cleaned.drop(columns=v2_cleaned.columns)
    #         corr = v2_cleaned.corr()
    #         log.info("Corr columns: %s", list(corr.columns), tab=1)
    #         sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
    #         ax.set_title(f"Mult: {mult}")
    #         # fig.suptitle(f"Column: {column}")
    #         # fig.tight_layout()
    #         # fig.savefig(column_dir / f"heatmap_{i}.png")
    #         # plt.close(fig)
    #     fig.tight_layout()
    #     fig.savefig(column_dir / "heatmap.png")
    #     plt.close(fig)

    # for column, v in data_dict.items():
    #     for mult, v2 in v.items():
    #         log.info("Column: %s, Mult: %s, Shape: %s", column, mult, v2.shape, tab=1)

    # pairwise plot between the base and the mult
    sns.set_theme(font_scale=2.0)
    for column, v in data_dict.items():
        column_dir = base_column_dir / column
        log.info("Column: %s", column, tab=0)
        for mult, v2 in v.items():
            if mult == 1:
                continue
            log.info("Mult: %s", mult, tab=1)
            base_df = v[1]
            mult_df = v[mult]
            base_df = base_df.drop(columns=["name", "disks", "ts_unit", "size_unit"])
            # fig, ax = plt.subplots(figsize=(10, 10))
            # ax.set_title(f"Column: {column}, Mult: {mult}")
            # pick only _avg columns and num_io, iops
            cols = [col for col in base_df.columns if "_avg" in col]
            cols.append("num_io")
            cols.append("iops")
            base_df = base_df[cols]
            base_df["mult"] = 1
            mult_df = mult_df[cols]
            mult_df["mult"] = mult
            df = pd.concat([base_df, mult_df])

            fig = plt.figure(figsize=(10, 10))
            n_col = 4
            n_row = (len(df.columns) // n_col) + 1
            gs = GridSpec(n_row, n_col, figure=fig)
            fig.suptitle(f"Column: {column}, Mult: {mult}")

            # generate barplot of multiplier of each column
            for i, col in enumerate(df.columns):
                if col == "mult":
                    continue
                ax = fig.add_subplot(gs[i // n_col, i % n_col])
                sns.barplot(x="mult", y=col, data=df, ax=ax, hue="mult", palette="tab10")
                # remove legend
                ax.get_legend().remove()
                ax.set_title(col)
                ax.set_xlabel("Multiplier")
                ax.set_ylabel("")
                # ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
                # ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
                ax.legend

            fig.tight_layout()
            fig.savefig(column_dir / f"barplot_{mult}.png")

            # g = sns.pairplot(df, diag_kind="kde", hue="mult", palette="tab10")
            # sns.move_legend(g, "upper right", bbox_to_anchor=(1.0, 1.0))
            # g.figure.suptitle(f"Column: {column}, Mult: {mult}", y=1.05)
            # g.figure.savefig(column_dir / f"pairplot_{mult}.png", dpi=300)

            # for col in base_df.columns:
            #     if col == "mult":
            #         continue
            #     log.info("Column: %s", col, tab=1)
            #     stat, p = kruskal(base_df[col], mult_df[col])
            #     log.info("Kruskal stats: %s, p-value: %s", stat, p, tab=1)
            #     log.info("Different distribution: %s", p < 0.05, tab=1)
            #     stat, p = mannwhitneyu(base_df[col], mult_df[col])
            #     log.info("Mannwhitneyu stats: %s, p-value: %s", stat, p, tab=1)
            #     log.info("Different distribution: %s", p < 0.05, tab=1)

    windows_idx: dict[str, list[int]] = {}
    for name in names:
        # <name>.idx_<idx> and <name> might contain "." in the name
        name, idx = name.split(".idx_")
        idx = int(idx)
        if name not in windows_idx:
            windows_idx[name] = []
        windows_idx[name].append(idx)

    windows_idx = {k: sorted(v) for k, v in windows_idx.items()}
    window_dir = output / "window"
    window_dir.mkdir(parents=True, exist_ok=True)
    # for name, idxs in windows_idx.items():
    #     log.info("Name: %s, Idxs: %s", name, idxs, tab=1)

    # ctx = TraceWindowGeneratorContext()
    # window_size = 1
    # trace_paths_list = trace_get_labeled_paths(
    #     data_dir / name,
    #     profile_name="profile_v1",
    # )
    # initial_df = read_dataset_as_df(trace_paths_list[0])
    # reference = pd.DataFrame()
    # for i, ctx, reference, window, is_interval_valid, is_last in trace_time_window_generator(
    #     ctx=ctx,
    #     window_size=window_size * 60,
    #     trace_paths=trace_paths_list,
    #     n_data=len(trace_paths_list),
    #     current_trace=initial_df,
    #     reference=reference,
    #     return_last_remaining_data=True,
    #     curr_count=0,
    #     curr_ts_record=0,
    #     reader=read_dataset_as_df,
    #     end_ts=-1,
    # ):
    #     if not is_interval_valid:
    #         continue

    #     if i in idxs:
    #         log.info("Name: %s, Idx: %s, Window shape: %s", name, i, window.shape, tab=1)
    #         window.to_csv(window_dir / f"{name}.idx_{i}.csv", index=False)

    # window_agg = window.copy()
    # window_agg["norm_ts_record"] = window_agg["ts_record"] - window_agg["ts_record"].min()
    # window_agg = calculate_agg(window_agg, group

    global_end_time = default_timer()
    log.info("Total elapsed time: %s", global_end_time - global_start_time, tab=0)


if __name__ == "__main__":
    app()
