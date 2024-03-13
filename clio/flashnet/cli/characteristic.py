import json
import shutil
import warnings
from dataclasses import dataclass
from pathlib import Path
from textwrap import dedent
from typing import Annotated

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from matplotlib.gridspec import GridSpec

import torch

import typer
from scipy.stats import kruskal, mannwhitneyu, norm

from clio.flashnet.aggregate import calculate_agg
from clio.flashnet.preprocessing.add_filter import add_filter_v2
from clio.flashnet.preprocessing.feature_engineering import feature_engineering

from clio.utils.characteristic import Characteristic, CharacteristicDict, Statistic
from clio.utils.general import general_set_seed, parse_time
from clio.utils.indented_file import IndentedFile
from clio.utils.logging import LogLevel, log_get, log_global_setup
from clio.utils.timer import Timer, default_timer
from clio.utils.trace_pd import TraceWindowGeneratorContext, normalize_df_ts_record, trace_get_labeled_paths, trace_time_window_generator

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
            readonly_data = window[window["io_type"] == 1]
            writeonly_data = window[window["io_type"] == 0]
            log.debug("Generating size...")
            size = Statistic.generate(window["size"].values)
            log.debug("Generating read size...")
            read_size = Statistic.generate(readonly_data["size"].values)
            log.debug("Generating write size...")
            write_size = Statistic.generate(writeonly_data["size"].values)
            log.debug("Generating offset...")
            offset = Statistic.generate(window["offset"].values)
            log.debug("Generating iat...")
            iat = window["ts_record"].diff().dropna()
            iat[iat < 0] = 0
            iat = Statistic.generate(iat.values)
            read_iat = readonly_data["ts_record"].diff().dropna()
            read_iat[read_iat < 0] = 0
            read_iat = Statistic.generate(read_iat.values)
            write_iat = writeonly_data["ts_record"].diff().dropna()
            write_iat[write_iat < 0] = 0
            write_iat = Statistic.generate(write_iat.values)
            log.debug("Generating throughput...")
            throughput = Statistic.generate((window["size"] / window["latency"]).values)
            read_throughput = Statistic.generate((readonly_data["size"] / readonly_data["latency"]).values)
            write_throughput = Statistic.generate((writeonly_data["size"] / writeonly_data["latency"]).values)
            log.debug("Generating latency...")
            latency = Statistic.generate(window["latency"].values)
            read_latency = Statistic.generate(readonly_data["latency"].values)
            write_latency = Statistic.generate(writeonly_data["latency"].values)
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
                read_iat=read_iat,
                write_iat=write_iat,
                throughput=throughput,
                read_throughput=read_throughput,
                write_throughput=write_throughput,
                latency=latency,
                read_latency=read_latency,
                write_latency=write_latency,
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
    seed: Annotated[int, typer.Option(help="The seed to use", show_default=True)] = 3003,
):
    args = locals()

    global_start_time = default_timer()

    general_set_seed(seed)

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

    CHARACTERISTIC_COLUMNS = [
        # read
        "read_size_avg",
        "read_latency_avg",
        "read_iat_avg",
        "read_throughput_avg",
        # general
        "iat_avg",
        "num_io",
        "latency_avg",
        "size_avg",
        "throughput_avg",
        # write
        "write_size_avg",
        "write_latency_avg",
        "write_iat_avg",
        "write_throughput_avg",
    ]

    # pairwise window that has size vs 2*size vs 3*size and so on
    for column in CHARACTERISTIC_COLUMNS:
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

        # NOTE: REMOVE THIS
        if column_dir.exists():
            log.warning("Column directory %s already exists, skipping it...", column_dir, tab=0)
            continue

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
    sns.set_theme(font_scale=1.5)
    for column, v in data_dict.items():
        column_dir = base_column_dir / column
        column_plot_dir = column_dir / "plot"
        # column_plot_dir.mkdir(parents=True, exist_ok=True)
        log.info("Column: %s", column, tab=0)
        for mult, v2 in v.items():
            if mult == 1:
                continue

            mult_plot_dir = column_plot_dir / f"mult_{mult}"
            mult_plot_dir.mkdir(parents=True, exist_ok=True)
            log.info("Mult: %s", mult, tab=1)
            base_df = v[1]
            mult_df = v[mult]
            #
            # cols.append("num_io")
            # cols.append("iops")
            # base_df = base_df[cols]
            # base_df["mult"] = 1
            # mult_df = mult_df[cols]
            # mult_df["mult"] = mult

            # base df will only have 1 row
            assert len(base_df) == 1

            base_df_name = base_df["name"].values[0]
            base_df_data = pd.read_csv(data_dir / f"{base_df_name}.csv")
            base_df_data["iat"] = base_df_data["ts_record"].diff().dropna()
            base_df_data.loc[base_df_data["iat"] < 0, "iat"] = 0
            base_df_data["size"] = base_df_data["size"] / 1_000_000  # convert to MB
            base_df_data["latency"] = base_df_data["latency"] / 1000  # convert to ms
            base_df_data["throughput"] = base_df_data["size"] / base_df_data["latency"] * 1000  # MB/s
            base_df_data = base_df_data[base_df_data["latency"] <= 2.3]  # remove outliers
            base_df_data["mult"] = 1
            base_df_data = base_df_data.drop(
                columns=[
                    "ts_record",
                    "ts_submit",
                    "original_ts_record",
                    "size_after_replay",
                ]
            )
            base_df_data = base_df_data.reset_index(drop=True)
            # base_df_data = base_df_data.reset_index(drop=True)

            base_df_data_writeonly = base_df_data[base_df_data["io_type"] == 0].drop(columns=["io_type"])
            if len(base_df_data_writeonly) > 50000:
                base_df_data_writeonly = base_df_data_writeonly.sample(50000)
            base_df_data_readonly = base_df_data[base_df_data["io_type"] == 1].drop(columns=["io_type"])
            if len(base_df_data_readonly) > 50000:
                base_df_data_readonly = base_df_data_readonly.sample(50000)

            for name in list(mult_df["name"].unique()):
                m_df = mult_df[mult_df["name"] == name]
                assert len(m_df) == 1

                m_df_name = m_df["name"].values[0]
                m_df_data = pd.read_csv(data_dir / f"{m_df_name}.csv")
                m_df_data["iat"] = m_df_data["ts_record"].diff().dropna()
                m_df_data.loc[m_df_data["iat"] < 0, "iat"] = 0
                m_df_data["size"] = m_df_data["size"] / 1_000_000  # convert to MB
                m_df_data["latency"] = m_df_data["latency"] / 1000  # convert to ms
                m_df_data = m_df_data[m_df_data["latency"] <= 2.3]  # remove outliers
                m_df_data["throughput"] = m_df_data["size"] / m_df_data["latency"] * 1000  # MB/s
                m_df_data["mult"] = mult
                m_df_data = m_df_data.drop(
                    columns=[
                        "ts_record",
                        "ts_submit",
                        "original_ts_record",
                        "size_after_replay",
                    ]
                )
                m_df_data = m_df_data.reset_index(drop=True)
                # m_df_data = m_df_data.reset_index(drop=True)
                m_df_data_writeonly = m_df_data[m_df_data["io_type"] == 0].drop(columns=["io_type"])
                if len(m_df_data_writeonly) > 50000:
                    m_df_data_writeonly = m_df_data_writeonly.sample(50000)
                m_df_data_readonly = m_df_data[m_df_data["io_type"] == 1].drop(columns=["io_type"])
                if len(m_df_data_readonly) > 50000:
                    m_df_data_readonly = m_df_data_readonly.sample(50000)

                # base_df_data_ = base_df_data.drop(columns=["ts_record", "original_ts_record"])
                # m_df_data_ = m_df_data.drop(columns=["ts_record", "original_ts_record"])

                # g = sns.pairplot(
                #     pd.concat([base_df_data.sample(5000), m_df_data.sample(5000)]).reset_index(drop=True),
                #     diag_kind="kde",
                #     hue="mult",
                #     palette="tab10",
                # )
                # sns.move_legend(g, "upper right", bbox_to_anchor=(1.0, 1.0))
                # g.figure.suptitle(f"Base (1x) vs Mult ({mult}x), Column: {column}", y=1.05)
                # g.figure.savefig(mult_plot_dir / f"pairplot_{name}.png", dpi=300)
                # plt.close(g.figure)

                fig = plt.figure(figsize=(12, 12))

                fig.suptitle(
                    "\n".join([f"Base (1x) vs Mult ({mult}x)", f"Column: {column}", f"Base name: {base_df_name}", f"Mult name: {m_df_name}"]),
                    fontsize=14,
                )
                n_col = 3
                n_row = 12 // n_col
                gs = GridSpec(n_row, n_col, figure=fig)

                # plotting latency
                ax = fig.add_subplot(gs[0, 0])
                sns.kdeplot(base_df_data["latency"], ax=ax, label="Base")
                sns.kdeplot(m_df_data["latency"], ax=ax, label="Mult")
                ax.set_title("Latency")
                ax.set_xlabel("Latency (ms)")
                ax.set_ylabel("Density")
                # ax.set_xlim(0, 1.0)

                # plotting latency read
                ax = fig.add_subplot(gs[0, 1])
                sns.kdeplot(base_df_data_readonly["latency"], ax=ax, label="Base")
                sns.kdeplot(m_df_data_readonly["latency"], ax=ax, label="Mult")
                ax.set_title("Latency Read")
                ax.set_xlabel("Latency (ms)")
                ax.set_ylabel("Density")
                # ax.set_xlim(0, 1.0)

                ax = fig.add_subplot(gs[0, 2])
                sns.kdeplot(base_df_data_writeonly["latency"], ax=ax, label="Base")
                sns.kdeplot(m_df_data_writeonly["latency"], ax=ax, label="Mult")
                ax.set_title("Latency Write")
                ax.set_xlabel("Latency (ms)")
                ax.set_ylabel("Density")
                # ax.set_xlim(0, 1.0)

                ax = fig.add_subplot(gs[1, 0])
                sns.kdeplot(base_df_data["size"], ax=ax, label="Base")
                sns.kdeplot(m_df_data["size"], ax=ax, label="Mult")
                ax.set_title("Size")
                ax.set_xlabel("Size (MB)")
                ax.set_ylabel("Density")

                ax = fig.add_subplot(gs[1, 1])
                sns.kdeplot(base_df_data_readonly["size"], ax=ax, label="Base")
                sns.kdeplot(m_df_data_readonly["size"], ax=ax, label="Mult")
                ax.set_title("Size Read")
                ax.set_xlabel("Size (MB)")
                ax.set_ylabel("Density")

                ax = fig.add_subplot(gs[1, 2])
                sns.kdeplot(base_df_data_writeonly["size"], ax=ax, label="Base")
                sns.kdeplot(m_df_data_writeonly["size"], ax=ax, label="Mult")
                ax.set_title("Size Write")
                ax.set_xlabel("Size (MB)")
                ax.set_ylabel("Density")

                ax = fig.add_subplot(gs[2, 0])
                sns.kdeplot(base_df_data["iat"], ax=ax, label="Base")
                sns.kdeplot(m_df_data["iat"], ax=ax, label="Mult")
                ax.set_title("IAT")
                ax.set_xlabel("IAT (ms)")
                ax.set_ylabel("Density")

                ax = fig.add_subplot(gs[2, 1])
                sns.kdeplot(base_df_data_readonly["iat"], ax=ax, label="Base")
                sns.kdeplot(m_df_data_readonly["iat"], ax=ax, label="Mult")
                ax.set_title("IAT Read")
                ax.set_xlabel("IAT (ms)")
                ax.set_ylabel("Density")

                ax = fig.add_subplot(gs[2, 2])
                sns.kdeplot(base_df_data_writeonly["iat"], ax=ax, label="Base")
                sns.kdeplot(m_df_data_writeonly["iat"], ax=ax, label="Mult")
                ax.set_title("IAT Write")
                ax.set_xlabel("IAT (ms)")
                ax.set_ylabel("Density")

                ax = fig.add_subplot(gs[3, 0])
                sns.kdeplot(base_df_data["throughput"], ax=ax, label="Base")
                sns.kdeplot(m_df_data["throughput"], ax=ax, label="Mult")
                ax.set_title("Throughput")
                ax.set_xlabel("Throughput (MB/s)")
                ax.set_ylabel("Density")

                ax = fig.add_subplot(gs[3, 1])
                sns.kdeplot(base_df_data_readonly["throughput"], ax=ax, label="Base")
                sns.kdeplot(m_df_data_readonly["throughput"], ax=ax, label="Mult")
                ax.set_title("Throughput Read")
                ax.set_xlabel("Throughput (MB/s)")
                ax.set_ylabel("Density")

                ax = fig.add_subplot(gs[3, 2])
                sns.kdeplot(base_df_data_writeonly["throughput"], ax=ax, label="Base")
                sns.kdeplot(m_df_data_writeonly["throughput"], ax=ax, label="Mult")
                ax.set_title("Throughput Write")
                ax.set_xlabel("Throughput (MB/s)")

                handles, labels = ax.get_legend_handles_labels()
                fig.legend(handles, labels, loc="upper right", bbox_to_anchor=(1.0, 1.0))

                fig.tight_layout()
                fig.savefig(mult_plot_dir / f"plot.base_vs_mult_{mult}.base_{base_df_name}.mult_{m_df_name}.png", dpi=300)
                plt.close(fig)

                gmm_base_df = GaussianMixture(n_components=4, random_state=seed)
                values = base_df_data_readonly["latency"].values
                gmm_base_df.fit(values.reshape(-1, 1))
                # means = gmm_base_df.means_
                # std = np.sqrt(gmm_base_df.covariances_)
                # weights = gmm_base_df.weights_

                # plot GMM kde
                # fig, ax = plt.subplots(figsize=(10, 5))
                # ax.set_title(f"Base (1x) vs Mult ({mult}x), Column: {column}, GMM")
                # # ax.set_xlabel("Size (MB)")
                # values = m_df_data_readonly["latency"].values
                # x = np.linspace(np.min(values), np.max(values), 50000)
                # for i, (mean, std, weight) in enumerate(zip(means, std, weights)):
                #     pdf = weight * norm.pdf(x, mean, std)
                #     pdf = pdf.reshape(-1, 1)
                #     x_ = x.reshape(-1, 1)
                #     ax.plot(x_, pdf, label=i)
                # ax.legend()
                # fig.tight_layout()
                # fig.savefig(mult_plot_dir / f"gmm.base_{base_df_name}.mult_{m_df_name}.png", dpi=300)
                # plt.close(fig)

                # predict the mult dataset
                m_df_data_readonly["cluster"] = gmm_base_df.predict(m_df_data_readonly["latency"].values.reshape(-1, 1))

                # plot cluster
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.set_title(f"Base (1x) vs Mult ({mult}x), Column: {column}, GMM")
                ax.set_xlabel("Latency (ms)")
                ax.set_ylabel("Size (MB)")
                sns.scatterplot(
                    x="latency",
                    y="size",
                    hue="cluster",
                    data=m_df_data_readonly,
                    palette="tab10",
                    style="reject",
                    ax=ax,
                )
                # ax.set_xlim(0, 0.5)
                fig.tight_layout()
                fig.savefig(mult_plot_dir / f"cluster_base.mult_{mult}.base_{base_df_name}.mult_{m_df_name}.gmm.png", dpi=300)
                plt.close(fig)

                gmm_m_df = GaussianMixture(n_components=4, random_state=seed)
                values = m_df_data_readonly["latency"].values
                gmm_m_df.fit(values.reshape(-1, 1))

                m_df_data_readonly["cluster"] = gmm_m_df.predict(m_df_data_readonly["latency"].values.reshape(-1, 1))

                # plot cluster
                fig, ax = plt.subplots(figsize=(10, 5))
                ax.set_title(f"Base (1x) vs Mult ({mult}x), Column: {column}, GMM")
                ax.set_xlabel("Latency (ms)")
                ax.set_ylabel("Size (MB)")
                # markers based on reject (1 or 0)
                sns.scatterplot(
                    x="latency",
                    y="size",
                    hue="cluster",
                    data=m_df_data_readonly,
                    style="reject",
                    palette="tab10",
                    ax=ax,
                )
                # ax.set_xlim(0, 0.5)
                fig.tight_layout()
                fig.savefig(mult_plot_dir / f"cluster_mult.mult_{mult}.base_{base_df_name}.mult_{m_df_name}.gmm.png", dpi=300)
                plt.close(fig)

                # plot GMM kde

                # def getKernelDensityEstimation(values, x, bandwidth=0.2, kernel="gaussian"):
                #     from sklearn.neighbors import KernelDensity

                #     model = KernelDensity(kernel=kernel, bandwidth=bandwidth)
                #     model.fit(values[:, np.newaxis])
                #     log_density = model.score_samples(x[:, np.newaxis])
                #     return np.exp(log_density)

                # kde = getKernelDensityEstimation(m_df_data_readonly["size"].values, np.linspace(0, 1, 1000))
                # extreme_points_idx = getExtremePoints(kde, typeOfExtreme="min")
                # log.info("Extreme points idx: %s", extreme_points_idx, tab=1)
                # for idx in extreme_points_idx:
                #     log.info("Extreme point: %s, value: %s", idx, kde[idx], tab=2)

                break

            with open(mult_plot_dir / "mult-file.txt", "w") as f:
                f.write("! Base\n")
                f.write(base_df_name)
                f.write("\n")
                f.write("\n")
                f.write(f"! Multiplier {mult}")
                f.write("\n")
                for name in list(mult_df["name"].unique()):
                    f.write(name)
                    f.write("\n")

            # df = pd.concat([base_df, mult_df])

            # fig = plt.figure(figsize=(10, 10))
            # n_col = 4
            # n_row = (len(df.columns) // n_col) + 1
            # gs = GridSpec(n_row, n_col, figure=fig)
            # fig.suptitle(f"Column: {column}, Mult: {mult}")

            # # generate barplot of multiplier of each column
            # for i, col in enumerate(df.columns):
            #     if col == "mult":
            #         continue
            #     ax = fig.add_subplot(gs[i // n_col, i % n_col])
            #     sns.barplot(x="mult", y=col, data=df, ax=ax, hue="mult", palette="tab10")
            #     # remove legend
            #     ax.get_legend().remove()
            #     ax.set_title(col)
            #     ax.set_xlabel("Multiplier")
            #     ax.set_ylabel("")
            #     # ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
            #     # ax.xaxis.set_major_formatter(ticker.ScalarFormatter())
            #     ax.legend

            # fig.tight_layout()
            # fig.savefig(column_dir / f"barplot_{mult}.png")

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
    # window_dir = output / "window"
    # window_dir.mkdir(parents=True, exist_ok=True)
    for name, idxs in windows_idx.items():
        log.info("Name: %s, Idxs: %s", name, idxs, tab=1)

        # ctx = TraceWindowGeneratorContext()
        # window_size = 1
        # trace_paths_list = trace_get_labeled_paths(
        #     data_dir / name,
        #     profile_name="profile_v1",
        # )
        # initial_df = pd.read_csv(trace_paths_list[0])
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

    traces: dict[str, dict[str, str]] = {}
    key = None
    counter = 0
    with open(list_of_window, "r") as f:
        for line in f:
            line = line.strip()
            if line.startswith("#"):
                continue
            if line == "":
                continue

            # remove # .... from line
            line = line.split("#")[0]
            line = line.strip()

            if line.startswith("!"):
                counter = 0
                key = line[1:]
                key = key.strip()
                if key not in traces:
                    traces[key] = {}
                continue

            if ":" in line:
                name, value = line.split(":")
                name = name.strip()
                value = value.strip()
                traces[key][name] = value
                continue

            if key is not None:
                traces[key][counter] = line
                counter += 1

    raw_data_dir = output / "raw"
    preprocessed_data_dir = output / "preprocessed"

    for trace_group, trace_dict in traces.items():
        log.info("Trace group: %s", trace_group, tab=0)
        raw_trace_group_dir = raw_data_dir / trace_group
        raw_trace_group_dir.mkdir(parents=True, exist_ok=True)
        for trace in trace_dict.values():
            log.info("Trace: %s", trace, tab=1)
            src_path = data_dir / f"{trace}.csv"
            dst_path = raw_trace_group_dir / f"{trace}.csv"
            shutil.copy(src_path, dst_path)

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
            p = data_dir / f"{trace}.csv"
            df = pd.read_csv(p)
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


if __name__ == "__main__":
    app()
