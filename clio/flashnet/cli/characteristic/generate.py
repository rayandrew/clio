import json
import warnings
from pathlib import Path
from typing import Annotated

warnings.filterwarnings("ignore")


import pandas as pd

import typer

from clio.flashnet.cli.characteristic.utils import parse_list_file
from clio.flashnet.preprocessing.add_filter import add_filter_v2
from clio.flashnet.preprocessing.feature_engineering import feature_engineering
from clio.flashnet.preprocessing.labeling import labeling

from clio.utils.general import general_set_seed, parse_time
from clio.utils.logging import LogLevel, log_global_setup
from clio.utils.timer import Timer, default_timer
from clio.utils.trace_pd import normalize_df_ts_record

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


if __name__ == "__main__":
    app()
