from pathlib import Path
from typing import Annotated, List

import pandas as pd

import typer

from clio.flashnet.preprocessing.add_filter import add_filter_v2
from clio.flashnet.preprocessing.feature_engineering import feature_engineering
from clio.flashnet.preprocessing.labeling import labeling

from clio.utils.logging import LogLevel, log_global_setup
from clio.utils.timer import Timer, default_timer
from clio.utils.trace_pd import normalize_df_ts_record, trace_get_labeled_paths

app = typer.Typer(name="FlashNet Preprocess", pretty_exceptions_enable=False)


@app.command()
def directory(
    data: Annotated[Path, typer.Argument(help="The data directory to use for preprocessing", exists=True, file_okay=False, dir_okay=True, resolve_path=True)],
    output: Annotated[Path, typer.Option(help="The output path to write the results to")],
    log_level: Annotated[LogLevel, typer.Option(help="The log level to use")] = LogLevel.INFO,
    ext: Annotated[str, typer.Option(help="The extension of the files to use", show_default=True)] = ".trace",
    profile_name: Annotated[str, typer.Option(help="The profile name to use", show_default=True)] = "profile_v1",
    device: Annotated[int, typer.Option(help="The device to use for preprocessing")] = -1,
    continue_df: Annotated[bool, typer.Option(help="Continue from previous dataset", show_default=True)] = True,
):
    args = locals()

    global_start_time = default_timer()

    output.mkdir(parents=True, exist_ok=True)
    log = log_global_setup(output / "log.txt", level=log_level)

    log.info("Arguments:")
    for arg, value in args.items():
        log.info(f"{arg}: {value}")

    ########################################
    # Data directory
    ########################################

    # if len(data) == 0:
    #     raise FileNotFoundError(f"No paths found in {data}")
    # Get the dataset paths
    # dataset_paths: list[Path] = []
    # for data_dir in data:
    #     dataset_paths.extend(trace_get_labeled_paths(data_dir, profile_name=profile_name, ext=ext, sort_by=lambda p: int(p.with_suffix("").name.split("_")[1])))

    dataset_paths = trace_get_labeled_paths(data, profile_name=profile_name, ext=ext, sort_fn=lambda p: int(p.with_suffix("").name.split("_")[1]))

    if len(dataset_paths) == 0:
        raise FileNotFoundError(f"No labeled files found in {data}")

    prev_df: dict[int, pd.DataFrame] = {}
    for i in range(5):
        prev_df[i] = None

    for i, dataset_path in enumerate(dataset_paths):
        output_dir = output / f"chunk_{i}"
        output_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Processing dataset: {dataset_path}")

        dfs: dict[int, pd.DataFrame] = {}
        old_replay = False
        with open(dataset_path, "r") as f:
            line = f.readline()
        n_cols = len(line.split(","))
        log.info("Number of columns: %s", n_cols)
        if n_cols == 8:
            df = pd.read_csv(dataset_path, names=["ts_record", "latency", "io_type", "size", "offset", "ts_submit", "device", "ts"], header=None)
        else:
            df = pd.read_csv(dataset_path, names=["ts_record", "latency", "io_type", "size", "offset", "ts_submit", "size_after_replay"], header=None)
            old_replay = True

        if "device" in df.columns:
            if device == -1:
                log.info("Include all devices")
                for dev in df["device"].unique():
                    dfs[dev] = df[df["device"] == dev]
                    log.info("Max TS Record of Device %s: %s", dev, dfs[dev]["ts_submit"].max())
            else:
                log.info("Include device: %s", device)
                df = df[df["device"] == device]
                dfs[device] = df
        else:
            dfs[0] = df

        for dev, df in dfs.items():
            if old_replay:
                # Original dataset
                df_path = output_dir / "profile_v1.feat_v6_ts.dataset"
                readonly_df_path = output_dir / "profile_v1.feat_v6_ts.readonly.dataset"

                # Filtered dataset
                filtered_df_path = output_dir / "profile_v1_filter.feat_v6_ts.dataset"
                readonly_filtered_df_path = output_dir / "profile_v1_filter.feat_v6_ts.readonly.dataset"
            else:
                # original dataset
                base_dir = output_dir / f"device_{dev}"
                base_dir.mkdir(parents=True, exist_ok=True)
                df_path = base_dir / f"profile_v1.feat_v6_ts.dataset"
                readonly_df_path = base_dir / f"profile_v1.feat_v6_ts.readonly.dataset"

                # filtered dataset
                filtered_df_path = base_dir / f"profile_v1_filter.feat_v6_ts.dataset"
                readonly_filtered_df_path = base_dir / f"profile_v1_filter.feat_v6_ts.readonly.dataset"

            if all([df_path.exists(), readonly_df_path.exists(), filtered_df_path.exists(), readonly_filtered_df_path.exists()]):
                log.info("Skipping dataset %s with device %s", i, dev)
                continue

            if not df_path.exists():
                log.info("Columns: %s", df.columns)
                log.info("Number of rows: %s", len(df))
                log.info("[B] Device %s ts_record max: %s", dev, df["ts_record"].max())
                df = normalize_df_ts_record(df, col="ts_record")
                df = normalize_df_ts_record(df, col="ts_submit")
                log.info("[A] Device %s ts_record max: %s", dev, df["ts_record"].max())

                with Timer("Labeling") as t:
                    df = labeling(df)
                log.info("Labeling took %s s", t.elapsed)
                log.info("[AL] Device %s ts_record max: %s", dev, df["ts_record"].max())

                with Timer("Feature engineering") as t:
                    df, readonly_df = feature_engineering(df, prev_data=prev_df.get(dev))
                log.info("Feature engineering took %s s", t.elapsed)
                df.to_csv(df_path, index=False)
                readonly_df.to_csv(readonly_df_path, index=False)
            else:
                log.info("Reading dataset...")
                df = pd.read_csv(df_path)
                if readonly_df_path.exists():
                    log.info("Reading readonly dataset...")
                    readonly_df = pd.read_csv(readonly_df_path)
                else:
                    log.info("Creating readonly dataset...")
                    readonly_df = df[df["io_type"] == 1]
                    readonly_df.to_csv(readonly_df_path, index=False)

            if continue_df:
                prev_df[dev] = df
                log.info("Previous dataset of device %s ts_record max: %s", dev, prev_df[dev]["ts_record"].max())

            for k, v in prev_df.items():
                log.info("Key: %s, Value none: %s", k, v is None)

            if filtered_df_path.exists():
                filtered_df = pd.read_csv(filtered_df_path)
                if readonly_filtered_df_path.exists():
                    log.info("Reading readonly filtered dataset...")
                    readonly_filtered_df = pd.read_csv(readonly_filtered_df_path)
                else:
                    log.info("Creating readonly filtered dataset...")
                    readonly_filtered_df = filtered_df[filtered_df["io_type"] == 1]
                    readonly_filtered_df.to_csv(readonly_filtered_df_path, index=False)
            else:
                log.info("Filtering...")
                if filtered_df_path.exists():
                    filtered_df = pd.read_csv(filtered_df_path)
                else:
                    with Timer("Filtering") as t:
                        filtered_df = add_filter_v2(df)
                    log.info("Filtering took %s s", t.elapsed)
                    filtered_df.to_csv(filtered_df_path, index=False)

                if readonly_filtered_df_path.exists():
                    ...
                else:
                    readonly_filtered_df = filtered_df[filtered_df["io_type"] == 1]
                    readonly_filtered_df.to_csv(readonly_filtered_df_path, index=False)

    log.info("Preprocessing done in %ss", default_timer() - global_start_time)


@app.command()
def file(
    data: Annotated[Path, typer.Argument(help="The data directory to use for preprocessing", exists=True, file_okay=True, dir_okay=False, resolve_path=True)],
    output: Annotated[Path, typer.Option(help="The output path to write the results to")],
    log_level: Annotated[LogLevel, typer.Option(help="The log level to use")] = LogLevel.INFO,
    device: Annotated[int, typer.Option(help="The device to use for preprocessing")] = -1,
):
    args = locals()

    global_start_time = default_timer()

    output.mkdir(parents=True, exist_ok=True)
    output_dir = output / data.with_suffix("").name
    output_dir.mkdir(parents=True, exist_ok=True)

    log = log_global_setup(output_dir / "log.txt", level=log_level)

    log.info("Arguments:")
    for arg, value in args.items():
        log.info(f"{arg}: {value}")

    ########################################
    # Data directory
    ########################################
    dataset_path = data

    log.info(f"Processing dataset: {dataset_path}")

    dfs: dict[int, pd.DataFrame] = {}
    old_replay = False
    with open(dataset_path, "r") as f:
        line = f.readline()
    n_cols = len(line.split(","))
    log.info("Number of columns: %s", n_cols)
    if n_cols == 8:
        df = pd.read_csv(dataset_path, names=["ts_record", "latency", "io_type", "size", "offset", "ts_submit", "device", "ts"], header=None)
    else:
        df = pd.read_csv(dataset_path, names=["ts_record", "latency", "io_type", "size", "offset", "ts_submit", "size_after_replay"], header=None)
        old_replay = True

    if "device" in df.columns:
        if device == -1:
            log.info("Include all devices")
            for dev in df["device"].unique():
                dfs[dev] = df[df["device"] == dev]
        else:
            log.info("Include device: %s", device)
            df = df[df["device"] == device]
            dfs[device] = df
    else:
        dfs[0] = df

    for dev, df in dfs.items():
        if old_replay:
            # Original dataset
            df_path = output_dir / "profile_v1.feat_v6_ts.dataset"
            readonly_df_path = output_dir / "profile_v1.feat_v6_ts.readonly.dataset"

            # Filtered dataset
            filtered_df_path = output_dir / "profile_v1_filter.feat_v6_ts.dataset"
            readonly_filtered_df_path = output_dir / "profile_v1_filter.feat_v6_ts.readonly.dataset"
        else:
            # original dataset
            base_dir = output_dir / f"device_{dev}"
            base_dir.mkdir(parents=True, exist_ok=True)
            df_path = base_dir / f"profile_v1.feat_v6_ts.dataset"
            readonly_df_path = base_dir / f"profile_v1.feat_v6_ts.readonly.dataset"

            # filtered dataset
            filtered_df_path = base_dir / f"profile_v1_filter.feat_v6_ts.dataset"
            readonly_filtered_df_path = base_dir / f"profile_v1_filter.feat_v6_ts.readonly.dataset"

        if all([df_path.exists(), readonly_df_path.exists(), filtered_df_path.exists(), readonly_filtered_df_path.exists()]):
            log.info("Skipping dataset with device %s", dev)
            continue

        if not df_path.exists():
            log.info("Columns: %s", df.columns)
            log.info("Number of rows: %s", len(df))
            df = normalize_df_ts_record(df, col="ts_record")
            df = normalize_df_ts_record(df, col="ts_submit")

            with Timer("Labeling") as t:
                df = labeling(df)
            log.info("Labeling took %s s", t.elapsed)

            with Timer("Feature engineering") as t:
                df, readonly_df = feature_engineering(df, prev_data=None)
            log.info("Feature engineering took %s s", t.elapsed)
            df.to_csv(df_path, index=False)
            readonly_df.to_csv(readonly_df_path, index=False)
        else:
            log.info("Reading dataset...")
            df = pd.read_csv(df_path)
            if readonly_df_path.exists():
                log.info("Reading readonly dataset...")
                readonly_df = pd.read_csv(readonly_df_path)
            else:
                log.info("Creating readonly dataset...")
                readonly_df = df[df["io_type"] == 1]
                readonly_df.to_csv(readonly_df_path, index=False)

        if filtered_df_path.exists():
            filtered_df = pd.read_csv(filtered_df_path)
            if readonly_filtered_df_path.exists():
                log.info("Reading readonly filtered dataset...")
                readonly_filtered_df = pd.read_csv(readonly_filtered_df_path)
            else:
                log.info("Creating readonly filtered dataset...")
                readonly_filtered_df = filtered_df[filtered_df["io_type"] == 1]
                readonly_filtered_df.to_csv(readonly_filtered_df_path, index=False)
        else:
            log.info("Filtering...")
            if filtered_df_path.exists():
                filtered_df = pd.read_csv(filtered_df_path)
            else:
                with Timer("Filtering") as t:
                    filtered_df = add_filter_v2(df)
                log.info("Filtering took %s s", t.elapsed)
                filtered_df.to_csv(filtered_df_path, index=False)

            if readonly_filtered_df_path.exists():
                ...
            else:
                readonly_filtered_df = filtered_df[filtered_df["io_type"] == 1]
                readonly_filtered_df.to_csv(readonly_filtered_df_path, index=False)

    log.info("Preprocessing done in %ss", default_timer() - global_start_time)


if __name__ == "__main__":
    app()
