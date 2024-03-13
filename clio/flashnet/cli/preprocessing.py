from pathlib import Path
from typing import Annotated, List

import pandas as pd

import typer

from clio.flashnet.preprocessing.add_filter import add_filter_v2
from clio.flashnet.preprocessing.feature_engineering import feature_engineering
from clio.utils.logging import LogLevel, log_global_setup
from clio.utils.timer import Timer, default_timer
from clio.utils.trace_pd import normalize_df_ts_record, trace_get_labeled_paths

app = typer.Typer(name="FlashNet Preprocess", pretty_exceptions_enable=False)


@app.command()
def preprocess(
    data: Annotated[
        List[Path], typer.Argument(help="The data directory to use for preprocessing", exists=True, file_okay=False, dir_okay=True, resolve_path=True)
    ],
    output: Annotated[Path, typer.Option(help="The output path to write the results to")],
    log_level: Annotated[LogLevel, typer.Option(help="The log level to use")] = LogLevel.INFO,
):
    args = locals()

    global_start_time = default_timer()

    output.mkdir(parents=True, exist_ok=True)
    log = log_global_setup(output / "log.txt", level=log_level)

    ########################################
    # Data directory
    ########################################

    if len(data) == 0:
        raise FileNotFoundError(f"No paths found in {data}")

    # Get the dataset paths
    dataset_paths: list[Path] = []
    for data_dir in data:
        dataset_paths.extend(trace_get_labeled_paths(data_dir))

    if len(dataset_paths) == 0:
        raise FileNotFoundError(f"No labeled files found in {data}")

    prev_df = None

    for i, dataset_path in enumerate(dataset_paths):
        chunk_dir = output / f"chunk_{i}"
        chunk_dir.mkdir(parents=True, exist_ok=True)
        log.info(f"Processing dataset: {dataset_path}")

        # Original dataset
        df_path = chunk_dir / "profile_v1.feat_v6_ts.dataset"
        readonly_df_path = chunk_dir / "profile_v1.feat_v6_ts.readonly.dataset"
        if not df_path.exists():
            df = pd.read_csv(dataset_path)
            df = normalize_df_ts_record(df)
            with Timer("Feature engineering") as t:
                df, readonly_df = feature_engineering(df, prev_data=prev_df)
            log.info("Feature engineering took %s s", t.elapsed)
            df.to_csv(df_path, index=False)
            readonly_df.to_csv(readonly_df_path, index=False)
        else:
            df = pd.read_csv(df_path)
            if readonly_df_path.exists():
                readonly_df = pd.read_csv(readonly_df_path)
            else:
                log.info("Creating readonly dataset...")
                readonly_df = df[df["io_type"] == 1]
                readonly_df.to_csv(readonly_df_path, index=False)
        prev_df = df

        # Filtered dataset
        filtered_df_path = chunk_dir / "profile_v1_filter.feat_v6_ts.dataset"
        readonly_filtered_df_path = chunk_dir / "profile_v1_filter.feat_v6_ts.readonly.dataset"
        if filtered_df_path.exists():
            filtered_df = pd.read_csv(filtered_df_path)
            if readonly_filtered_df_path.exists():
                readonly_filtered_df = pd.read_csv(readonly_filtered_df_path)
            else:
                log.info("Creating readonly filtered dataset...")
                readonly_filtered_df = filtered_df[filtered_df["io_type"] == 1]
                readonly_filtered_df.to_csv(chunk_dir / "profile_v1_filter.feat_v6_ts.readonly.dataset", index=False)
        else:
            log.info("Filtering...")
            with Timer("Filtering") as t:
                filtered_df = add_filter_v2(df)
            log.info("Filtering took %s s", t.elapsed)
            filtered_df.to_csv(filtered_df_path, index=False)
            readonly_filtered_df = filtered_df[filtered_df["io_type"] == 1]
            readonly_filtered_df.to_csv(chunk_dir / "profile_v1_filter.feat_v6_ts.readonly.dataset", index=False)

    log.info("Preprocessing done in %ss", default_timer() - global_start_time)


if __name__ == "__main__":
    app()
