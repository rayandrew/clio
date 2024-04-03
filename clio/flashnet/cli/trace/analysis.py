import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import typer

from clio.flashnet.constants import FEATURE_COLUMNS

from clio.utils.dataframe import append_to_df
from clio.utils.general import ratio_to_percentage_str, torch_set_seed
from clio.utils.indented_file import IndentedFile
from clio.utils.logging import LogLevel, log_global_setup
from clio.utils.path import rmdir
from clio.utils.timer import Timer, default_timer
from clio.utils.trace_pd import trace_get_dataset_paths

app = typer.Typer(name="Trace -- Correlation Analysis")


@app.command()
def correlation(
    data_dir: Annotated[
        Path, typer.Argument(help="The test data directory to use for analysis", exists=True, file_okay=False, dir_okay=True, resolve_path=True)
    ],
    output: Annotated[Path, typer.Option(help="The output path to write the results to")],
    log_level: Annotated[LogLevel, typer.Option(help="The log level to use")] = LogLevel.INFO,
    profile_name: Annotated[str, typer.Option(help="The profile name to use for analysis", show_default=True)] = "profile_v1",
    feat_name: Annotated[str, typer.Option(help="The feature name to use for prediction", show_default=True)] = "feat_v6_ts",
    seed: Annotated[int, typer.Option(help="The seed to use for random number generation", show_default=True)] = 3003,
):
    args = locals()

    global_start_time = default_timer()

    output.mkdir(parents=True, exist_ok=True)
    log = log_global_setup(output / "log.txt", level=log_level)

    # window_size = parse_time(window_size)
    # duration = parse_time(duration)

    log.info("Args", tab=0)
    for arg in args:
        log.info("%s: %s", arg, args[arg], tab=1)

    data_paths = trace_get_dataset_paths(
        data_dir, profile_name=profile_name, feat_name=feat_name, readonly_data=True, sort_by=lambda x: int(x.name.split(".")[0])
    )
    if len(data_paths) == 0:
        raise ValueError(f"No dataset found in {data_dir}")

    # correlation analysis
    torch_set_seed(seed)

    for i, data_path in enumerate(data_paths):
        log.info("Processing data #%s: %s", i, data_path, tab=0)
        dataset = pd.read_csv(data_path)
        dataset = dataset.dropna()
        dataset = dataset[FEATURE_COLUMNS + ["reject", "latency", "offset"]]

        fig, ax = plt.subplots(figsize=(10, 6))
        # correlation matrix
        corr = dataset.corr()
        sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax)
        ax.set_title("Correlation Matrix")
        fig.tight_layout()
        fig.savefig(output / f"correlation_matrix_{i}.png", dpi=300)
        plt.close(fig)

    global_end_time = default_timer()
    log.info("Total elapsed time: %s s", global_end_time - global_start_time, tab=0)


@app.command()
def kde(
    data_dir: Annotated[
        Path, typer.Argument(help="The test data directory to use for analysis", exists=True, file_okay=False, dir_okay=True, resolve_path=True)
    ],
    output: Annotated[Path, typer.Option(help="The output path to write the results to")],
    log_level: Annotated[LogLevel, typer.Option(help="The log level to use")] = LogLevel.INFO,
    profile_name: Annotated[str, typer.Option(help="The profile name to use for analysis", show_default=True)] = "profile_v1",
    feat_name: Annotated[str, typer.Option(help="The feature name to use for prediction", show_default=True)] = "feat_v6_ts",
    seed: Annotated[int, typer.Option(help="The seed to use for random number generation", show_default=True)] = 3003,
):
    args = locals()

    global_start_time = default_timer()

    output.mkdir(parents=True, exist_ok=True)
    log = log_global_setup(output / "log.txt", level=log_level)

    # window_size = parse_time(window_size)
    # duration = parse_time(duration)

    log.info("Args", tab=0)
    for arg in args:
        log.info("%s: %s", arg, args[arg], tab=1)

    data_paths = trace_get_dataset_paths(
        data_dir, profile_name=profile_name, feat_name=feat_name, readonly_data=True, sort_by=lambda x: int(x.name.split(".")[0])
    )
    if len(data_paths) == 0:
        raise ValueError(f"No dataset found in {data_dir}")

    # correlation analysis
    torch_set_seed(seed)

    columns = FEATURE_COLUMNS + ["reject", "latency", "offset"]

    datasets = {str(data_path): pd.read_csv(data_path).dropna()[columns] for data_path in data_paths}

    for column in columns:
        fig, ax = plt.subplots(figsize=(8, 5))

        for i, data_path in enumerate(data_paths):
            dataset = datasets[str(data_path)]
            sns.kdeplot(dataset[column], label=f"Data {i}", ax=ax)

        ax.set_title(f"KDE of {column}")
        fig.tight_layout()
        fig.savefig(output / f"kde_{column}.png", dpi=300)
        plt.close(fig)

    global_end_time = default_timer()
    log.info("Total elapsed time: %s s", global_end_time - global_start_time, tab=0)


if __name__ == "__main__":
    app()
