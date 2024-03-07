import json
import sys
from pathlib import Path
from timeit import default_timer as timer
from typing import Annotated, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from matplotlib.gridspec import GridSpec

import typer
from fitsne import FItSNE as fitsne

from clio.flashnet.eval import Trainer, flashnet_evaluate, flashnet_predict
from clio.flashnet.training import flashnet_train
from clio.utils.cpu_usage import CPUUsage
from clio.utils.indented_file import IndentedFile
from clio.utils.logging import LogLevel, log_get, log_global_setup
from clio.utils.timer import Timer, default_timer
from clio.utils.trace_pd import TraceWindowGeneratorContext, read_dataset_as_df, trace_get_dataset_paths, trace_time_window_generator

_log = log_get(__name__)

app = typer.Typer(name="Worload Prediction Analysis", pretty_exceptions_enable=False)


@app.command()
def workload_prediction(
    result_dir: Annotated[
        Path, typer.Argument(help="The test data directory to use for prediction", exists=True, file_okay=False, dir_okay=True, resolve_path=True)
    ],
    output: Annotated[Path, typer.Option(help="The output path to write the results to")],
    log_level: Annotated[LogLevel, typer.Option(help="The log level to use")] = LogLevel.INFO,
):
    args = locals()

    global_start_time = default_timer()

    output.mkdir(parents=True, exist_ok=True)
    log = log_global_setup(output / "log.txt", level=log_level)

    ########################################
    # Result directory
    ########################################

    model_dir_path = result_dir / "models"
    if not model_dir_path.exists():
        raise FileNotFoundError(f"Model directory not found: {model_dir_path}")

    window_dir_path = result_dir / "window"
    if not window_dir_path.exists():
        raise FileNotFoundError(f"Window directory not found: {window_dir_path}")

    results_path = result_dir / "results.csv"
    if not results_path.exists():
        raise FileNotFoundError(f"Results file not found: {results_path}")

    models_result_path = result_dir / "models.csv"
    if not models_result_path.exists():
        raise FileNotFoundError(f"Models result file not found: {models_result_path}")

    ########################################
    # Analyze the workload prediction
    ########################################

    models_df = pd.read_csv(models_result_path)
    results_df = pd.read_csv(results_path)

    ########################################
    # Plot models selection
    ########################################

    top_models = models_df.sort_values(by="count", ascending=False).head(10)
    top_models = top_models.reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(top_models["model"], top_models["count"])
    ax.set_xlabel("Model")
    ax.set_ylabel("Count")
    ax.set_title("Top 10 Models")
    ax.set_xticklabels(top_models["model"], rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(output / "top_models.png", dpi=300)
    plt.close(fig)

    ########################################
    # Plot window data
    ########################################

    window_df = pd.DataFrame([])
    for model in list(top_models["model"].unique()):
        window_path = window_dir_path / f"{model}.csv"
        if not window_path.exists():
            log.warning("Window file not found: %s", window_path)
            continue
        window_id = None
        try:
            window_id = int(model.split("_")[1])
        except ValueError:
            pass
        temp_window_df = pd.read_csv(
            window_path,
            usecols=[
                "size",
                "queue_len",
                "prev_queue_len_1",
                "prev_queue_len_2",
                "prev_queue_len_3",
                "prev_latency_1",
                "prev_latency_2",
                "prev_latency_3",
                "prev_throughput_1",
                "prev_throughput_2",
                "prev_throughput_3",
            ],
        )

        temp_window_df["window"] = window_id if window_id is not None else "initial"
        window_df = pd.concat([window_df, temp_window_df])

    log.info("Columns: %s", list(window_df.columns), tab=0)

    start_time = default_timer()
    window_df = window_df.dropna()
    windows = window_df["window"].values
    data_without_window = window_df.drop(columns=["window"])
    pca = PCA(n_components=2)
    data_pca = pca.fit_transform(data_without_window)
    end_time = default_timer()
    log.info("PCA elapsed time: %s", end_time - start_time, tab=0)
    data_pca = pd.DataFrame(data_pca, columns=["x", "y"])
    data_pca["window"] = windows
    # data_tsne = fitsne(np.ascontiguousarray(data_without_window.values), nthreads=8)
    # log.info("TSNE elapsed time: %s", end_time - start_time, tab=0)

    # data_tsne = pd.DataFrame(data_tsne, columns=["x", "y"])  # type: ignore
    # data_tsne["window"] = windows

    fig, ax = plt.subplots(figsize=(10, 5))
    for window in data_pca["window"].unique():
        # log.info("Plotting window: %s", window, tab=0)
        data = data_pca[data_pca["window"] == window]
        ax.scatter(data["x"], data["y"], label=window)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Window Data")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.05), ncol=5)
    # ax.xaxis.set_major_locator(ticker.MaxNLocator(integer=True))
    fig.tight_layout()
    fig.savefig(output / f"window.png", dpi=300)
    plt.close(fig)

    ########################################
    # Plot KDE of window data
    ########################################

    n_columns = len(window_df.columns)
    n_cols = 4
    n_rows = n_columns // n_cols
    gs = GridSpec(n_rows, n_cols)

    fig = plt.figure(figsize=(20, 10))
    for i, column in enumerate(window_df.columns):
        log.info("Plotting column: %s", column, tab=0)
        col = i % n_cols
        row = i // n_cols
        ax = fig.add_subplot(gs[row, col])
        data = window_df[[column, "window"]]
        sns.kdeplot(data=data, ax=ax, x=column, hue="window")
        ax.set_title(column)
        ax.set_ylabel("Density")
        ax.set_xlabel(column)

    fig.tight_layout()
    fig.savefig(output / f"window_kde.png", dpi=300)

    global_end_time = default_timer()
    log.info("Total elapsed time: %s", global_end_time - global_start_time, tab=0)


if __name__ == "__main__":
    app()
