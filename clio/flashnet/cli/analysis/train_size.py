import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated, Literal

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import ticker

from natsort import natsorted

from clio.utils.logging import LogLevel, log_get, log_global_setup
from clio.utils.query import QueryExecutionException, get_query
from clio.utils.timer import default_timer, timeit
from clio.utils.typer import Typer, typer

plt.rcParams.update(
    {
        "font.family": "Arial",
        # "text.latex.preamble": r"""
        #     \usepackage{amsmath}
        #     \usepackage{amsfonts}
        #     \usepackage[dvipsnames]{xcolor}
        # """,
    }
)

app = Typer(name="Exp -- Analysis -- Train Size", pretty_exceptions_enable=False)

# @dataclass
# class Result:
#     dataset:

# @dataclass
# class Result:


@app.command()
def analyze(
    base_result_dir: Annotated[
        Path, typer.Argument(help="The base result directory to use for prediction", exists=True, file_okay=False, dir_okay=True, resolve_path=True)
    ],
    output: Annotated[Path, typer.Option(help="The output path to write the results to")],
    log_level: Annotated[LogLevel, typer.Option(help="The log level to use")] = LogLevel.INFO,
):
    # return
    args = locals()

    global_start_time = default_timer()

    output.mkdir(parents=True, exist_ok=True)
    log = log_global_setup(output / "log.txt", level=log_level)
    log = log_global_setup()

    log.info("Args", tab=0)
    for arg in args:
        log.info("%s: %s", arg, args[arg], tab=1)

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Analysis
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # ---------------------------------------------------------------------------
    # 1. Load the results
    # ---------------------------------------------------------------------------

    dataset_dirs = list(filter(lambda p: "analysis" not in p.name, list(base_result_dir.iterdir())))
    log.info("Results dirs", tab=0)
    # results: dict[str, dict[str, list[pd.DataFrame]]] = {}
    # results: dict[str, dict[str, pd.DataFrame]] = {}
    results: dict[str, pd.DataFrame] = {}
    for dataset_dir in dataset_dirs:
        log.info("%s", dir, tab=1)
        train_dirs = natsorted(dataset_dir.iterdir())
        dataset_name = dataset_dir.name
        if dataset_name not in results:
            results[dataset_name] = pd.DataFrame()
        df_dataset = pd.DataFrame()
        for train_dir in train_dirs:
            if "analysis" in str(train_dir):
                continue
            train_name = train_dir.name
            log.info("%s", train_dir, tab=2)
            exp_results = natsorted(train_dir.glob("**/results.csv"))
            if train_name not in results[dataset_name]:
                results[dataset_name][train_name] = {}
            df = pd.DataFrame()
            for p in exp_results:
                if "analysis" in str(p):
                    continue
                log.info("P %s", p, tab=3)
                _df = pd.read_csv(p)
                _df["window"] = list(range(1, len(_df) + 1))
                name = p.relative_to(train_dir)
                name = name.parts[0]
                name = name.split("-")[-1]
                _df["rep_name"] = int(name)

                df = pd.concat([df, _df], ignore_index=True)

            df = df[df["type"] == "window"]
            exclude_cols = df.columns.difference(["window", "rep_name", "dataset", "group", "type"]).tolist()
            df_agg = df.groupby("window")[exclude_cols].mean()
            df_agg = df_agg.reset_index()
            df_agg["train_size"] = train_name
            # results[dataset_name][train_name] = df_agg
            df_dataset = pd.concat([df_dataset, df_agg], ignore_index=True)

        df_dataset_agg = df_dataset.groupby("train_size").mean()
        df_dataset_agg = df_dataset_agg.reindex(natsorted(df_dataset_agg.index))
        df_dataset_agg = df_dataset_agg.reset_index()
        results[dataset_name] = df_dataset_agg

    # ---------------------------------------------------------------------------
    # 2. Plot the results
    # ---------------------------------------------------------------------------

    # log.info("Results %s", results.keys(), tab=0)
    for k in results:
        # log.info("%s", k, tab=1)
        log.info("%s (%s)", k, len(results[k]), tab=1)
        # for k2 in results[k]:
        #     log.info("%s (%s)", k2, len(results[k][k2]), tab=2)

    # 2.1 Plot line of AUC for each dataset
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.set_style("whitegrid")
    sns.set_context("paper")
    sns.set_palette("tab10")

    # y axis is auc
    # x axis will be train_time
    # hue is dataset
    x = list(results.keys())

    # lineplot_data = {
    #     "x": list(range(0, len(x))),
    #     "y": [],
    # }

    for dataset_name in results:
        df = results[dataset_name]
        ax.plot(df["train_size"], df["auc"], label=dataset_name)

    ax.set_xlabel("Train Size")
    ax.set_ylabel("AUC")
    ax.set_title("AUC vs Train Size")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "auc_vs_train_size.png", dpi=300)
    plt.close(fig)

    global_end_time = default_timer()
    log.info("Total time taken: %s", global_end_time - global_start_time, tab=0)


if __name__ == "__main__":
    app()
