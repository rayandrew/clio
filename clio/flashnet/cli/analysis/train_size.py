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
    results: dict[str, pd.DataFrame] = {}
    for dataset_dir in dataset_dirs:
        train_dirs = natsorted(dataset_dir.iterdir())
        dataset_name = dataset_dir.name
        if dataset_name not in results:
            results[dataset_name] = pd.DataFrame()
        df_dataset = pd.DataFrame()
        for train_dir in train_dirs:
            if "analysis" in str(train_dir):
                continue
            train_name = train_dir.name
            train_name = train_name.split("-")[-1]
            log.info("%s", train_dir, tab=2)
            exp_results = natsorted(train_dir.glob("**/results.csv"))
            if train_name not in results[dataset_name]:
                results[dataset_name][train_name] = {}
            df = pd.DataFrame()
            for p in exp_results:
                if "analysis" in str(p):
                    continue
                # log.info("P %s", p, tab=3)
                _df = pd.read_csv(p)
                _df["window"] = list(range(1, len(_df) + 1))
                name = p.relative_to(train_dir)
                name = name.parts[0]
                name = name.split("-")[-1]
                # log.info("Name %s", name, tab=4)
                # return
                _df["rep_name"] = name

                df = pd.concat([df, _df], ignore_index=True)

            if df.empty:
                continue

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

    for k in results:
        log.info("%s (%s)", k, len(results[k]), tab=1)

    # 2.1 Plot line of AUC for each dataset
    fig, ax = plt.subplots(figsize=(4, 2.2))
    sns.set_style("whitegrid")
    sns.set_context("paper")
    sns.set_palette("tab10")

    for dataset_name in results:
        df = results[dataset_name]
        ax.plot(df["train_size"], df["auc"], label=dataset_name)

    ax.set_xlabel("")
    ax.set_ylabel("AUC")
    ax.set_title("AUC vs Train Size")
    handles, labels = ax.get_legend_handles_labels()
    leg = ax.legend(
        handles,
        labels,
        loc="upper center",
        # ncol=len(results),
        # bbox_to_anchor=(0.5, 1.3),
        # title="Dataset",
        ncol=1,
        bbox_to_anchor=(1.22, 1.0),
        fancybox=False,
        frameon=False,
    )
    fig.savefig(output / "auc_vs_train_size.png", dpi=300, bbox_inches="tight", bbox_extra_artists=[leg])
    plt.close(fig)

    global_end_time = default_timer()
    log.info("Total time taken: %s", global_end_time - global_start_time, tab=0)


if __name__ == "__main__":
    app()
