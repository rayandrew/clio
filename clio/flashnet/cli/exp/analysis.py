from pathlib import Path
from typing import Annotated

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns

import typer

from clio.utils.logging import LogLevel, log_global_setup
from clio.utils.timer import default_timer

app = typer.Typer(name="Exp -- Analysis", pretty_exceptions_enable=False)


@app.command()
def analysis(
    results: Annotated[
        list[Path], typer.Argument(help="The result data directory to use for prediction", exists=False, file_okay=False, dir_okay=True, resolve_path=True)
    ],
    output: Annotated[Path, typer.Option(help="The output path to write the results to")],
    log_level: Annotated[LogLevel, typer.Option(help="The log level to use")] = LogLevel.INFO,
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

    ###########################################################################
    # Analysis
    ###########################################################################

    # 1. Load the results

    dfs: dict[str, pd.DataFrame] = {}
    for result in results:
        if not result.exists() or not result.is_dir():
            continue

        algo = ""
        if "simple.initial-only" in str(result):
            algo = "initial-only"
        elif "ensemble.use-recent-model" in str(result):
            algo = "ensemble.use-recent-model"
        elif "ensemble.initial-only" in str(result):
            algo = "ensemble.initial-only"
        else:
            raise ValueError(f"Unknown result name: {result}")

        assert algo != "", "sanity check, algo should not be empty"
        assert algo not in dfs, "sanity check, algo should not be in dfs"

        log.info("Processing result: %s", result, tab=1)
        result_path = result / "results.csv"
        if not result_path.exists():
            log.warning("Skipping result: %s", result, tab=2)
            continue
        dfs[algo] = pd.read_csv(result_path)
        dfs[algo]["algo"] = algo

    df = pd.concat(dfs.values(), ignore_index=True)

    df["percent_not_confident_case"] = df["percent_lucky_case"] + df["percent_clueless_case"]
    df["percent_confident_case"] = 100 - df["percent_not_confident_case"]

    # 2. Plot the results ######################################################

    # 2.1. Plot the lineplot of AUC over time ##################################

    fig, ax = plt.subplots(figsize=(12, 3))

    sns.lineplot(data=df, x="window_id", y="auc", hue="algo", ax=ax)
    ax.set_title("AUC over Time")
    ax.set_xlabel("Window ID")
    ax.set_ylabel("AUC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "auc_over_time.png", dpi=300)
    plt.close(fig)

    # 2.2. Plot the lineplot of Accuracy over time ##############################

    fig, ax = plt.subplots(figsize=(12, 3))

    sns.lineplot(data=df, x="window_id", y="accuracy", hue="algo", ax=ax)
    ax.set_title("Accuracy over Time")
    ax.set_xlabel("Window ID")
    ax.set_ylabel("Accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "accuracy_over_time.png", dpi=300)
    plt.close(fig)

    # 2.3. Plot the barplot of AUC over algo ###################################

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(data=df, x="window_id", y="auc", hue="algo", ax=ax)
    ax.set_title("")
    ax.set_xlabel("Window ID")
    ax.set_ylabel("AUC")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "auc_over_algo.png", dpi=300)
    plt.close(fig)

    # 2.4 Average AUC over algo ################################################

    fig, ax = plt.subplots(figsize=(4, 3))
    sns.barplot(data=df, x="algo", y="auc", ax=ax)
    ax.set_title("Average AUC over Algo")
    ax.set_xlabel("Algo")
    ax.set_ylabel("AUC")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment="right")
    fig.tight_layout()
    fig.savefig(output / "average_auc_over_algo.png", dpi=300)
    plt.close(fig)

    # 2.5 Average Percent Best Case over algo ###################################

    fig, ax = plt.subplots(figsize=(4, 3))
    sns.barplot(data=df, x="algo", y="percent_best_case", ax=ax)
    ax.set_title("Average Percent Best Case over Algo")
    ax.set_xlabel("Algo")
    ax.set_ylabel("Percent Best Case")
    fig.tight_layout()
    fig.savefig(output / "average_percent_best_case_over_algo.png", dpi=300)
    plt.close(fig)

    # 2.6 Percent Worst Case + Clueless Case + Lucky Case over algo ##############

    fig, ax = plt.subplots(figsize=(4, 3))
    sns.barplot(data=df, x="algo", y="percent_worst_case", ax=ax, color="red", label="Worst Case")
    sns.barplot(data=df, x="algo", y="percent_clueless_case", ax=ax, color="blue", label="Clueless Case")
    sns.barplot(data=df, x="algo", y="percent_lucky_case", ax=ax, color="green", label="Lucky Case")
    ax.set_title("Percent Worst Case + Clueless Case + Lucky Case over Algo")
    ax.set_xlabel("Algo")
    ax.set_ylabel("Percent")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "percent_worst_clueless_lucky_case_over_algo.png", dpi=300)
    plt.close(fig)

    # 2.7 Average Percent Worst Case + Clueless Case + Lucky Case over time ######

    fig, ax = plt.subplots(figsize=(12, 3))
    df["combination_case"] = df["percent_worst_case"] + df["percent_clueless_case"] + df["percent_lucky_case"]

    sns.lineplot(data=df, x="window_id", y="combination_case", hue="algo", ax=ax)
    ax.set_title("Average Percent Worst Case + Clueless Case + Lucky Case over Algo")
    ax.set_xlabel("Window ID")
    ax.set_ylabel("Percent")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "average_percent_worst_clueless_lucky_case_over_algo.png", dpi=300)
    plt.close(fig)

    # 2.8 Average Worst Case over time ###########################################

    fig, ax = plt.subplots(figsize=(12, 3))
    sns.lineplot(data=df, x="window_id", y="percent_worst_case", hue="algo", ax=ax)
    ax.set_title("Average Percent Worst Case over Algo")
    ax.set_xlabel("Window ID")
    ax.set_ylabel("Worst Case")
    fig.tight_layout()
    fig.savefig(output / "percent_worst_case_over_time.png", dpi=300)
    plt.close(fig)

    # 2.9 Average Worst Case over algo #########################################

    fig, ax = plt.subplots(figsize=(4, 4))
    sns.barplot(data=df, x="algo", y="percent_worst_case", ax=ax)
    ax.set_title("Average Percent Worst Case over Algo")
    ax.set_xlabel("Algo")
    ax.set_ylabel("Worst Case")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment="right")
    fig.tight_layout()
    fig.savefig(output / "average_percent_worst_case_over_algo.png", dpi=300)
    plt.close(fig)

    # 2.9 Average Best Case over algo #########################################

    fig, ax = plt.subplots(figsize=(4, 4))
    sns.barplot(data=df, x="algo", y="percent_best_case", ax=ax)
    ax.set_title("Average Percent Best Case over Algo")
    ax.set_xlabel("Algo")
    ax.set_ylabel("Best Case")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment="right")
    fig.tight_layout()
    fig.savefig(output / "average_percent_best_case_over_algo.png", dpi=300)
    plt.close(fig)

    # 2.10 Average Clueless Case over algo ######################################

    fig, ax = plt.subplots(figsize=(4, 4))
    sns.barplot(data=df, x="algo", y="percent_clueless_case", ax=ax)
    ax.set_title("Average Percent Clueless Case over Algo")
    ax.set_xlabel("Algo")
    ax.set_ylabel("Clueless Case")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment="right")
    fig.tight_layout()
    fig.savefig(output / "average_percent_clueless_case_over_algo.png", dpi=300)
    plt.close(fig)

    # 2.11 Average Lucky Case over algo #########################################
    fig, ax = plt.subplots(figsize=(4, 4))
    sns.barplot(data=df, x="algo", y="percent_lucky_case", ax=ax)
    ax.set_title("Average Percent Lucky Case over Algo")
    ax.set_xlabel("Algo")
    ax.set_ylabel("Lucky Case")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment="right")
    fig.tight_layout()
    fig.savefig(output / "average_percent_lucky_case_over_algo.png", dpi=300)
    plt.close(fig)

    # 2.12 Average Not Confident Case over time #################################

    fig, ax = plt.subplots(figsize=(12, 3))
    sns.lineplot(data=df, x="window_id", y="percent_not_confident_case", hue="algo", ax=ax)
    ax.set_title("Average Percent Not Confident Case over Algo")
    ax.set_xlabel("Window ID")
    ax.set_ylabel("Not Confident Case")
    fig.tight_layout()
    fig.savefig(output / "percent_not_confident_case_over_time.png", dpi=300)
    plt.close(fig)

    # 3. Algo specific plots ##################################################

    # 3.1 Confident Case vs Not Confident Case vs Accuracy over time ##########
    base_algo_output = output / "algo"

    for algo in df["algo"].unique():
        df_algo = df[df["algo"] == algo]
        algo_output = base_algo_output / algo
        algo_output.mkdir(parents=True, exist_ok=True)

        df_algo.to_csv(algo_output / "results.csv", index=False)

    for algo in df["algo"].unique():
        df_algo = df[df["algo"] == algo]
        df_algo["accuracy"] = 100 * df_algo["accuracy"]
        algo_output = base_algo_output / algo
        algo_output.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(df_algo["window_id"], df_algo["percent_not_confident_case"], label="Not Confident Case", color="red")
        ax.plot(df_algo["window_id"], df_algo["accuracy"], label="Accuracy", color="blue")
        ax.plot(df_algo["window_id"], df_algo["percent_confident_case"], label="Confident Case", color="green")
        ax.set_title(r"Confident Case vs Not Confident Case vs Accuracy over Time" "\n" "Algo: " + algo)
        ax.set_xlabel("Window ID")
        ax.set_ylabel("Percentages")
        ax.legend()
        fig.tight_layout()
        fig.savefig(algo_output / "confident_not_confident_accuracy_over_time.png", dpi=300)
        plt.close(fig)

    ############################################################################

    global_end_time = default_timer()
    log.info("Total elapsed time: %s s", global_end_time - global_start_time, tab=0)


if __name__ == "__main__":
    app()
