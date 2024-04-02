from pathlib import Path
from typing import Annotated

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import ticker

import typer

from clio.utils.logging import LogLevel, log_global_setup
from clio.utils.timer import default_timer

app = typer.Typer(name="Exp -- Analysis", pretty_exceptions_enable=False)


@app.command()
def analysis(
    # results: Annotated[
    #     list[Path], typer.Argument(help="The result data directory to use for prediction", exists=False, file_okay=False, dir_okay=True, resolve_path=True)
    # ],
    base_result_dir: Annotated[
        Path, typer.Argument(help="The base result directory to use for prediction", exists=True, file_okay=False, dir_okay=True, resolve_path=True)
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

    results = list(base_result_dir.glob("**/results.csv"))

    log.info("Results %s", results, tab=0)

    dfs: dict[str, pd.DataFrame] = {}
    for result in results:
        if "__analysis__" in str(result):
            continue
        # if not result.exists():
        #     continue
        algo = ""
        if "single.initial-only.dropout.with-eval" in str(result):
            algo = "single.initial-only.dropout.with-eval"
        elif "single.initial-only.dropout" in str(result):
            algo = "single.initial-only.dropout"
        elif "single.initial-only" in str(result):
            algo = "single.initial-only"
        elif "single.retrain.entropy" in str(result):
            algo = "single.retrain.entropy"
        elif "single.retrain.uncertainty" in str(result):
            algo = "single.retrain.uncertainty"
        elif "single.retrain.confidence" in str(result):
            algo = "single.retrain.confidence"
        elif "single.retrain.all-data" in str(result):
            algo = "single.retrain.all-data"
        elif "multiple.admit.uncertain.dropout" in str(result):
            algo = "multiple.admit.uncertain.dropout"
        elif "multiple.admit.uncertain" in str(result):
            algo = "multiple.admit.uncertain"
        elif "multiple.admit.entropy.dropout" in str(result):
            algo = "multiple.admit.entropy.dropout"
        elif "multiple.admit.entropy" in str(result):
            algo = "multiple.admit.entropy"
        elif "ensemble.use-recent-model" in str(result):
            algo = "ensemble.use-recent-model"
        elif "ensemble.initial-only.dropout" in str(result):
            algo = "ensemble.initial-only.dropout"
        elif "ensemble.initial-only" in str(result):
            algo = "ensemble.initial-only"
        else:
            continue
            # raise ValueError(f"Unknown result name: {result}")

        assert algo != "", "sanity check, algo should not be empty"
        # log.info("Algo: %s, dfs keys: %s", algo, dfs.keys(), tab=1)
        assert algo not in dfs, "sanity check, algo should not be in dfs"

        log.info("Processing result: %s", result, tab=1)
        # result_path = result / "results.csv"
        # if not result_path.exists():
        #     log.warning("Skipping result: %s", result, tab=2)
        #     continue
        result_path = result
        dfs[algo] = pd.read_csv(result_path)
        dfs[algo]["algo"] = algo

    df = pd.concat(dfs.values(), ignore_index=True)

    df["percent_not_confident_case"] = df["percent_lucky_case"] + df["percent_clueless_case"]
    df["percent_confident_case"] = 100 - df["percent_not_confident_case"]
    df["accuracy"] = 100 * df["accuracy"]
    df["auc"] = 100 * df["auc"]
    df["uncertainty"] = 100 * df["uncertainty"]
    df["entropy"] = 100 * df["entropy"]
    df["train_overhead"] = 100 * (df["train_time"] / df["train_time"].max())
    df["train_computation_efficiency"] = 100 - df["train_overhead"]
    df["inference_time"] = df["prediction_time"] / df["num_io"]

    # generate colors for each algo
    colors = sns.color_palette("tab10", len(df["algo"].unique()))
    algo_colors = {algo: color for algo, color in zip(df["algo"].unique(), colors)}

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

    sns.lineplot(data=df, x="window_id", y="accuracy", hue="algo", ax=ax, palette=algo_colors)
    ax.set_title("Accuracy over Time")
    ax.set_xlabel("Window ID")
    ax.set_ylabel("Accuracy")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "accuracy_over_time.png", dpi=300)
    plt.close(fig)

    # 2.3. Plot the barplot of AUC over algo ###################################

    fig, ax = plt.subplots(figsize=(24, 4))
    sns.barplot(data=df, x="window_id", y="auc", hue="algo", ax=ax, palette=algo_colors)
    ax.set_title("")
    ax.set_xlabel("Window ID")
    ax.set_ylabel("AUC")
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment="right")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "auc_over_algo.png", dpi=300)
    plt.close(fig)

    # 2.4 Average AUC over algo ################################################

    fig, ax = plt.subplots(figsize=(4, 6))
    sns.barplot(data=df, x="algo", y="auc", ax=ax, palette=algo_colors)
    ax.set_title("Average AUC over Algo")
    ax.set_xlabel("Algo")
    ax.set_ylabel("AUC")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment="right")
    fig.tight_layout()
    fig.savefig(output / "average_auc_over_algo.png", dpi=300)
    plt.close(fig)

    # 2.5 Average Percent Best Case over algo ###################################

    fig, ax = plt.subplots(figsize=(4, 6))
    sns.barplot(data=df, x="algo", y="percent_best_case", ax=ax, palette=algo_colors)
    ax.set_title("Average Percent Best Case over Algo")
    ax.set_xlabel("Algo")
    ax.set_ylabel("Percent Best Case")
    fig.tight_layout()
    fig.savefig(output / "average_percent_best_case_over_algo.png", dpi=300)
    plt.close(fig)

    # 2.6 Percent Worst Case + Clueless Case + Lucky Case over algo ##############

    fig, ax = plt.subplots(figsize=(4, 6))
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

    sns.lineplot(data=df, x="window_id", y="combination_case", hue="algo", ax=ax, palette=algo_colors)
    ax.set_title("Average Percent Worst Case + Clueless Case + Lucky Case over Algo")
    ax.set_xlabel("Window ID")
    ax.set_ylabel("Percent")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "average_percent_worst_clueless_lucky_case_over_algo.png", dpi=300)
    plt.close(fig)

    # 2.8 Average Worst Case over time ###########################################

    fig, ax = plt.subplots(figsize=(12, 3))
    sns.lineplot(data=df, x="window_id", y="percent_worst_case", hue="algo", ax=ax, palette=algo_colors)
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
    sns.lineplot(data=df, x="window_id", y="percent_not_confident_case", hue="algo", ax=ax, palette=algo_colors)
    ax.set_title("Average Percent Not Confident Case over Algo")
    ax.set_xlabel("Window ID")
    ax.set_ylabel("Not Confident Case")
    fig.tight_layout()
    fig.savefig(output / "percent_not_confident_case_over_time.png", dpi=300)
    plt.close(fig)

    # 2.13 Not Confident Case vs Accuracy over time ###########################

    fig, ax = plt.subplots(figsize=(12, 3))
    sns.lineplot(data=df, x="window_id", y="percent_not_confident_case", hue="algo", ax=ax, linestyle="--", palette=algo_colors)
    sns.lineplot(data=df, x="window_id", y="accuracy", hue="algo", ax=ax, linestyle="-")
    ax.set_title("Average Percent Not Confident Case vs Accuracy over Time")
    ax.set_xlabel("Window ID")
    ax.set_ylabel("Percentages")
    labels, handles = ax.get_legend_handles_labels()
    # remove duplicates
    by_label = dict(zip(handles, labels))
    ax.legend(by_label.values(), by_label.keys())
    fig.tight_layout()
    fig.savefig(output / "percent_not_confident_case_accuracy_over_time.png", dpi=300)
    plt.close(fig)

    # 2.14 Uncertainty vs Accuracy over time ###################################

    fig, ax = plt.subplots(figsize=(12, 3))
    sns.lineplot(data=df, x="window_id", y="uncertainty", hue="algo", ax=ax, linestyle="--", palette=algo_colors)
    sns.lineplot(data=df, x="window_id", y="accuracy", hue="algo", ax=ax, linestyle="-", palette=algo_colors)
    ax.set_title("Average Uncertainty vs Accuracy over Time")
    ax.set_xlabel("Window ID")
    ax.set_ylabel("Percentages")
    labels, handles = ax.get_legend_handles_labels()
    # remove duplicates
    by_label = dict(zip(handles, labels))
    ax.legend(by_label.values(), by_label.keys())
    fig.tight_layout()
    fig.savefig(output / "uncertainty_accuracy_over_time.png", dpi=300)
    plt.close(fig)

    # 2.15 Entropy vs Accuracy over time #######################################

    fig, ax = plt.subplots(figsize=(12, 3))
    sns.lineplot(data=df, x="window_id", y="entropy", hue="algo", ax=ax, linestyle="--", palette=algo_colors)
    sns.lineplot(data=df, x="window_id", y="accuracy", hue="algo", ax=ax, linestyle="-", palette=algo_colors)
    ax.set_title("Average Entropy vs Accuracy over Time")
    ax.set_xlabel("Window ID")
    ax.set_ylabel("Percentages")
    labels, handles = ax.get_legend_handles_labels()
    # remove duplicates
    by_label = dict(zip(handles, labels))
    ax.legend(by_label.values(), by_label.keys())
    fig.tight_layout()
    fig.savefig(output / "entropy_accuracy_over_time.png", dpi=300)
    plt.close(fig)

    # 2.16 Accuracy vs Train Time ####################################

    # 2.16.1 Accuracy vs Train Time over time ##################################

    fig, ax = plt.subplots(figsize=(12, 5))
    ax2 = ax.twinx()
    # sns.lineplot(data=df, x="window_id", y="accuracy", hue="algo", ax=ax, linestyle="-")
    # sns.lineplot(data=df, x="window_id", y="train_time", hue="algo", ax=ax2, linestyle="--")
    # sns.scatterplot(data=df, x="window_id", y="train_time", hue="algo", ax=ax2, marker="o", s=50)

    # generate markers for each algo
    # markers = ["o", "s", "D", "P", "X", "v", "^", "<", ">", "1", "2", "3", "4", "8", "p", "*", "h", "H", "+", "x", "D", "d", "|", "_"]

    for i, algo in enumerate(df["algo"].unique()):
        # marker = markers[i % len(markers)]
        df_algo = df[df["algo"] == algo]
        color = algo_colors[algo]
        ax.plot(df_algo["window_id"], df_algo["accuracy"], label=algo, linestyle="-", color=color)
        ax2.plot(df_algo["window_id"], df_algo["train_time"], label=algo, linestyle="--", color=color)

    ax.set_title("Average Accuracy vs Train Time over Time")
    ax.set_xlabel("Window ID")
    ax.set_ylabel("Accuracy")
    ax2.set_ylabel("Train Time (s)")
    ax.set_ylim(0, 100)
    ax2.set_ylim(-100, int(df["train_time"].max()) + 150)
    ax2.set_yticks([i for i in range(0, int(df["train_time"].max()) + 1, 100)])
    labels, handles = ax.get_legend_handles_labels()
    # remove duplicates
    by_label = dict(zip(handles, labels))
    ax2.legend(
        by_label.values(), by_label.keys(), ncol=min(len(df["algo"].unique()), 4), loc="upper center", bbox_to_anchor=(0.5, 1.2), fancybox=False, frameon=False
    )
    ax.legend().remove()
    # show legend where solid line is accuracy and dashed line is train time
    # labels = ["Accuracy", "Train Time"]
    # handles = [plt.Line2D([0], [0], color="black", linestyle="-"), plt.Line2D([0], [0], color="black", linestyle="--")]
    # ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), fancybox=False, frameon=False, ncol=2)
    ax2.spines["right"].set_linestyle((0, (8, 5)))
    ax.spines["right"].set_linestyle((0, (8, 5)))
    fig.tight_layout()
    fig.savefig(output / "accuracy_train_time_over_time.png", dpi=300)
    plt.close(fig)

    # 2.16.2 Average Accuracy vs Train Computational Efficiency over algo ############################

    fig, ax = plt.subplots(figsize=(7, 3))
    mean_acc_train_time = df.groupby("algo")[["accuracy", "train_computation_efficiency"]].mean().reset_index()
    sns.scatterplot(data=mean_acc_train_time, x="train_computation_efficiency", y="accuracy", hue="algo", ax=ax, s=100, palette=algo_colors)
    ax.set_title("Average Accuracy vs Train Computational Efficiency")
    ax.set_xlabel("Train Computational Efficiency (%)" "\n" "(100 - Train Overhead)")
    ax.set_ylabel("Accuracy")
    ax.set_xlim(None, 101)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    labels, handles = ax.get_legend_handles_labels()
    # remove duplicates
    by_label = dict(zip(handles, labels))
    ax.legend(by_label.values(), by_label.keys(), loc="upper left", bbox_to_anchor=(1.01, 1.05), fancybox=False, frameon=False)
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment="right")
    fig.tight_layout()
    fig.savefig(output / "accuracy_train_computational_efficiency_over_algo.png", dpi=300)
    plt.close(fig)

    # 2.17 Accuracy vs Train Data Size ####################################

    fig, ax = plt.subplots(figsize=(7, 3))
    mean_acc_train_data_size = df.groupby("algo")[["accuracy", "train_data_size"]].mean().reset_index()
    sns.scatterplot(data=mean_acc_train_data_size, x="train_data_size", y="accuracy", hue="algo", ax=ax, s=100, palette=algo_colors)
    ax.set_title("Average Accuracy vs Train Data Size")
    ax.set_xlabel("Train Data Size")
    ax.set_ylabel("Accuracy")
    # ax.set_xscale("log")
    ax.set_yscale("linear")
    # ax.set_xscale("log")
    # ax.set_xticks([i for i in range(0, 101, 10)])
    # ax.set_xlim(None, 101)
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    labels, handles = ax.get_legend_handles_labels()
    # remove duplicates
    by_label = dict(zip(handles, labels))
    ax.legend(by_label.values(), by_label.keys(), loc="upper left", bbox_to_anchor=(1.05, 1.05), fancybox=False, frameon=False)
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment="right")
    fig.tight_layout()
    fig.savefig(output / "accuracy_train_data_size_over_algo.png", dpi=300)
    plt.close(fig)

    # 2.18 AUC vs Train Data Size ####################################

    fig, ax = plt.subplots(figsize=(7, 3))
    mean_auc_train_data_size = df.groupby("algo")[["auc", "train_data_size"]].mean().reset_index()
    sns.scatterplot(data=mean_auc_train_data_size, x="train_data_size", y="auc", hue="algo", ax=ax, s=100, palette=algo_colors)
    ax.set_title("Average AUC vs Train Data Size")
    ax.set_xlabel("Train Data Size")
    ax.set_ylabel("AUC")
    # ax.set_xscale("log")
    ax.set_yscale("linear")
    # ax.set_xscale("log")
    # ax.set_xticks([i for i in range(0, 101, 10)])
    # ax.set_xlim(None, 101)
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    labels, handles = ax.get_legend_handles_labels()
    # remove duplicates
    by_label = dict(zip(handles, labels))
    ax.legend(by_label.values(), by_label.keys(), loc="upper left", bbox_to_anchor=(1.05, 1.05), fancybox=False, frameon=False)
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment="right")
    fig.tight_layout()
    fig.savefig(output / "auc_train_data_size_over_algo.png", dpi=300)

    # 2.19 AUC vs Train Computational Efficiency ################################

    fig, ax = plt.subplots(figsize=(7, 3))
    mean_auc_train_computation_efficiency = df.groupby("algo")[["auc", "train_computation_efficiency"]].mean().reset_index()
    sns.scatterplot(data=mean_auc_train_computation_efficiency, x="train_computation_efficiency", y="auc", hue="algo", ax=ax, s=100, palette=algo_colors)
    ax.set_title("Average AUC vs Train Computational Efficiency")
    ax.set_xlabel("Train Computational Efficiency (%)" "\n" "(100 - Train Overhead)")
    ax.set_ylabel("AUC")
    # ax.set_xscale("log")
    ax.set_yscale("linear")
    # ax.set_xscale("log")
    # ax.set_xticks([i for i in range(0, 101, 10)])
    # ax.set_xlim(None, 101)
    # ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    labels, handles = ax.get_legend_handles_labels()
    # remove duplicates
    by_label = dict(zip(handles, labels))
    ax.legend(by_label.values(), by_label.keys(), loc="upper left", bbox_to_anchor=(1.05, 1.05), fancybox=False, frameon=False)
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment="right")
    fig.tight_layout()
    fig.savefig(output / "auc_train_computational_efficiency_over_algo.png", dpi=300)
    plt.close(fig)

    # 2.20 AUC vs Inference Time ###############################################

    # 2.20.1 AUC vs Inference Time over time ####################################

    fig, ax = plt.subplots(figsize=(12, 5))
    ax2 = ax.twinx()

    sns.lineplot(data=df, x="window_id", y="auc", hue="algo", ax=ax, linestyle="-", palette=algo_colors)
    sns.lineplot(data=df, x="window_id", y="inference_time", hue="algo", ax=ax2, linestyle="--", palette=algo_colors)
    ax.set_title("Average AUC vs Inference Time over Time")
    ax.set_xlabel("Window ID")
    ax.set_ylabel("AUC")
    ax2.set_ylabel("Inference Time (s)")
    ax.set_ylim(0, 100)
    # ax2.set_ylim(-100, int(df["inference_time"].max()) + 150)
    # ax2.set_yticks([i for i in range(0, int(df["inference_time"].max()) + 1, 1)])
    labels, handles = ax.get_legend_handles_labels()
    # remove duplicates
    by_label = dict(zip(handles, labels))
    ax2.legend(
        by_label.values(), by_label.keys(), ncol=min(len(df["algo"].unique()), 4), loc="upper center", bbox_to_anchor=(0.5, 1.3), fancybox=False, frameon=False
    )
    ax.legend().remove()
    # show legend where solid line is accuracy and dashed line is train time
    # labels = ["Accuracy", "Train Time"]
    # handles = [plt.Line2D([0], [0], color="black", linestyle="-"), plt.Line2D([0], [0], color="black", linestyle="--")]
    # ax.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.0), fancybox=False, frameon=False, ncol=2)
    ax2.spines["right"].set_linestyle((0, (8, 5)))
    ax.spines["right"].set_linestyle((0, (8, 5)))
    fig.tight_layout()
    fig.savefig(output / "auc_inference_time_over_time.png", dpi=300)
    plt.close(fig)

    # 2.20.2 Average AUC vs Inference Time over algo ############################

    fig, ax = plt.subplots(figsize=(7, 3))
    mean_auc_inference_time = df.groupby("algo")[["auc", "inference_time"]].mean().reset_index()
    sns.scatterplot(data=mean_auc_inference_time, x="inference_time", y="auc", hue="algo", ax=ax, s=100, palette=algo_colors)
    ax.set_title("Average AUC vs Inference Time over Algo")
    ax.set_xlabel("Inference Time (s)")
    ax.set_ylabel("AUC")
    # ax.set_xlim(-1, int(mean_auc_inference_time["inference_time"].max()) + 1)
    ax.set_ylim(0, 100)
    ax.set_yticks([i for i in range(0, 101, 10)])
    labels, handles = ax.get_legend_handles_labels()
    # remove duplicates
    by_label = dict(zip(handles, labels))
    ax.legend(by_label.values(), by_label.keys(), loc="upper left", bbox_to_anchor=(1.05, 1.05), fancybox=False, frameon=False)
    fig.tight_layout()
    fig.savefig(output / "auc_inference_time_over_algo.png", dpi=300)
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
        algo_output = base_algo_output / algo
        algo_output.mkdir(parents=True, exist_ok=True)

        # 3.1.1 Confident Case vs Not Confident Case vs Accuracy over time ######

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

        # 3.1.2 Confident Case vs Not Confident Case vs AUC over time ############

        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(df_algo["window_id"], df_algo["percent_not_confident_case"], label="Not Confident Case", color="red")
        ax.plot(df_algo["window_id"], df_algo["auc"], label="AUC", color="blue")
        ax.plot(df_algo["window_id"], df_algo["percent_confident_case"], label="Confident Case", color="green")
        ax.set_title(r"Confident Case vs Not Confident Case vs AUC over Time" "\n" "Algo: " + algo)
        ax.set_xlabel("Window ID")
        ax.set_ylabel("Percentages")
        ax.legend()
        fig.tight_layout()
        fig.savefig(algo_output / "confident_not_confident_auc_over_time.png", dpi=300)
        plt.close(fig)

    ############################################################################

    global_end_time = default_timer()
    log.info("Total elapsed time: %s s", global_end_time - global_start_time, tab=0)


if __name__ == "__main__":
    app()
