import sys
from pathlib import Path
from typing import Annotated, Literal

import pandas as pd

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import ticker

import typer

from clio.utils.logging import LogLevel, log_get, log_global_setup
from clio.utils.query import QueryExecutionException, get_query
from clio.utils.timer import default_timer

plt.rcParams.update(
    {
        "font.family": "Arial",
        "text.latex.preamble": r"""
            \usepackage{amsmath}
            \usepackage{amsfonts}
            \usepackage[dvipsnames]{xcolor}
        """,
    }
)

app = typer.Typer(name="Exp -- Analysis", pretty_exceptions_enable=False)


def model_perf_based_analysis(
    data: pd.DataFrame,
    metric: Literal["accuracy", "auc"],
    output: Path,
    algo_colors: dict[str, str],
    name: str = "",
):
    name_all_caps = name.upper()
    assert metric in ["accuracy", "auc"], "sanity check, metric should be either accuracy or auc"

    log = log_get(__name__ + "-- model_perf_based_analysis")

    label = ""
    if metric == "accuracy":
        label = "Accuracy"
    elif metric == "auc":
        label = "AUC"

    assert label != "", "sanity check, label should not be empty"

    barplot_order = data.groupby("algo")[metric].mean().sort_values(ascending=False).index
    num_algo = len(barplot_order)
    locs = list(range(num_algo))

    log.info("Model Performance Based Analysis", tab=0)
    log.info("Metric: %s", metric, tab=1)
    log.info("Plotting: ...", tab=1)

    ###########################################################################
    # 1. `metric` over time
    ###########################################################################

    log.info("%s over time...", label, tab=2)

    fig, ax = plt.subplots(figsize=(12, 3))
    sns.lineplot(data=data, x="window_id", y=metric, hue="algo", ax=ax)
    ax.set_title(f"{label}")
    ax.set_xlabel("Window ID")
    ax.set_ylabel(label)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / f"{metric}_over_time.png", dpi=300)
    plt.close(fig)

    ###########################################################################
    # 2. Barplot of `metric` for each algo for each window
    ###########################################################################

    log.info("%s for each algo for each window...", label, tab=2)

    fig, ax = plt.subplots(figsize=(24, 4))
    sns.barplot(data=data, x="window_id", y=metric, hue="algo", ax=ax, palette=algo_colors)
    ax.set_title("")
    ax.set_xlabel("Window ID")
    ax.set_ylabel(label)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / f"{metric}_each_algo_each_window.png", dpi=300)
    plt.close(fig)

    ###########################################################################
    # 3. Average `metric` over algo
    ###########################################################################

    log.info("Average %s over algo...", label, tab=2)

    fig, ax = plt.subplots(figsize=(4, 4))
    sns.barplot(data=data, x="algo", y=metric, ax=ax, palette=algo_colors, order=barplot_order, hue="algo", legend=False)
    ax.set_title(f"{name_all_caps}\n Average {label}")
    ax.set_xlabel("")
    ax.set_ylabel(label)
    ax.xaxis.set_major_locator(ticker.FixedLocator(locs))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment="right")
    fig.tight_layout()
    fig.savefig(output / f"{metric}_over_algo.png", dpi=300)
    plt.close(fig)

    ###########################################################################
    # 4. `metric` vs Train Time over time
    ###########################################################################

    log.info("%s vs Train Time over time...", label, tab=2)

    fig, ax = plt.subplots(figsize=(12, 5))
    ax2 = ax.twinx()

    for i, algo in enumerate(barplot_order):
        # marker = markers[i % len(markers)]
        df_algo = data[data["algo"] == algo]
        color = algo_colors[algo]
        ax.plot(df_algo["window_id"], df_algo[metric], label=algo, linestyle="-", color=color)
        ax2.plot(df_algo["window_id"], df_algo["train_time"], label=algo, linestyle="--", color=color)

    ax.set_title(f"Average {label} vs Train Time")
    ax.set_xlabel("Window ID")
    ax.set_ylabel(label)
    ax2.set_ylabel("Train Time (s)")
    ax.set_ylim(0, 100)
    ax2.set_ylim(-100, int(data["train_time"].max()) + 150)
    ax2.set_yticks([i for i in range(0, int(data["train_time"].max()) + 1, 100)])
    handles, labels = ax.get_legend_handles_labels()
    # remove duplicates
    by_label = dict(zip(labels, handles))
    ax2.legend(
        by_label.values(),
        by_label.keys(),
        ncol=min(num_algo, 4),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.2),
        fancybox=False,
        frameon=False,
    )
    ax.legend().remove()
    ax2.spines["right"].set_linestyle((0, (8, 5)))
    ax.spines["right"].set_linestyle((0, (8, 5)))
    fig.tight_layout()
    fig.savefig(output / f"{metric}_train_time_over_time.png", dpi=300)
    plt.close(fig)

    ###########################################################################
    # 5. Average `metric` vs Train Computational Efficiency over algo
    ###########################################################################

    log.info("average %s vs Train Computational Efficiency over algo...", label, tab=2)

    fig, ax = plt.subplots(figsize=(7, 3))
    mean_metric_train_time = data.groupby("algo")[[metric, "train_computation_efficiency"]].mean().reset_index()
    sns.scatterplot(data=mean_metric_train_time, x="train_computation_efficiency", y=metric, hue="algo", ax=ax, s=100, palette=algo_colors)
    ax.set_title(f"Average {label} vs Train Computational Efficiency")
    ax.set_xlabel("Train Computational Efficiency (%)" "\n" "(100 - Train Overhead)")
    ax.set_ylabel(label)
    ax.set_xlim(None, 101)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(5))
    handles, labels = ax.get_legend_handles_labels()
    # remove duplicates
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper left", bbox_to_anchor=(1.01, 1.05), fancybox=False, frameon=False)
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment="right")
    fig.tight_layout()
    fig.savefig(output / f"{metric}_train_computational_efficiency_over_algo.png", dpi=300)
    plt.close(fig)

    ###########################################################################
    # 6. `metric` vs Train Data Size
    ###########################################################################

    log.info("%s vs Train Data Size...", label, tab=2)

    fig, ax = plt.subplots(figsize=(7, 3))
    mean_metric_train_data_size = data.groupby("algo")[[metric, "train_data_size"]].mean().reset_index()
    sns.scatterplot(data=mean_metric_train_data_size, x="train_data_size", y=metric, hue="algo", ax=ax, s=100, palette=algo_colors)
    ax.set_title(f"Average {label} vs Average Train Data Size")
    ax.set_xlabel("Train Data Size")
    ax.get_xaxis().set_major_formatter(ticker.EngFormatter())
    ax.set_ylabel(label)
    # ax.set_xscale("log")
    ax.set_yscale("linear")
    handles, labels = ax.get_legend_handles_labels()
    # remove duplicates
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper left", bbox_to_anchor=(1.05, 1.05), fancybox=False, frameon=False)
    # ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment="right")
    fig.tight_layout()
    fig.savefig(output / f"{metric}_train_data_size_over_algo.png", dpi=300)
    plt.close(fig)

    ###########################################################################
    # 7. `metric` vs Inference Time over time
    ###########################################################################

    log.info("%s vs Inference Time over time...", label, tab=2)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax2 = ax.twinx()

    sns.lineplot(data=data, x="window_id", y=metric, hue="algo", ax=ax, linestyle="-", palette=algo_colors)
    sns.lineplot(data=data, x="window_id", y="inference_time", hue="algo", ax=ax2, linestyle="--", palette=algo_colors)
    ax.set_title(f"Average {label} vs Inference Time")
    ax.set_xlabel("Window ID")
    ax.set_ylabel(label)
    ax2.set_ylabel("Inference Time (us)")
    ax.set_ylim(0, 100)
    handles, labels = ax.get_legend_handles_labels()
    # remove duplicates
    by_label = dict(zip(labels, handles))
    leg = ax2.legend(
        by_label.values(),
        by_label.keys(),
        ncol=min(num_algo, 4),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        fancybox=False,
        frameon=False,
    )
    ax.legend().remove()
    ax2.spines["right"].set_linestyle((0, (8, 5)))
    ax.spines["right"].set_linestyle((0, (8, 5)))
    # fig.tight_layout()
    fig.savefig(output / f"{metric}_inference_time_over_time.png", dpi=300, bbox_extra_artists=(leg,), bbox_inches="tight")
    plt.close(fig)

    ###########################################################################
    # 8. `metric` vs Inference Time over algo
    ###########################################################################

    log.info("%s vs Inference Time over algo...", label, tab=2)

    fig, ax = plt.subplots(figsize=(7, 3))
    mean_auc_inference_time = data.groupby("algo")[[metric, "inference_time"]].mean().reset_index()
    sns.scatterplot(data=mean_auc_inference_time, x="inference_time", y=metric, hue="algo", ax=ax, s=100, palette=algo_colors)
    ax.set_title(f"Average {label} vs Inference Time")
    ax.set_xlabel("Inference Time (us)")
    ax.set_ylabel(label)
    # ax.set_ylim(0, 100)
    # ax.set_yticks([i for i in range(0, 101, 10)])
    handles, labels = ax.get_legend_handles_labels()
    # remove duplicates
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper left", bbox_to_anchor=(1.05, 1.05), fancybox=False, frameon=False)
    fig.tight_layout()
    fig.savefig(output / f"{metric}_inference_time_over_algo.png", dpi=300)
    plt.close(fig)

    ###########################################################################
    # 9. `metric` vs Train Time over time
    ###########################################################################

    log.info("%s vs Train Time over time...", label, tab=2)

    fig, ax = plt.subplots(figsize=(10, 3))
    ax2 = ax.twinx()

    sns.lineplot(data=data, x="window_id", y=metric, hue="algo", ax=ax, linestyle="-", palette=algo_colors)
    sns.lineplot(data=data, x="window_id", y="train_time", hue="algo", ax=ax2, linestyle="--", palette=algo_colors)
    ax.set_title(f"Average {label} vs Train Time")
    ax.set_xlabel("Window ID")
    ax.set_ylabel(label)
    ax2.set_ylabel("Train Time (s)")
    # ax.set_ylim(0, 100)
    # ax2.set_ylim(-100, int(data["train_time"].max()) + 150)
    ax2.set_yticks([i for i in range(0, int(data["train_time"].max()) + 1, 100)])
    handles, labels = ax.get_legend_handles_labels()
    # remove duplicates
    by_label = dict(zip(labels, handles))
    leg = ax2.legend(
        by_label.values(),
        by_label.keys(),
        ncol=min(num_algo, 4),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.25),
        fancybox=False,
        frameon=False,
    )
    ax.legend().remove()
    ax2.spines["right"].set_linestyle((0, (8, 5)))
    ax.spines["right"].set_linestyle((0, (8, 5)))
    fig.tight_layout()
    fig.savefig(output / f"{metric}_train_time_over_time.png", dpi=300, bbox_extra_artists=(leg,), bbox_inches="tight")
    plt.close(fig)

    ###########################################################################
    # 10. `metric` vs Train Time over algo
    ###########################################################################

    log.info("%s vs Train Time over algo...", label, tab=2)

    fig, ax = plt.subplots(figsize=(7, 3))
    mean_auc_train_time = data.groupby("algo")[[metric, "train_time"]].mean().reset_index()
    sns.scatterplot(data=mean_auc_train_time, x="train_time", y=metric, hue="algo", ax=ax, s=100, palette=algo_colors)
    ax.set_title(f"Average {label} vs Train Time")
    ax.set_xlabel("Train Time (s)")
    ax.set_ylabel(label)
    # ax.set_ylim(None, 100)
    # ax.set_yticks([i for i in range(0, 101, 10)])
    # ax.yaxis.set_major_locator(ticker.MultipleLocator(5))
    handles, labels = ax.get_legend_handles_labels()
    # remove duplicates
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys(), loc="upper left", bbox_to_anchor=(1.05, 1.05), fancybox=False, frameon=False)
    fig.tight_layout()
    fig.savefig(output / f"{metric}_train_time_over_algo.png", dpi=300)
    plt.close(fig)

    if "drift" in name:
        ###########################################################################
        # `metric` vs Multiplier over time
        ###########################################################################

        log.info("%s vs Multiplier over time...", label, tab=2)

        fig, ax = plt.subplots(figsize=(15, 3))
        ax2 = ax.twinx()

        sns.lineplot(data=data, x="window_id", y=metric, hue="algo", ax=ax, linestyle="-", palette=algo_colors)
        sns.lineplot(data=data, x="window_id", y="mult", ax=ax2, linestyle="--")
        ax.set_title(f"{name_all_caps}: Average {label} vs Multiplier ")
        ax.set_xlabel("Window ID")
        ax.set_ylabel(label)
        ax2.set_ylabel("Multiplier")
        ax.set_ylim(0, 100)
        handles, labels = ax.get_legend_handles_labels()
        # remove duplicates
        by_label = dict(zip(labels, handles))
        leg = ax2.legend(
            by_label.values(),
            by_label.keys(),
            ncol=min(num_algo, 4),
            loc="upper center",
            bbox_to_anchor=(0.5, 1.25),
            fancybox=False,
            frameon=False,
        )
        ax.legend().remove()
        ax2.spines["right"].set_linestyle((0, (8, 5)))
        ax.spines["right"].set_linestyle((0, (8, 5)))
        # fig.tight_layout()
        fig.savefig(output / f"{metric}_multiplier_over_time.png", dpi=300, bbox_extra_artists=(leg,), bbox_inches="tight")
        plt.close(fig)


def confidence_based_analysis(
    data: pd.DataFrame,
    output: Path,
    algo_colors: dict[str, str],
):

    log = log_get(__name__ + "-- confidence_based_analysis")

    barplot_order = list(data["algo"].unique())
    num_algo = len(barplot_order)
    locs = list(range(num_algo))

    log.info("Confidence Based Analysis", tab=0)
    log.info("Plotting: ...", tab=1)

    ###########################################################################
    # 1. Average Percent Best Case over algo
    ###########################################################################

    log.info("Average Percent Best Case over algo...", tab=2)

    fig, ax = plt.subplots(figsize=(4, 6))
    sns.barplot(data=data, x="algo", y="percent_best_case", ax=ax, palette=algo_colors, hue="algo", legend=False)
    ax.set_title("Average Percent Best Case over Algo")
    ax.set_xlabel("")
    ax.set_ylabel("Percent Best Case")
    fig.tight_layout()
    fig.savefig(output / "average_percent_best_case_over_algo.png", dpi=300)
    plt.close(fig)

    ###########################################################################
    # 2 Percent Worst Case + Clueless Case + Lucky Case over algo
    ###########################################################################

    log.info("Percent Worst Case + Clueless Case + Lucky Case over algo...", tab=2)

    # plt.rcParams.update(
    #     {
    #         "text.usetex": True,
    #     }
    # )

    fig, ax = plt.subplots(figsize=(3, 3))
    COLORS = ["red", "orange", "green"]
    for algo in barplot_order:
        bottom = 0.0
        for i, col in enumerate(["percent_worst_case", "percent_clueless_case", "percent_lucky_case"]):
            df_algo = data[data["algo"] == algo]
            ax.bar(df_algo["algo"], df_algo[col], bottom=bottom, label=col, color=COLORS[i])
            bottom += df_algo[col].max()

    # ax.set_title(r"Percent \textcolor{red}{Worst Case} + \textcolor{orange}{Clueless Case} + \\ \textcolor{green}{Lucky Case} over Algo")
    ax.set_title("Worst Case + Clueless Case \n+ Lucky Case over Algo")
    ax.set_xlabel("")
    ax.set_ylabel("Percent")
    ax.xaxis.set_major_locator(ticker.FixedLocator(locs))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment="right")
    labels = ["Worst Case", "Clueless Case", "Lucky Case"]
    handles = [plt.Rectangle((0, 0), 1, 1, color=COLORS[i]) for i in range(3)]
    leg = ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.01, 1.00), fancybox=False, frameon=False, ncol=1)
    # fig.tight_layout()
    fig.savefig(output / "percent_worst_clueless_lucky_case_over_algo.png", dpi=300, bbox_extra_artists=(leg,), bbox_inches="tight")
    plt.close(fig)

    # plt.rcParams.update(
    #     {
    #         "text.usetex": False,
    #     }
    # )

    ###########################################################################
    # 3. Average Percent Worst Case + Clueless Case + Lucky Case over time
    ###########################################################################

    log.info("Average Percent Worst Case + Clueless Case + Lucky Case over time...", tab=2)

    fig, ax = plt.subplots(figsize=(12, 3))
    data["combination_case"] = data["percent_worst_case"] + data["percent_clueless_case"] + data["percent_lucky_case"]

    sns.lineplot(data=data, x="window_id", y="combination_case", hue="algo", ax=ax, palette=algo_colors)
    ax.set_title("Average Percent Worst Case + Clueless Case + Lucky Case over Algo")
    ax.set_xlabel("Window ID")
    ax.set_ylabel("Percent")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "average_percent_worst_clueless_lucky_case_over_algo.png", dpi=300)
    plt.close(fig)

    ###########################################################################
    # 4. Average Worst Case over time
    ###########################################################################

    log.info("Average Worst Case over time...", tab=2)

    fig, ax = plt.subplots(figsize=(12, 3))
    sns.lineplot(data=data, x="window_id", y="percent_worst_case", hue="algo", ax=ax, palette=algo_colors)
    ax.set_title("Average Percent Worst Case over Algo")
    ax.set_xlabel("Window ID")
    ax.set_ylabel("Worst Case")
    fig.tight_layout()
    fig.savefig(output / "percent_worst_case_over_time.png", dpi=300)
    plt.close(fig)

    ###########################################################################
    # 5. Average Worst Case over algo
    ###########################################################################

    log.info("Average Worst Case over algo...", tab=2)

    fig, ax = plt.subplots(figsize=(4, 4))
    sns.barplot(data=data, x="algo", y="percent_worst_case", ax=ax, palette=algo_colors, hue="algo", legend=False)
    ax.set_title("Average Percent Worst Case over Algo")
    ax.set_xlabel("")
    ax.set_ylabel("Worst Case")
    ax.xaxis.set_major_locator(ticker.FixedLocator(locs))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment="right")
    fig.tight_layout()
    fig.savefig(output / "average_percent_worst_case_over_algo.png", dpi=300)
    plt.close(fig)

    ###########################################################################
    # 6. Average Best Case over algo
    ###########################################################################

    log.info("Average Best Case over algo...", tab=2)

    fig, ax = plt.subplots(figsize=(4, 4))
    sns.barplot(data=data, x="algo", y="percent_best_case", ax=ax, palette=algo_colors, hue="algo", legend=False)
    ax.set_title("Average Percent Best Case over Algo")
    ax.set_xlabel("")
    ax.set_ylabel("Best Case")
    ax.xaxis.set_major_locator(ticker.FixedLocator(locs))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment="right")
    fig.tight_layout()
    fig.savefig(output / "average_percent_best_case_over_algo.png", dpi=300)
    plt.close(fig)

    ###########################################################################
    # 7. Average Clueless Case over algo
    ###########################################################################

    log.info("Average Clueless Case over algo...", tab=2)

    fig, ax = plt.subplots(figsize=(4, 4))
    sns.barplot(data=data, x="algo", y="percent_clueless_case", ax=ax, palette=algo_colors, hue="algo", legend=False)
    ax.set_title("Average Percent Clueless Case over Algo")
    ax.set_xlabel("")
    ax.set_ylabel("Clueless Case")
    ax.xaxis.set_major_locator(ticker.FixedLocator(locs))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment="right")
    fig.tight_layout()
    fig.savefig(output / "average_percent_clueless_case_over_algo.png", dpi=300)
    plt.close(fig)

    ###########################################################################
    # 8. Average Lucky Case over algo
    ###########################################################################

    log.info("Average Lucky Case over algo...", tab=2)

    fig, ax = plt.subplots(figsize=(4, 4))
    sns.barplot(data=data, x="algo", y="percent_lucky_case", ax=ax, palette=algo_colors, hue="algo", legend=False)
    ax.set_title("Average Percent Lucky Case over Algo")
    ax.set_xlabel("")
    ax.set_ylabel("Lucky Case")
    ax.xaxis.set_major_locator(ticker.FixedLocator(locs))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment="right")
    fig.tight_layout()
    fig.savefig(output / "average_percent_lucky_case_over_algo.png", dpi=300)
    plt.close(fig)

    ###########################################################################
    # 9. Average Not Confident Case over time
    ###########################################################################

    log.info("Average Not Confident Case over time...", tab=2)

    fig, ax = plt.subplots(figsize=(12, 3))
    sns.lineplot(data=data, x="window_id", y="percent_not_confident_case", hue="algo", ax=ax, palette=algo_colors)
    ax.set_title("Average Percent Not Confident Case over Algo")
    ax.set_xlabel("Window ID")
    ax.set_ylabel("Not Confident Case")
    fig.tight_layout()
    fig.savefig(output / "percent_not_confident_case_over_time.png", dpi=300)
    plt.close(fig)

    ###########################################################################
    # 10. Not Confident Case vs Accuracy over time
    ###########################################################################

    log.info("Not Confident Case vs Accuracy over time...", tab=2)

    fig, ax = plt.subplots(figsize=(12, 3))
    sns.lineplot(data=data, x="window_id", y="percent_not_confident_case", hue="algo", ax=ax, linestyle="--", palette=algo_colors)
    sns.lineplot(data=data, x="window_id", y="accuracy", hue="algo", ax=ax, linestyle="-", palette=algo_colors)
    ax.set_title("Average Percent Not Confident Case vs Accuracy over Time")
    ax.set_xlabel("Window ID")
    ax.set_ylabel("Percentages")
    handles, labels = ax.get_legend_handles_labels()
    # remove duplicates
    by_label = dict(zip(labels, handles))
    ax.legend(by_label.values(), by_label.keys())
    fig.tight_layout()
    fig.savefig(output / "percent_not_confident_case_accuracy_over_time.png", dpi=300)
    plt.close(fig)


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
    query: Annotated[str, typer.Option(help="The query to filter the aths")] = "",
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

    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Analysis
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # ---------------------------------------------------------------------------
    # 1. Load the results
    # ---------------------------------------------------------------------------

    try:
        q = get_query(query)

        if q:

            def temp(p):
                # log.info(p)
                return q({"path": str(p)})

            results = [p for p in base_result_dir.glob("**/results.csv") if temp(p)]
            trace_dicts = [p for p in base_result_dir.glob("**/trace_dict.json") if temp(p)]
        else:
            results = [p for p in base_result_dir.glob("**/results.csv")]
            trace_dicts = [p for p in base_result_dir.glob("**/trace_dict.json")]
    except QueryExecutionException as e:
        log.error("Failed to execute expression: %s", e)
        sys.exit(1)

    log.info("Results %s", results, tab=0)

    dfs: dict[str, pd.DataFrame] = {}
    results = [result for result in results if "__analysis__" not in str(result)]
    trace_dicts = [trace_dict for trace_dict in trace_dicts if "__analysis__" not in str(trace_dict)]
    for result, trace_dict in zip(results, trace_dicts):
        # if not result.exists():
        #     continue
        algo = ""
        if "single.initial-only.dropout.with-eval" in str(result):
            algo = "single.initial-only.dropout.with-eval"
        elif "single.initial-only.dropout" in str(result):
            algo = "single.initial-only.dropout"
            continue
        elif "single.initial-only" in str(result):
            algo = "single.initial-only"
        elif "single.retrain.entropy" in str(result):
            algo = "single.retrain.entropy"
        elif "single.retrain.uncertainty" in str(result):
            algo = "single.retrain.uncertain"
        elif "single.retrain.confidence" in str(result):
            algo = "single.retrain.confidence"
        elif "single.retrain.window" in str(result):
            algo = "single.retrain.window"
        elif "multiple.admit.uncertain.dropout" in str(result):
            algo = "multiple.admit.uncertain.dropout"
            continue
        elif "multiple.admit.uncertain" in str(result):
            algo = "multiple.admit.uncertain"
        elif "multiple.admit.entropy.dropout" in str(result):
            algo = "multiple.admit.entropy.dropout"
            continue
        elif "multiple.admit.entropy" in str(result):
            algo = "multiple.admit.entropy"
        elif "multiple.admit.confidence.dropout" in str(result):
            algo = "multiple.admit.confidence.dropout"
            continue
        elif "multiple.admit.window" in str(result):
            algo = "multiple.admit.window"
        elif "multiple.admit.confidence" in str(result):
            algo = "multiple.admit.confidence"
        elif "ensemble.use-recent-model" in str(result):
            algo = "ensemble.use-recent-model"
        elif "ensemble.initial-only.dropout" in str(result):
            algo = "ensemble.initial-only.dropout"
        elif "ensemble.initial-only" in str(result):
            algo = "ensemble.initial-only"
        elif "matchmaker.window" in str(result):
            algo = "matchmaker.window"
        elif "matchmaker.batch" in str(result):
            algo = "matchmaker.batch"
        elif "matchmaker.single" in str(result):
            algo = "matchmaker.single"
        elif "matchmaker.scikit" in str(result):
            algo = "matchmaker.scikit"
        elif "aue.flashnet" in str(result):
            algo = "aue.flashnet"
        elif "aue.scikit" in str(result):
            algo = "aue.scikit"
        elif "driftsurf" in str(result):
            algo = "driftsurf"
        else:
            continue
            # raise ValueError(f"Unknown result name: {result}")

        assert algo != "", "sanity check, algo should not be empty"
        log.info("Algo: %s, dfs keys: %s", algo, dfs.keys(), tab=1)
        assert algo not in dfs, "sanity check, algo should not be in dfs "

        log.info("Processing result: %s", result, tab=1)
        # result_path = result / "results.csv"
        # if not result_path.exists():
        #     log.warning("Skipping result: %s", result, tab=2)
        #     continue
        result_path = result
        dfs[algo] = pd.read_csv(result_path)
        dfs[algo]["algo"] = algo

        import json

        trace_dict_read = json.load(open(trace_dict))
        trace_dict_keys = trace_dict_read.keys()
        if "-" in trace_dict_keys[0]:
            dfs[algo]["keys"] = trace_dict_keys
            multipliers = [float(k.split("-")[-1]) for k in trace_dict_keys]
            dfs[algo]["mult"] = multipliers

    df = pd.concat(dfs.values(), ignore_index=True)

    df["percent_not_confident_case"] = df["percent_lucky_case"] + df["percent_clueless_case"]
    df["percent_confident_case"] = 100 - df["percent_not_confident_case"]
    df["accuracy"] = 100 * df["accuracy"]
    df["auc"] = 100 * df["auc"]
    df["uncertainty"] = 100 * df["uncertainty"]
    df["entropy"] = 100 * df["entropy"]
    df["train_overhead"] = 100 * (df["train_time"] / df["train_time"].max())
    df["train_computation_efficiency"] = 100 - df["train_overhead"]
    df["inference_time"] = df["prediction_time"] / df["num_io"]  # seconds
    df["inference_time"] = 1e6 * df["inference_time"]  # microseconds

    # generate colors for each algo
    algos = list(df["algo"].unique())
    num_algo = len(algos)
    colors = sns.color_palette("tab10", num_algo)
    algo_colors = {algo: color for algo, color in zip(algos, colors)}
    locs = list(range(num_algo))

    # ---------------------------------------------------------------------------
    # 2. Plot the results
    # ---------------------------------------------------------------------------

    ###########################################################################
    # 2.1 Model performance-based analysis
    ###########################################################################
    import re

    # extract the path with *.nim using regex
    dir_path = str(base_result_dir)
    name = re.search(r"(\w+).nim", dir_path).group(1)

    model_perf_based_analysis(data=df, metric="accuracy", output=output, algo_colors=algo_colors, name=name)
    model_perf_based_analysis(data=df, metric="auc", output=output, algo_colors=algo_colors, name=name)

    return

    ###########################################################################
    # 2.2 Confidence-based analysis
    ###########################################################################

    confidence_based_analysis(data=df, output=output, algo_colors=algo_colors)

    ###########################################################################
    # 2.3 Uncertainty over Algo
    ###########################################################################

    fig, ax = plt.subplots(figsize=(4, 4))
    sns.barplot(data=df, x="algo", y="uncertainty", ax=ax, palette=algo_colors, hue="algo", legend=False)
    ax.set_title("Uncertainty over Algo")
    ax.set_xlabel("")
    ax.set_ylabel("Uncertainty")
    ax.xaxis.set_major_locator(ticker.FixedLocator(locs))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment="right")
    fig.tight_layout()
    fig.savefig(output / "uncertainty_over_algo.png", dpi=300)
    plt.close(fig)

    ###########################################################################
    # 2.4 Uncertainty vs Accuracy over time
    ###########################################################################

    fig, ax = plt.subplots(figsize=(12, 3))
    sns.lineplot(data=df, x="window_id", y="accuracy", hue="algo", ax=ax, linestyle="-", palette=algo_colors)
    sns.lineplot(data=df, x="window_id", y="uncertainty", hue="algo", ax=ax, linestyle="--", palette=algo_colors)
    ax.set_title("Average Uncertainty vs Accuracy over Time")
    ax.set_xlabel("Window ID")
    ax.set_ylabel("Percentages")
    handles, labels = ax.get_legend_handles_labels()
    labels = list(set(labels))
    handles = []
    for label in labels:
        handles.append(plt.Rectangle((0, 0), 1, 1, color=algo_colors[label]))
    leg = ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.01, 0.85), fancybox=False, frameon=False)
    ax.add_artist(leg)
    # create new legend to show linestyle
    labels = ["Accuracy", "Uncertainty"]
    handles = [
        plt.Line2D([0], [0], color="black", lw=2, linestyle="-"),
        plt.Line2D([0], [0], color="black", lw=2, linestyle="--"),
    ]
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.01, 1.05), fancybox=False, frameon=False)
    fig.savefig(output / "uncertainty_accuracy_over_time.png", dpi=300, bbox_extra_artists=(leg,), bbox_inches="tight")
    plt.close(fig)

    ###########################################################################
    # 2.5 Entropy over Algo
    ###########################################################################

    fig, ax = plt.subplots(figsize=(4, 4))

    sns.barplot(data=df, x="algo", y="entropy", ax=ax, palette=algo_colors, hue="algo", legend=False)
    ax.set_title("Entropy over Algo")
    ax.set_xlabel("")
    ax.set_ylabel("Entropy")
    ax.xaxis.set_major_locator(ticker.FixedLocator(locs))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment="right")
    fig.tight_layout()
    fig.savefig(output / "entropy_over_algo.png", dpi=300)
    plt.close(fig)

    ###########################################################################
    # 2.6 Entropy vs Accuracy over time
    ###########################################################################

    fig, ax = plt.subplots(figsize=(12, 3))
    sns.lineplot(data=df, x="window_id", y="accuracy", hue="algo", ax=ax, linestyle="-", palette=algo_colors)
    sns.lineplot(data=df, x="window_id", y="entropy", hue="algo", ax=ax, linestyle="--", palette=algo_colors)
    ax.set_title("Average Entropy vs Accuracy over Time")
    ax.set_xlabel("Window ID")
    ax.set_ylabel("Percentages")
    handles, labels = ax.get_legend_handles_labels()
    labels = list(set(labels))
    handles = []
    for label in labels:
        handles.append(plt.Rectangle((0, 0), 1, 1, color=algo_colors[label]))
    leg = ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.01, 0.85), fancybox=False, frameon=False)
    ax.add_artist(leg)
    # create new legend to show linestyle
    labels = ["Accuracy", "Entropy"]
    handles = [
        plt.Line2D([0], [0], color="black", lw=2, linestyle="-"),
        plt.Line2D([0], [0], color="black", lw=2, linestyle="--"),
    ]
    ax.legend(handles, labels, loc="upper left", bbox_to_anchor=(1.01, 1.05), fancybox=False, frameon=False)
    fig.savefig(output / "entropy_accuracy_over_time.png", dpi=300, bbox_extra_artists=(leg,), bbox_inches="tight")
    plt.close(fig)

    ###########################################################################
    # 2.7 Inference Time over algo
    ###########################################################################

    fig, ax = plt.subplots(figsize=(4, 4))
    sns.barplot(data=df, x="algo", y="inference_time", ax=ax, palette=algo_colors, hue="algo", legend=False)
    ax.set_title("Inference Time")
    ax.set_xlabel("")
    ax.set_ylabel("Inference Time (us)")
    ax.xaxis.set_major_locator(ticker.FixedLocator(locs))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment="right")
    fig.tight_layout()
    fig.savefig(output / "inference_time_over_algo.png", dpi=300)
    plt.close(fig)

    ###########################################################################
    # 2.8 Train Time over algo
    ###########################################################################

    fig, ax = plt.subplots(figsize=(4, 4))
    sns.barplot(data=df, x="algo", y="train_time", ax=ax, palette=algo_colors, hue="algo", legend=False)
    ax.set_title("Train Time")
    ax.set_xlabel("")
    ax.set_ylabel("Train Time (s)")
    ax.xaxis.set_major_locator(ticker.FixedLocator(locs))
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, horizontalalignment="right")
    fig.tight_layout()
    fig.savefig(output / "train_time_over_algo.png", dpi=300)
    plt.close(fig)

    # ---------------------------------------------------------------------------
    # 3. Algo specific plots
    # ---------------------------------------------------------------------------

    ###########################################################################
    # 3.1 Confident Case vs Not Confident Case vs Accuracy over time
    ###########################################################################

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
