import json
import shutil
import sys
from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
from matplotlib.gridspec import GridSpec

import torch

import typer

import clio.flashnet.training.simple as flashnet_simple
from clio.flashnet.confidence import get_confidence_cases
from clio.flashnet.eval import flashnet_evaluate
from clio.flashnet.normalization import get_cached_norm, save_norm

from clio.utils.cpu_usage import CPUUsage
from clio.utils.dataframe import append_to_df
from clio.utils.general import parse_time, torch_set_seed
from clio.utils.indented_file import IndentedFile
from clio.utils.logging import LogLevel, log_get, log_global_setup
from clio.utils.timer import Timer, default_timer
from clio.utils.trace_pd import TraceWindowGeneratorContext, trace_get_dataset_paths, trace_time_window_generator

app = typer.Typer(name="Impact", pretty_exceptions_enable=False)
_log = log_get(__name__)


# def frame_image(img: np.ndarray, frame_width: int = 10) -> np.ndarray:
#     b = frame_width  # border size in pixel
#     ny, nx = img.shape[0], img.shape[1]  # resolution / number of pixels in x and y
#     if img.ndim == 3:  # rgb or rgba array
#         framed_img = np.zeros((b + ny + b, b + nx + b, img.shape[2]))
#     elif img.ndim == 2:  # grayscale image
#         framed_img = np.zeros((b + ny + b, b + nx + b))
#     framed_img[b:-b, b:-b] = img
#     return framed_img


def highlight_cell(x: float, y: float, ax=None, **kwargs):
    rect = plt.Rectangle((x - 0.5, y - 0.5), 1, 1, fill=False, **kwargs)
    ax = ax or plt.gca()
    ax.add_patch(rect)
    return rect


@app.command()
def exp(
    data_dir: Annotated[
        Path, typer.Argument(help="The test data directory to use for prediction", exists=True, file_okay=False, dir_okay=True, resolve_path=True)
    ],
    output: Annotated[Path, typer.Option(help="The output path to write the results to")],
    # window_size: Annotated[str, typer.Option(help="The window size to use for prediction (in minute(s))", show_default=True)] = "10",
    log_level: Annotated[LogLevel, typer.Option(help="The log level to use")] = LogLevel.INFO,
    profile_name: Annotated[str, typer.Option(help="The profile name to use for prediction", show_default=True)] = "profile_v1",
    feat_name: Annotated[str, typer.Option(help="The feature name to use for prediction", show_default=True)] = "feat_v6_ts",
    learning_rate: Annotated[float, typer.Option(help="The learning rate to use for training", show_default=True)] = 0.0001,
    epochs: Annotated[int, typer.Option(help="The number of epochs to use for training", show_default=True)] = 20,
    batch_size: Annotated[int, typer.Option(help="The batch size to use for training", show_default=True)] = 32,
    prediction_batch_size: Annotated[int, typer.Option(help="The batch size to use for prediction", show_default=True)] = -1,
    # duration: Annotated[str, typer.Option(help="The duration to use for prediction (in minute(s))", show_default=True)] = "-1",
    seed: Annotated[int, typer.Option(help="The seed to use for random number generation", show_default=True)] = 3003,
    cuda: Annotated[int, typer.Option(help="Use CUDA for training and prediction", show_default=True)] = 0,
    threshold: Annotated[float, typer.Option(help="The threshold to use for prediction", show_default=True)] = 0.5,
    confidence_threshold: Annotated[float, typer.Option(help="The confidence threshold to use for prediction", show_default=True)] = 0.1,
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

    if prediction_batch_size < 0:
        prediction_batch_size = batch_size

    ###########################################################################
    # PIPELINE
    ###########################################################################

    trace_dict_path = data_dir / "trace_dict.json"
    if trace_dict_path.exists():
        # copy to output
        trace_dict_output_path = output / "trace_dict.json"
        shutil.copy(trace_dict_path, trace_dict_output_path)

    results = pd.DataFrame()

    #######################
    ## PREDICTION WINDOW ##
    #######################

    torch_set_seed(seed)
    device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() and cuda >= 0 else "cpu")

    model: torch.nn.Module = None

    base_model_dir = output / "models"
    base_model_dir.mkdir(parents=True, exist_ok=True)

    ifile = IndentedFile(output / "stats.txt")

    for i, data_path in enumerate(data_paths):
        log.info("Processing dataset: %s", data_path, tab=1)
        data = pd.read_csv(data_path)
        log.info("Length of data: %s", len(data), tab=2)
        if i == 0:
            ## Use filtered data in training
            if "filter" not in str(data_path):
                data_path = str(data_path).replace(profile_name, profile_name+"_filter")
                data_path = Path(data_path)
            log.info("TRANING PATH" + str(data_path))
            data = pd.read_csv(data_path)
            log.info("Training", tab=1)

            train_cpu_usage = CPUUsage()
            train_cpu_usage.update()
            model_dir = base_model_dir / f"window_{i}"
            model_path = model_dir / "model.pt"
            norm_mean, norm_std = get_cached_norm(model_dir)

            if not model_path.exists():
                model_dir.mkdir(parents=True, exist_ok=True)
                with Timer(name="Pipeline -- Initial Model Training -- Window %d" % i) as timer:
                    train_result = flashnet_simple.flashnet_train(
                        model_path=model_path,
                        dataset=data,
                        retrain=False,
                        batch_size=batch_size,
                        prediction_batch_size=prediction_batch_size,
                        # tqdm=True,
                        lr=learning_rate,
                        epochs=epochs,
                        norm_mean=norm_mean,
                        norm_std=norm_std,
                        n_data=None,
                        device=device,
                    )
                train_cpu_usage.update()
                log.info("Pipeline Initial Model")
                log.info("Elapsed time: %s", timer.elapsed, tab=2)
                log.info("CPU Usage: %s", train_cpu_usage.result, tab=2)
                log.info("AUC: %s", train_result.auc, tab=2)
                # log.info("Train Result: %s", train_result, tab=2)

                assert len(data) == train_result.num_io, "sanity check, number of data should be the same as the number of input/output"

                confidence_result = get_confidence_cases(
                    labels=train_result.labels,
                    predictions=train_result.predictions,
                    probabilities=train_result.probabilities,
                    threshold=threshold,
                    confidence_threshold=confidence_threshold,
                )

                with ifile.section("Window 0"):
                    with ifile.section("Evaluation"):
                        train_result.to_indented_file(ifile)
                    with ifile.section("Confidence Analysis"):
                        confidence_result.to_indented_file(ifile)

                results = append_to_df(
                    df=results,
                    data={
                        **train_result.eval_dict(),
                        "num_io": len(data),
                        "num_reject": len(data[data["reject"] == 1]),
                        "elapsed_time": timer.elapsed,
                        "train_time": train_result.train_time,
                        "prediction_time": train_result.prediction_time,
                        "type": "window",
                        "window_id": i,
                        "cpu_usage": train_cpu_usage.result,
                        "model_selection_time": 0.0,
                        "model": f"window_{i}",
                        "dataset": data_path.name,
                        **confidence_result.as_dict(),
                    },
                )
                assert train_result.model_path == model_path, "sanity check, model path should be the same as the initial model path"
                model = flashnet_simple.load_model(model_path, device=device)
                continue
            else:
                log.info("Model %s already trained, reusing it...", model_path, tab=2)
                model = flashnet_simple.load_model(model_path, device=device)

        #######################
        ## PREDICTION WINDOW ##
        #######################

        log.info("Predicting %s", data_path, tab=1)

        with Timer(name="Pipeline -- Window %s" % i) as window_timer:
            #######################
            ##     PREDICTION    ##
            #######################

            predict_cpu_usage = CPUUsage()
            predict_cpu_usage.update()
            with Timer(name="Pipeline -- Prediction -- Window %s" % i) as pred_timer:
                prediction_result = flashnet_simple.flashnet_predict(
                    model,
                    data,
                    batch_size=prediction_batch_size,
                    device=device,
                    threshold=threshold,
                )
            predict_cpu_usage.update()
            prediction_time = pred_timer.elapsed
            log.info("Prediction", tab=2)
            log.info("Time elapsed: %s", prediction_time, tab=3)
            log.info("CPU Usage: %s", predict_cpu_usage.result, tab=3)

            #######################
            ## ASSESS CONFIDENCE ##
            #######################

            confidence_result = get_confidence_cases(
                labels=prediction_result.labels,
                predictions=prediction_result.predictions,
                probabilities=prediction_result.probabilities,
                confidence_threshold=confidence_threshold,
                threshold=threshold,
            )
            log.info("Confidence", tab=2)
            log.info("Best Case: %s", confidence_result.best_case, tab=3)
            log.info("Worst Case: %s", confidence_result.worst_case, tab=3)
            log.info("Clueless Case: %s", confidence_result.clueless_case, tab=3)
            log.info("Lucky Case: %s", confidence_result.lucky_case, tab=3)

            #######################
            ##     EVALUATION    ##
            #######################

            eval_cpu_usage = CPUUsage()
            eval_cpu_usage.update()
            with Timer(name="Pipeline -- Evaluation -- Window %s" % i) as eval_timer:
                eval_result = flashnet_evaluate(
                    labels=prediction_result.labels,
                    predictions=prediction_result.predictions,
                    probabilities=prediction_result.probabilities,
                )
            eval_cpu_usage.update()
            log.info("Evaluation", tab=2)
            log.info("Accuracy: %s", eval_result.accuracy, tab=3)
            log.info("AUC: %s", eval_result.auc, tab=3)
            log.info("Time elapsed: %s", eval_timer.elapsed, tab=3)
            log.info("CPU Usage: %s", eval_cpu_usage.result, tab=3)

            with ifile.section(f"Window {i}"):
                with ifile.section("Evaluation"):
                    eval_result.to_indented_file(ifile)
                with ifile.section("Confidence Analysis"):
                    confidence_result.to_indented_file(ifile)

        results = append_to_df(
            df=results,
            data={
                **eval_result.as_dict(),
                "num_io": len(data),
                "num_reject": len(data[data["reject"] == 1]),
                "elapsed_time": window_timer.elapsed,
                "train_time": 0.0,
                "prediction_time": pred_timer.elapsed,
                "type": "window",
                "window_id": i,
                "cpu_usage": predict_cpu_usage.result,
                "model_selection_time": 0.0,
                "model": f"window_0",
                "dataset": data_path.name,
                **confidence_result.as_dict(),
            },
        )

        results.to_csv(output / "results.csv", index=False)

    results.to_csv(output / "results.csv", index=False)
    ifile.close()

    global_end_time = default_timer()
    log.info("Total elapsed time: %s s", global_end_time - global_start_time, tab=0)


@app.command()
def analyze(
    result_dir: Annotated[Path, typer.Argument(help="The result directory to analyze", exists=False, file_okay=True, dir_okay=True, resolve_path=True)],
    output: Annotated[Path, typer.Option(help="The output path to write the results to")],
    log_level: Annotated[LogLevel, typer.Option(help="The log level to use")] = LogLevel.INFO,
    no_multiplier: Annotated[bool, typer.Option(help="Do not use multiplier", show_default=True)] = False,
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

    multipliers: list[str] = []
    if no_multiplier:
        col = result_dir.name
    else:
        trace_dict_path = result_dir / "trace_dict.json"
        if trace_dict_path.exists():
            col = result_dir.name
            with open(trace_dict_path, "r") as f:
                trace_dict = json.load(f)
                multipliers = list(trace_dict.keys())
        else:
            col = result_dir.name[: result_dir.name.find(".")]
            # infer from name
            # find first dot
            name_without_col = result_dir.name[result_dir.name.find(".") + 1 :]
            multipliers = name_without_col.split("_")
        if "pool" not in str(trace_dict_path):
            multipliers = [f"{m}x" for m in multipliers]

    log.info("Multipliers %s", multipliers)

    df_result = pd.read_csv(result_dir / "results.csv")
    if no_multiplier:
        multipliers = [i for i in range(len(df_result))]

    assert len(df_result) == len(multipliers), "sanity check, length of result should be the same as the length of multipliers"

    df_result["multiplier"] = multipliers

    palette = ["orange"]
    for _ in range(len(multipliers) - 1):
        palette.append("blue")

    # PLOT BAR
    for metric in ["accuracy", "auc"]:
        fig, ax = plt.subplots(figsize=(4.3, 3))
        sns.barplot(data=df_result, x="multiplier", y=metric, hue="multiplier", ax=ax, palette=palette)
        ax.set_title(
            "\n".join(
                [
                    f"{metric.upper()} on Multiple Windows" if no_multiplier else f"{metric.upper()} vs Multiplier",
                    f"Description: {col}",
                ]
            )
        )
        legend = ax.legend()
        if legend:
            legend.remove()
        # ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
        ax.set_xlabel("")
        ax.set_ylabel(metric.upper())
        fig.tight_layout()
        fig.savefig(output / f"{metric}_vs_multiplier.png", dpi=300)
        plt.close(fig)

    # df_result["percent_low_confidence"] = df_result["percent_low_confidence"] * 100
    # df_result["percent_low_confidence"] = df_result["percent_low_confidence"].round(2)
    # df_result["percent_high_confidence"] = df_result["percent_low_confidence"].apply(lambda x: 100 - x)

    # HIGH CONFIDENCE

    fig, ax = plt.subplots(figsize=(4.3, 3))
    sns.barplot(data=df_result, x="multiplier", y="percent_high_confidence", hue="multiplier", ax=ax, palette=palette)
    ax.set_title(
        "\n".join(
            [
                "Percentage of High Confidence Data vs Multiplier",
                f"Description: {col}",
            ]
        )
    )
    legend = ax.legend()
    if legend:
        legend.remove()
    ax.set_xlabel("")
    ax.set_ylabel("Percentage of High Confidence Data (%)")
    fig.tight_layout()
    fig.savefig(output / "percent_high_confidence_vs_multiplier.png", dpi=300)
    plt.close(fig)

    # LOW CONFIDENCE
    fig, ax = plt.subplots(figsize=(4.3, 3))
    sns.barplot(data=df_result, x="multiplier", y="percent_low_confidence", hue="multiplier", ax=ax, palette=palette)
    ax.set_title(
        "\n".join(
            [
                "Percentage of Low Confidence Data vs Multiplier",
                f"Description: {col}",
            ]
        )
    )
    legend = ax.legend()
    if legend:
        legend.remove()
    ax.set_xlabel("")
    ax.set_ylabel("Percentage of Low Confidence Data (%)")
    fig.tight_layout()
    fig.savefig(output / "percent_low_confidence_vs_multiplier.png", dpi=300)
    plt.close(fig)

    # QUADRANT PLOT for each window
    window_dir = output / "window"
    for i in range(len(multipliers)):
        curr_dir = window_dir / f"window_{i}"
        curr_dir.mkdir(parents=True, exist_ok=True)
        df = df_result[df_result["multiplier"] == multipliers[i]]
        fig, ax = plt.subplots(figsize=(4.3, 3))

        # 4 quadrant:
        #######################
        # 1 | 2
        # -----
        # 3 | 4
        #######################

        # 1. Low Confidence, Incorrect (lucky case)
        # 2. High Confidence, Correct (best case)
        # 3. Low Confidence, Incorrect (clueless case)
        # 4. High Confidence, Incorrect (worst case)

        # confusion matrix-like
        lucky_case = df["lucky_case"].values[0]
        clueless_case = df["clueless_case"].values[0]
        worst_case = df["worst_case"].values[0]
        best_case = df["best_case"].values[0]
        matrix = np.array([[lucky_case, best_case], [clueless_case, worst_case]])

        im = ax.imshow(
            matrix,
            cmap="Blues",
            interpolation="nearest",
            # alpha=0.1,
        )
        cmap_min, cmap_max = im.cmap(0), im.cmap(1.0)
        thresh = (matrix.max() + matrix.min()) / 2.0

        # 1. Low Confidence, Incorrect (lucky case)
        color = cmap_max if lucky_case < thresh else cmap_min
        ax.text(0.0, 0.0, lucky_case, va="center", ha="center", fontsize=12, color=color)
        # highlight_cell(0, 0, ax, edgecolor="black", linewidth=2)
        # 2. High Confidence, Correct (best case)
        color = cmap_max if best_case < thresh else cmap_min
        ax.text(1.0, 0.0, best_case, va="center", ha="center", fontsize=12, color=color)
        # highlight_cell(1, 0, ax, edgecolor="black", linewidth=2)
        # 3. Low Confidence, Incorrect (clueless case)
        color = cmap_max if clueless_case < thresh else cmap_min
        ax.text(0.0, 1.0, clueless_case, va="center", ha="center", fontsize=12, color=color)
        # highlight_cell(0, 1, ax, edgecolor="black", linewidth=2)
        # # 4. High Confidence, Incorrect (worst case)
        color = cmap_max if worst_case < thresh else cmap_min
        ax.text(1.0, 1.0, worst_case, va="center", ha="center", fontsize=12, color=color)
        # highlight_cell(1, 1, ax, edgecolor="black", linewidth=2)

        ax.set_title(
            "\n".join(
                [
                    f"Confidence Cases for Window {i}",
                    f"Description: {col}",
                ]
            )
        )
        ax.set_xlabel("Confidence")

        # color bar

        ax.set_xlabel("Confidence")
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Low", "High"])
        ax.set_ylabel("Performance")
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["High", "Low"])

        fig.colorbar(im, ax=ax)

        fig.tight_layout()
        fig.savefig(curr_dir / "quadrant_plot.png", dpi=300)
        plt.close(fig)

    global_end_time = default_timer()
    log.info("Total elapsed time: %s s", global_end_time - global_start_time, tab=0)


@app.command()
def exp_ensemble(
    data_dir: Annotated[
        Path, typer.Argument(help="The test data directory to use for prediction", exists=True, file_okay=False, dir_okay=True, resolve_path=True)
    ],
    output: Annotated[Path, typer.Option(help="The output path to write the results to")],
    # window_size: Annotated[str, typer.Option(help="The window size to use for prediction (in minute(s))", show_default=True)] = "10",
    log_level: Annotated[LogLevel, typer.Option(help="The log level to use")] = LogLevel.INFO,
    profile_name: Annotated[str, typer.Option(help="The profile name to use for prediction", show_default=True)] = "profile_v1_filter",
    feat_name: Annotated[str, typer.Option(help="The feature name to use for prediction", show_default=True)] = "feat_v6_ts",
    learning_rate: Annotated[float, typer.Option(help="The learning rate to use for training", show_default=True)] = 0.0001,
    epochs: Annotated[int, typer.Option(help="The number of epochs to use for training", show_default=True)] = 20,
    batch_size: Annotated[int, typer.Option(help="The batch size to use for training", show_default=True)] = 32,
    prediction_batch_size: Annotated[int, typer.Option(help="The batch size to use for prediction", show_default=True)] = -1,
    # duration: Annotated[str, typer.Option(help="The duration to use for prediction (in minute(s))", show_default=True)] = "-1",
    seed: Annotated[int, typer.Option(help="The seed to use for random number generation", show_default=True)] = 3003,
    cuda: Annotated[int, typer.Option(help="Use CUDA for training and prediction", show_default=True)] = 0,
    threshold: Annotated[float, typer.Option(help="The threshold to use for prediction", show_default=True)] = 0.5,
    confidence_threshold: Annotated[float, typer.Option(help="The confidence threshold to use for prediction", show_default=True)] = 0.1,
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

    if prediction_batch_size < 0:
        prediction_batch_size = batch_size

    ###########################################################################
    # PIPELINE
    ###########################################################################

    trace_dict_path = data_dir / "trace_dict.json"
    if trace_dict_path.exists():
        # copy to output
        trace_dict_output_path = output / "trace_dict.json"
        shutil.copy(trace_dict_path, trace_dict_output_path)

    results = pd.DataFrame()

    #######################
    ## PREDICTION WINDOW ##
    #######################

    torch_set_seed(seed)
    device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() and cuda >= 0 else "cpu")

    model: torch.nn.Module = None

    base_model_dir = output / "models"
    base_model_dir.mkdir(parents=True, exist_ok=True)

    ifile = IndentedFile(output / "stats.txt")

    for i, data_path in enumerate(data_paths):
        log.info("Processing dataset: %s", data_path, tab=1)
        data = pd.read_csv(data_path)
        log.info("Length of data: %s", len(data), tab=2)
        if i == 0:
            log.info("Training", tab=1)

            train_cpu_usage = CPUUsage()
            train_cpu_usage.update()
            model_dir = base_model_dir / f"window_{i}"
            model_path = model_dir / "model.pt"
            norm_mean, norm_std = get_cached_norm(model_dir)

            if not model_path.exists():
                model_dir.mkdir(parents=True, exist_ok=True)
                with Timer(name="Pipeline -- Initial Model Training -- Window %d" % i) as timer:
                    train_result = flashnet_simple.flashnet_train(
                        model_path=model_path,
                        dataset=data,
                        retrain=False,
                        batch_size=batch_size,
                        prediction_batch_size=prediction_batch_size,
                        # tqdm=True,
                        lr=learning_rate,
                        epochs=epochs,
                        norm_mean=norm_mean,
                        norm_std=norm_std,
                        n_data=None,
                        device=device,
                    )
                train_cpu_usage.update()
                log.info("Pipeline Initial Model")
                log.info("Elapsed time: %s", timer.elapsed, tab=2)
                log.info("CPU Usage: %s", train_cpu_usage.result, tab=2)
                log.info("AUC: %s", train_result.auc, tab=2)
                # log.info("Train Result: %s", train_result, tab=2)

                assert len(data) == train_result.num_io, "sanity check, number of data should be the same as the number of input/output"

                confidence_result = get_confidence_cases(
                    labels=train_result.labels,
                    predictions=train_result.predictions,
                    probabilities=train_result.probabilities,
                    threshold=threshold,
                    confidence_threshold=confidence_threshold,
                )

                with ifile.section("Window 0"):
                    with ifile.section("Evaluation"):
                        train_result.to_indented_file(ifile)
                    with ifile.section("Confidence Analysis"):
                        confidence_result.to_indented_file(ifile)

                results = append_to_df(
                    df=results,
                    data={
                        **train_result.eval_dict(),
                        "num_io": len(data),
                        "num_reject": len(data[data["reject"] == 1]),
                        "elapsed_time": timer.elapsed,
                        "train_time": train_result.train_time,
                        "prediction_time": train_result.prediction_time,
                        "type": "window",
                        "window_id": i,
                        "cpu_usage": train_cpu_usage.result,
                        "model_selection_time": 0.0,
                        "model": f"window_{i}",
                        "dataset": data_path.name,
                        **confidence_result.as_dict(),
                    },
                )
                assert train_result.model_path == model_path, "sanity check, model path should be the same as the initial model path"
                model = flashnet_simple.load_model(model_path, device=device)
                continue
            else:
                log.info("Model %s already trained, reusing it...", model_path, tab=2)
                model = flashnet_simple.load_model(model_path, device=device)

        #######################
        ## PREDICTION WINDOW ##
        #######################

        log.info("Predicting %s", data_path, tab=1)

        with Timer(name="Pipeline -- Window %s" % i) as window_timer:
            #######################
            ##     PREDICTION    ##
            #######################

            predict_cpu_usage = CPUUsage()
            predict_cpu_usage.update()
            with Timer(name="Pipeline -- Prediction -- Window %s" % i) as pred_timer:
                prediction_result = flashnet_simple.flashnet_predict(
                    model,
                    data,
                    batch_size=prediction_batch_size,
                    device=device,
                    threshold=threshold,
                )
            predict_cpu_usage.update()
            prediction_time = pred_timer.elapsed
            log.info("Prediction", tab=2)
            log.info("Time elapsed: %s", prediction_time, tab=3)
            log.info("CPU Usage: %s", predict_cpu_usage.result, tab=3)

            #######################
            ## ASSESS CONFIDENCE ##
            #######################

            confidence_result = get_confidence_cases(
                labels=prediction_result.labels,
                predictions=prediction_result.predictions,
                probabilities=prediction_result.probabilities,
                confidence_threshold=confidence_threshold,
                threshold=threshold,
            )
            log.info("Confidence", tab=2)
            log.info("Best Case: %s", confidence_result.best_case, tab=3)
            log.info("Worst Case: %s", confidence_result.worst_case, tab=3)
            log.info("Clueless Case: %s", confidence_result.clueless_case, tab=3)
            log.info("Lucky Case: %s", confidence_result.lucky_case, tab=3)

            #######################
            ##     EVALUATION    ##
            #######################

            eval_cpu_usage = CPUUsage()
            eval_cpu_usage.update()
            with Timer(name="Pipeline -- Evaluation -- Window %s" % i) as eval_timer:
                eval_result = flashnet_evaluate(
                    labels=prediction_result.labels,
                    predictions=prediction_result.predictions,
                    probabilities=prediction_result.probabilities,
                )
            eval_cpu_usage.update()
            log.info("Evaluation", tab=2)
            log.info("Accuracy: %s", eval_result.accuracy, tab=3)
            log.info("AUC: %s", eval_result.auc, tab=3)
            log.info("Time elapsed: %s", eval_timer.elapsed, tab=3)
            log.info("CPU Usage: %s", eval_cpu_usage.result, tab=3)

            with ifile.section(f"Window {i}"):
                with ifile.section("Evaluation"):
                    eval_result.to_indented_file(ifile)
                with ifile.section("Confidence Analysis"):
                    confidence_result.to_indented_file(ifile)

        results = append_to_df(
            df=results,
            data={
                **eval_result.as_dict(),
                "num_io": len(data),
                "num_reject": len(data[data["reject"] == 1]),
                "elapsed_time": window_timer.elapsed,
                "train_time": 0.0,
                "prediction_time": pred_timer.elapsed,
                "type": "window",
                "window_id": i,
                "cpu_usage": predict_cpu_usage.result,
                "model_selection_time": 0.0,
                "model": f"window_0",
                "dataset": data_path.name,
                **confidence_result.as_dict(),
            },
        )

        results.to_csv(output / "results.csv", index=False)

    results.to_csv(output / "results.csv", index=False)
    ifile.close()

    global_end_time = default_timer()
    log.info("Total elapsed time: %s s", global_end_time - global_start_time, tab=0)


if __name__ == "__main__":
    app()
