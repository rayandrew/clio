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
from clio.flashnet.cli._shared import get_cached_norm, save_norm
from clio.flashnet.eval import flashnet_evaluate, get_confidence_data

from clio.utils.cpu_usage import CPUUsage
from clio.utils.general import parse_time, torch_set_seed
from clio.utils.indented_file import IndentedFile
from clio.utils.logging import LogLevel, log_get, log_global_setup
from clio.utils.timer import Timer, default_timer
from clio.utils.trace_pd import TraceWindowGeneratorContext, trace_get_dataset_paths, trace_time_window_generator

app = typer.Typer(name="Student Management", pretty_exceptions_enable=False)
_log = log_get(__name__)


def get_balanced_confidence_and_model_performance(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray, confidence_threshold: float = 0.1):
    assert len(y_true) == len(y_pred) == len(y_prob), "sanity check, all data should have the same length"
    # to make sure we pick the right data
    # 1. assess the accuracy of that specific confidence data
    # 2. if the prediction of that data is wrong and the confidence is low, consider it as a NICE CASE
    # 3. if the prediction of that data is right and the confidence is high, consider it as a BEST CASE
    # 4. if the prediction of that data is correct, but the confidence is low, consider it as a SLIGHTLY GOOD CASE
    #    - How to increase the confidence of that data?
    # 5. if the prediction of that data is wrong, but the confidence is high -> WORST CASE

    high_conf_indices, low_conf_indices = get_confidence_data(y_prob, threshold=0.5, confidence_threshold=confidence_threshold)

    # check high confidence data
    high_conf_correct = y_true[high_conf_indices] == y_pred[high_conf_indices]
    high_conf_incorrect = ~high_conf_correct

    # check low confidence data
    low_conf_correct = y_true[low_conf_indices] == y_pred[low_conf_indices]
    low_conf_incorrect = ~low_conf_correct

    best_case = high_conf_correct.sum()
    worst_case = high_conf_incorrect.sum()
    slightly_good_case = low_conf_correct.sum()
    nice_case = low_conf_incorrect.sum()

    _log.info("Low confidence data", tab=2)
    _log.info("Total number of data: %s", len(y_true), tab=3)
    _log.info("Number of low confidence data: %s", len(low_conf_indices), tab=3)
    _log.info("Sample of low confidence data indices: %s", np.random.choice(low_conf_indices, 5), tab=3)

    return high_conf_indices, low_conf_indices, best_case, worst_case, slightly_good_case, nice_case


@app.command()
def exp(
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

    results = pd.DataFrame(
        [],
        columns=[
            "accuracy",
            "precision",
            "recall",
            "f1",
            "auc",
            "fpr",
            "fnr",
            "num_io",
            "num_reject",
            "elapsed_time",
            "prediction_time",
            "train_time",
            "type",
            "window_id",
            "cpu_usage",
            "model_selection_time",
            "model",
            "dataset",
        ],
    )

    #######################
    ## PREDICTION WINDOW ##
    #######################

    torch_set_seed(seed)
    device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() and cuda >= 0 else "cpu")

    model: torch.nn.Module = None

    base_model_dir = output / "models"
    base_model_dir.mkdir(parents=True, exist_ok=True)

    for i, data_path in enumerate(data_paths):
        log.info("Processing dataset: %s", data_path, tab=1)
        data = pd.read_csv(data_path)
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
                results.loc[len(results)] = {
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
                }
                assert train_result.model_path == model_path, "sanity check, model path should be the same as the initial model path"
                model = flashnet_simple.load_model(model_path, map_location=device)
                norm_mean = train_result.norm_mean
                norm_std = train_result.norm_std
                save_norm(model_dir, norm_mean, norm_std)
                continue
            else:
                log.info("Model %s already trained, reusing it...", model_path, tab=2)
                model = flashnet_simple.load_model(model_path, map_location=device)
                model = model.to(device)

        #######################
        ## PREDICTION WINDOW ##
        #######################

        log.info("Predicting %s", data_path, tab=1)
        num_data = len(data)
        num_reject = data["reject"].sum()
        num_accept = num_data - num_reject

        log.info("Data", tab=2)
        log.info("Number of data: %s", num_data, tab=3)
        log.info("Reject: %s (%s %%)", num_reject, num_reject / num_data * 100, tab=3)
        log.info("Accept: %s (%s %%)", num_accept, num_accept / num_data * 100, tab=3)

        with Timer(name="Pipeline -- Window %s" % i) as window_timer:
            # Predict
            predict_cpu_usage = CPUUsage()
            predict_cpu_usage.update()
            with Timer(name="Pipeline -- Prediction -- Window %s" % i) as pred_timer:
                pred, label, probs = flashnet_simple.flashnet_predict(
                    model, data, batch_size=prediction_batch_size, device=device, norm_mean=norm_mean, norm_std=norm_std
                )
            predict_cpu_usage.update()
            prediction_time = pred_timer.elapsed
            log.info("Prediction", tab=2)
            log.info("Time elapsed: %s", pred_timer.elapsed, tab=3)
            log.info("CPU Usage: %s", predict_cpu_usage.result, tab=3)

            #######################
            ## ASSESS CONFIDENCE ##
            #######################

            high_conf_indices, low_conf_indices, best_case, worst_case, slightly_good_case, nice_case = get_balanced_confidence_and_model_performance(
                label, pred, probs, confidence_threshold=0.1
            )
            log.info("Confidence", tab=2)
            log.info("Best Case: %s", best_case, tab=3)
            log.info("Worst Case: %s", worst_case, tab=3)
            log.info("Slightly Good Case: %s", slightly_good_case, tab=3)
            log.info("Nice Case: %s", nice_case, tab=3)

            # Evaluate
            eval_cpu_usage = CPUUsage()
            eval_cpu_usage.update()
            with Timer(name="Pipeline -- Evaluation -- Window %s" % i) as eval_timer:
                eval_result = flashnet_evaluate(label, pred)
            eval_cpu_usage.update()
            log.info("Evaluation", tab=2)
            log.info("Accuracy: %s", eval_result.accuracy, tab=3)
            log.info("AUC: %s", eval_result.auc, tab=3)
            log.info("Time elapsed: %s", eval_timer.elapsed, tab=3)
            log.info("CPU Usage: %s", eval_cpu_usage.result, tab=3)

        results.loc[len(results)] = {
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
        }

    results.to_csv(output / "results.csv", index=False)

    global_end_time = default_timer()
    log.info("Total elapsed time: %s s", global_end_time - global_start_time, tab=0)


if __name__ == "__main__":
    app()
