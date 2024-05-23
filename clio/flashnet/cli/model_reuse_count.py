import json
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

from clio.flashnet.eval import flashnet_evaluate
from clio.flashnet.training import flashnet_predict, flashnet_train
from clio.utils.cpu_usage import CPUUsage
from clio.utils.general import parse_time, torch_set_seed
from clio.utils.indented_file import IndentedFile
from clio.utils.logging import LogLevel, log_get, log_global_setup
from clio.utils.timer import Timer, default_timer
from clio.utils.trace_pd import TraceWindowGeneratorContext, trace_get_dataset_paths, trace_time_window_generator

_log = log_get(__name__)


app = typer.Typer(name="Model Reuse Count", pretty_exceptions_enable=False)


def get_cached_norm(path: str | Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    # check if norm_mean and norm_std exists in train_data
    norm_mean = None
    norm_std = None
    norm_mean_path = path / "norm_mean.npy"
    norm_std_path = path / "norm_std.npy"
    if norm_mean_path.exists() and norm_std_path.exists():
        _log.info("Loading precomputed norm_mean and norm_std", tab=2)
        norm_mean = np.load(norm_mean_path, allow_pickle=True)
        norm_std = np.load(norm_std_path, allow_pickle=True)
        # check if norm_mean and norm_std is valid
        if norm_mean.size <= 1 or norm_std.size <= 1:
            _log.error("Invalid norm_mean and norm_std")
            norm_mean = None
            norm_std = None

    return norm_mean, norm_std


@app.command()
def model_reuse_count(
    data_dir: Annotated[
        Path, typer.Argument(help="The test data directory to use for prediction", exists=True, file_okay=False, dir_okay=True, resolve_path=True)
    ],
    output: Annotated[Path, typer.Option(help="The output path to write the results to")],
    window_size: Annotated[str, typer.Option(help="The window size to use for prediction (in minute(s))", show_default=True)] = "10",
    log_level: Annotated[LogLevel, typer.Option(help="The log level to use")] = LogLevel.INFO,
    profile_name: Annotated[str, typer.Option(help="The profile name to use for prediction", show_default=True)] = "profile_v1_filter",
    feat_name: Annotated[str, typer.Option(help="The feature name to use for prediction", show_default=True)] = "feat_v6_ts",
    learning_rate: Annotated[float, typer.Option(help="The learning rate to use for training", show_default=True)] = 0.0001,
    epochs: Annotated[int, typer.Option(help="The number of epochs to use for training", show_default=True)] = 20,
    batch_size: Annotated[int, typer.Option(help="The batch size to use for training", show_default=True)] = 32,
    prediction_batch_size: Annotated[int, typer.Option(help="The batch size to use for prediction", show_default=True)] = -1,
    duration: Annotated[str, typer.Option(help="The duration to use for prediction (in minute(s))", show_default=True)] = "-1",
    seed: Annotated[int, typer.Option(help="The seed to use for random number generation", show_default=True)] = 3003,
    cuda: Annotated[int, typer.Option(help="Use CUDA for training and prediction", show_default=True)] = 0,
):
    args = locals()

    global_start_time = default_timer()

    output.mkdir(parents=True, exist_ok=True)
    log = log_global_setup(output / "log.txt", level=log_level)

    window_size = parse_time(window_size)
    duration = parse_time(duration)

    log.info("Args", tab=0)
    for arg in args:
        log.info("%s: %s", arg, args[arg], tab=1)

    data_paths = trace_get_dataset_paths(data_dir, profile_name=profile_name, feat_name=feat_name, readonly_data=True)
    if len(data_paths) == 0:
        raise ValueError(f"No dataset found in {data_dir}")

    if prediction_batch_size < 0:
        prediction_batch_size = batch_size

    # Load cached norm
    norm_dir = output / "norm"
    norm_dir.mkdir(parents=True, exist_ok=True)
    norm_mean, norm_std = get_cached_norm(norm_dir)

    ###########################################################################
    # PIPELINE
    ###########################################################################

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
        ],
    )

    models_dictionary = {}

    base_model_dir = output / "models"
    base_model_dir.mkdir(parents=True, exist_ok=True)

    window_dir = output / "window"
    window_dir.mkdir(parents=True, exist_ok=True)

    # Remove all existing models except initial model
    # for p in base_model_dir.glob("*.pt"):
    #     if p.stem != "initial":
    #         p.unlink()

    # Remove all existing window data
    for p in window_dir.glob("*.csv"):
        p.unlink()

    #######################
    ## PREDICTION WINDOW ##
    #######################

    torch_set_seed(seed)
    device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() and cuda >= 0 else "cpu")

    model: torch.nn.Module = None
    ctx = TraceWindowGeneratorContext()
    initial_df = pd.read_csv(data_paths[0])
    reference_data = pd.DataFrame()
    model_selection_time = 0.0
    current_model_name = "" if model is None else model_path.stem

    for i, ctx, curr_path, reference, window, is_interval_valid, is_last in trace_time_window_generator(
        ctx=ctx,
        window_size=window_size * 60,
        trace_paths=data_paths,
        n_data=len(data_paths),
        current_trace=initial_df,
        reference=reference_data,
        return_last_remaining_data=True,
        curr_count=0,
        curr_ts_record=0,
        end_ts=duration * 60 * 1000,
    ):
        # log.info("Processing window %d (reference: %d, window: %d)", i, len(reference_data), len(window_data), tab=1)
        log.info("Processing window %d", i, tab=1)

        if model is None:
            # Train initial model using this window
            log.info("Training initial model")

            train_cpu_usage = CPUUsage()
            train_cpu_usage.update()
            model_path = base_model_dir / f"window_{i}.pt"
            with Timer(name="Pipeline -- Initial Model Training -- Window %d" % i) as timer:
                train_result = flashnet_train(
                    model_path=model_path,
                    dataset=window,
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
                "num_io": len(window),
                "num_reject": len(window[window["reject"] == 1]),
                "elapsed_time": timer.elapsed,
                "train_time": train_result.train_time,
                "prediction_time": train_result.prediction_time,
                "type": "window",
                "window_id": i,
                "cpu_usage": train_cpu_usage.result,
                "model_selection_time": 0.0,
                "model": f"window_{i}",
            }
            assert train_result.model_path == model_path, "sanity check, model path should be the same as the initial model path"
            model = torch.jit.load(model_path)
            model = model.to(device)
            current_model_name = model_path.stem

            window_data_path = window_dir / f"window_{i}.csv"
            window.to_csv(window_data_path, index=False)
            continue

        with Timer(name="Pipeline -- Window %s" % i) as window_timer:
            # Predict
            predict_cpu_usage = CPUUsage()
            predict_cpu_usage.update()
            with Timer(name="Pipeline -- Prediction -- Window %s" % i) as pred_timer:
                pred, label = flashnet_predict(model, window, batch_size=prediction_batch_size, device=device)
            predict_cpu_usage.update()
            prediction_time = pred_timer.elapsed
            log.info("Prediction", tab=2)
            log.info("Time elapsed: %s", pred_timer.elapsed, tab=3)
            log.info("CPU Usage: %s", predict_cpu_usage.result, tab=3)

            # Evaluate
            eval_cpu_usage = CPUUsage()
            eval_cpu_usage.update()
            with Timer(name="Pipeline -- Evaluation -- Window %s" % i) as eval_timer:
                eval_result = flashnet_evaluate(label, pred)
            eval_cpu_usage.update()
            log.info("Evaluation", tab=2)
            log.info("Time elapsed: %s", eval_timer.elapsed, tab=3)
            log.info("CPU Usage: %s", eval_cpu_usage.result, tab=3)

            ##################################
            # TRAIN NEW MODEL ON THIS WINDOW #
            ##################################

            model_path = base_model_dir / f"window_{i}.pt"
            train_time = 0.0
            log.info("Exist model path: %s", model_path.exists(), tab=2)
            if not model_path.exists():
                log.info("Training new model %s", model_path, tab=2)
                new_train_result = flashnet_train(
                    model_path=model_path,
                    dataset=window,
                    retrain=False,
                    batch_size=batch_size,
                    prediction_batch_size=prediction_batch_size,
                    # tqdm=True,
                    lr=learning_rate,
                    epochs=epochs,
                    norm_mean=norm_mean,
                    norm_std=norm_std,
                    n_data=None,
                )
                train_time = new_train_result.train_time
            else:
                log.info("Model %s already trained, reusing it...", model_path, tab=2)

            window_data_path = window_dir / f"window_{i}.csv"
            window.to_csv(window_data_path, index=False)

        results.loc[len(results)] = {
            **eval_result.as_dict(),
            "num_io": len(window),
            "num_reject": len(window[window["reject"] == 1]),
            "elapsed_time": window_timer.elapsed,
            "train_time": train_time,
            "prediction_time": pred_timer.elapsed,
            "type": "window",
            "window_id": i,
            "cpu_usage": predict_cpu_usage.result,
            "model_selection_time": model_selection_time,
            "model": current_model_name,
        }

        if i % 2 == 0:
            # Save results every 2 windows
            results.to_csv(output / "results.csv", index=False)

        ##############################################
        # RETRAIN MODEL BASED ON PREVIOUS BEST MODEL #
        ##############################################
        # TODO...

        #########################################
        # PICK BEST MODEL FOR THIS WINDOW       #
        # AND CHOOSE THIS MODEL FOR NEXT WINDOW #
        #########################################

        with Timer(name="Pipeline -- Model Selection -- Window %s" % i) as timer:
            temp_results = {}
            model_paths: list[Path] = []
            for p in sorted(base_model_dir.glob("*.pt"), key=lambda x: int(x.stem.split("_")[1]) if "_" in x.stem else -1):
                # log.info("Checking model %s", p, tab=3)
                # parse int
                window_id = None
                try:
                    window_id = int(p.stem.split("_")[1])
                except IndexError:
                    pass
                except ValueError:
                    pass
                if window_id is None:
                    # log.info("Appending initial model %s", p, tab=4)
                    # initial model
                    model_paths.append(p)
                else:
                    # window model
                    # append model that is less equal than i
                    if window_id <= i:
                        # log.info("Appending window model %s", p, tab=4)
                        model_paths.append(p)

            log.info("Checking models from %s to %s", model_paths[0], model_paths[-1], tab=2)

            for p in model_paths:
                if p.stem not in models_dictionary:
                    models_dictionary[p.stem] = 0

                m = torch.jit.load(p)
                m = m.to(device)
                pred, label = flashnet_predict(m, window, batch_size=prediction_batch_size, device=device)
                eval_result = flashnet_evaluate(label, pred)
                temp_results[p.stem] = eval_result.auc

            best_model = max(temp_results, key=temp_results.get)
            models_dictionary[best_model] += 1
            log.info("Best model for window %d: %s", i, best_model, tab=2)
            best_model_path = base_model_dir / f"{best_model}.pt"
            log.info("Loading best model %s", best_model_path, tab=2)
            model = torch.jit.load(best_model_path)
            model = model.to(device)
            current_model_name = best_model

        log.info("Model Selection", tab=2)
        log.info("Time elapsed: %s", timer.elapsed, tab=3)
        model_selection_time = timer.elapsed

    log.info("Writing results.csv", tab=0)
    results.to_csv(output / "results.csv", index=False)

    log.info("Writing models.csv", tab=0)
    models_dictionary_df = pd.DataFrame(models_dictionary.items(), columns=["model", "count"])
    models_dictionary_df.to_csv(output / "models.csv", index=False)

    log.info("Writing stats.stats", tab=0)

    with IndentedFile(output / "stats.stats") as stats_file:
        with stats_file.section("Args"):
            for arg in args:
                stats_file.writeln(f"{arg}: {args[arg]}")

        with stats_file.section("Model Selection"):
            for model, count in models_dictionary.items():
                stats_file.writeln(f"{model}: {count}")

        with stats_file.section("Results"):
            for _, row in results.iterrows():
                with stats_file.section("Window %d" % row["window_id"]):
                    for col in results.columns:
                        stats_file.writeln(f"{col}: {row[col]}")

    global_end_time = default_timer()
    log.info("Total elapsed time: %s", global_end_time - global_start_time, tab=0)


if __name__ == "__main__":
    app()
