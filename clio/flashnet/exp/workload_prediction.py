import json
import sys
from pathlib import Path
from timeit import default_timer as timer
from typing import Annotated, Optional

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import typer

from clio.flashnet.eval import Trainer, flashnet_evaluate, flashnet_predict
from clio.flashnet.training import flashnet_train
from clio.utils.cpu_usage import CPUUsage
from clio.utils.general import parse_time, tf_set_seed
from clio.utils.indented_file import IndentedFile
from clio.utils.keras import load_model
from clio.utils.logging import LogLevel, log_get, log_global_setup
from clio.utils.timer import Timer, default_timer
from clio.utils.trace_pd import TraceWindowGeneratorContext, read_dataset_as_df, trace_get_dataset_paths, trace_time_window_generator

_log = log_get(__name__)

app = typer.Typer(name="Worload Prediction", pretty_exceptions_enable=False)


def get_cached_norm(path: str | Path, data_size: int | None = None) -> tuple[np.ndarray | None, np.ndarray | None]:
    # check if norm_mean and norm_variance exists in train_data
    norm_mean = None
    norm_variance = None
    norm_mean_path = path / "norm_mean.npy" if data_size is None else path / f"norm_mean_{data_size}.npy"
    norm_variance_path = path / "norm_variance.npy" if data_size is None else path / f"norm_variance_{data_size}.npy"
    if norm_mean_path.exists() and norm_variance_path.exists():
        _log.info("Loading precomputed norm_mean and norm_variance", tab=2)
        norm_mean = np.load(norm_mean_path, allow_pickle=True)
        norm_variance = np.load(norm_variance_path, allow_pickle=True)
        # check if norm_mean and norm_variance is valid
        if norm_mean.size <= 1 or norm_variance.size <= 1:
            _log.error("Invalid norm_mean and norm_variance")
            norm_mean = None
            norm_variance = None

    return norm_mean, norm_variance


@app.command()
def workload_prediction(
    test_data_dir: Annotated[
        Path, typer.Argument(help="The test data directory to use for prediction", exists=True, file_okay=False, dir_okay=True, resolve_path=True)
    ],
    output: Annotated[Path, typer.Option(help="The output path to write the results to")],
    initial_data_dir: Annotated[
        Optional[Path], typer.Option(help="The initial data directory to use for training", exists=True, file_okay=False, dir_okay=True, resolve_path=True)
    ] = None,
    window_size: Annotated[str, typer.Option(help="The window size to use for prediction (in minute(s))", show_default=True)] = "10",
    log_level: Annotated[LogLevel, typer.Option(help="The log level to use")] = LogLevel.INFO,
    learning_rate: Annotated[float, typer.Option(help="The learning rate to use for training", show_default=True)] = 0.0001,
    epochs: Annotated[int, typer.Option(help="The number of epochs to use for training", show_default=True)] = 20,
    batch_size: Annotated[int, typer.Option(help="The batch size to use for training", show_default=True)] = 8192,
    prediction_batch_size: Annotated[int, typer.Option(help="The batch size to use for prediction", show_default=True)] = -1,
    duration: Annotated[str, typer.Option(help="The duration to use for prediction (in minute(s))", show_default=True)] = "-1",
    seed: Annotated[int, typer.Option(help="The seed to use for random number generation", show_default=True)] = 3003,
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

    initial_data_paths: list[Path] = []
    if initial_data_dir is not None:
        initial_data_paths = trace_get_dataset_paths(initial_data_dir)
        if len(initial_data_paths) == 0:
            raise ValueError(f"No dataset found in {initial_data_dir}")

    test_data_paths = trace_get_dataset_paths(test_data_dir)
    if len(test_data_paths) == 0:
        raise ValueError(f"No dataset found in {test_data_dir}")

    if prediction_batch_size < 0:
        prediction_batch_size = batch_size

    # Load cached norm
    norm_mean, norm_variance = None, None
    if initial_data_dir is not None:
        norm_mean, norm_variance = get_cached_norm(initial_data_dir)

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
    initial_model_exists = False

    base_model_dir = output / "models"
    base_model_dir.mkdir(parents=True, exist_ok=True)

    window_dir = output / "window"
    window_dir.mkdir(parents=True, exist_ok=True)

    # Remove all existing models except initial model
    # for p in base_model_dir.glob("*.keras"):
    #     if p.stem != "initial":
    #         p.unlink()

    # Remove all existing window data
    for p in window_dir.glob("*.csv"):
        p.unlink()

    #########################
    ## TRAIN INITIAL MODEL ##
    #########################

    tf_set_seed(seed)

    if len(initial_data_paths) > 0:
        initial_model_path = base_model_dir / "initial.keras"
        model_path = initial_model_path
        if not initial_model_path.exists():
            # model has not been trained yet
            log.info("Training initial model")

            train_cpu_usage = CPUUsage()
            train_cpu_usage.update()
            with Timer(name="Pipeline -- Initial Model Training") as timer:
                train_data = pd.concat([read_dataset_as_df(path) for path in initial_data_paths])
                train_result = flashnet_train(
                    model_path=model_path,
                    dataset_ori=train_data,
                    retrain=False,
                    batch_size=batch_size,
                    prediction_batch_size=prediction_batch_size,
                    # tqdm=True,
                    lr=learning_rate,
                    epochs=epochs,
                    norm_mean=norm_mean,
                    norm_variance=norm_variance,
                    n_data=None,
                )
            train_cpu_usage.update()
            log.info("Pipeline Initial Model")
            log.info("Elapsed time: %s", timer.elapsed, tab=2)
            log.info("CPU Usage: %s", train_cpu_usage.result, tab=2)
            log.info("AUC: %s", train_result.auc, tab=2)
            # log.info("Train Result: %s", train_result, tab=2)
            results.loc[len(results)] = {
                **train_result.eval_dict(),
                "num_io": len(train_data),
                "num_reject": len(train_data[train_data["reject"] == 1]),
                "prediction_time": 0,
                "type": "initial",
                "window_id": -1,
                "cpu_usage": train_cpu_usage.result,
                "train_time": timer.elapsed,
                "model_selection_time": 0.0,
                "model": "initial",
            }
            assert train_result.model_path == model_path, "sanity check, model path should be the same as the initial model path"
        else:
            log.info("Initial model already trained")

        models_dictionary[model_path.stem] = 1  # initial model will be used once no matter what
        initial_model_exists = True

    #######################
    ## PREDICTION WINDOW ##
    #######################

    model: Trainer | None = None
    if initial_model_exists:
        model = load_model(model_path)
    ctx = TraceWindowGeneratorContext()
    initial_df = read_dataset_as_df(test_data_paths[0])
    reference_data = pd.DataFrame()
    model_selection_time = 0.0
    current_model_name = "" if model is None else model_path.stem

    for i, ctx, reference, window, is_interval_valid, is_last in trace_time_window_generator(
        ctx=ctx,
        window_size=window_size * 60,
        trace_paths=test_data_paths,
        n_data=len(test_data_paths),
        current_trace=initial_df,
        reference=reference_data,
        return_last_remaining_data=True,
        curr_count=0,
        curr_ts_record=0,
        reader=read_dataset_as_df,
        end_ts=duration * 60 * 1000,
    ):
        # log.info("Processing window %d (reference: %d, window: %d)", i, len(reference_data), len(window_data), tab=1)
        log.info("Processing window %d", i, tab=1)

        if model is None:
            # Train initial model using this window
            log.info("Training initial model")

            train_cpu_usage = CPUUsage()
            train_cpu_usage.update()
            model_path = base_model_dir / f"window_{i}.keras"
            with Timer(name="Pipeline -- Initial Model Training -- Window %d" % i) as timer:
                train_result = flashnet_train(
                    model_path=model_path,
                    dataset_ori=window,
                    retrain=False,
                    batch_size=batch_size,
                    prediction_batch_size=prediction_batch_size,
                    # tqdm=True,
                    lr=learning_rate,
                    epochs=epochs,
                    norm_mean=norm_mean,
                    norm_variance=norm_variance,
                    n_data=None,
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
            model = load_model(model_path)
            current_model_name = model_path.stem

            window_data_path = window_dir / f"window_{i}.csv"
            window.to_csv(window_data_path, index=False)
            continue

        with Timer(name="Pipeline -- Window %s" % i) as window_timer:
            # Predict
            predict_cpu_usage = CPUUsage()
            predict_cpu_usage.update()
            with Timer(name="Pipeline -- Prediction -- Window %s" % i) as pred_timer:
                pred, label = flashnet_predict(model, window, batch_size=prediction_batch_size, tqdm=False)
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

            model_path = base_model_dir / f"window_{i}.keras"
            train_time = 0.0
            log.info("Exist model path: %s", model_path.exists(), tab=2)
            if not model_path.exists():
                log.info("Training new model %s", model_path, tab=2)
                new_train_result = flashnet_train(
                    model_path=model_path,
                    dataset_ori=window,
                    retrain=False,
                    batch_size=batch_size,
                    prediction_batch_size=prediction_batch_size,
                    # tqdm=True,
                    lr=learning_rate,
                    epochs=epochs,
                    norm_mean=norm_mean,
                    norm_variance=norm_variance,
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
            for p in sorted(base_model_dir.glob("*.keras"), key=lambda x: int(x.stem.split("_")[1]) if "_" in x.stem else -1):
                # log.info("Checking model %s", p, tab=3)
                # parse int
                window_id = None
                try:
                    window_id = int(p.stem.split("_")[1])
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

                pred, label = flashnet_predict(load_model(p), window, batch_size=prediction_batch_size, tqdm=False)
                eval_result = flashnet_evaluate(label, pred)
                temp_results[p.stem] = eval_result.auc

            best_model = max(temp_results, key=temp_results.get)
            models_dictionary[best_model] += 1
            log.info("Best model for window %d: %s", i, best_model, tab=2)
            best_model_path = base_model_dir / f"{best_model}.keras"
            log.info("Loading best model %s", best_model_path, tab=2)
            model = load_model(best_model_path)
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
