import shutil
import sys
from pathlib import Path
from typing import Annotated, Optional

import pandas as pd

import torch

import shortuuid as suid
import typer

import clio.flashnet.training.simple as flashnet_simple
from clio.flashnet.confidence import get_confidence_cases
from clio.flashnet.constants import FEATURE_COLUMNS
from clio.flashnet.entropy import get_entropy_result
from clio.flashnet.eval import PredictionResults, flashnet_evaluate
from clio.flashnet.preprocessing.add_filter import add_filter_v2
from clio.flashnet.uncertainty import get_uncertainty_result

from clio.utils.cpu_usage import CPUUsage
from clio.utils.dataframe import append_to_df
from clio.utils.general import parse_time, ratio_to_percentage_str, torch_set_seed
from clio.utils.indented_file import IndentedFile
from clio.utils.logging import LogLevel, log_global_setup
from clio.utils.path import rmdir
from clio.utils.timer import Timer, default_timer
from clio.utils.trace_pd import TraceWindowGeneratorContext, trace_get_dataset_paths, trace_time_window_generator

app = typer.Typer(name="Exp -- Single -- Initial Only", pretty_exceptions_enable=False)


@app.command()
def exp_initial_only_b(
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
    eval_confidence_threshold: Annotated[float, typer.Option(help="The confidence threshold to for evaluation", show_default=True)] = 0.1,
    drop_rate: Annotated[float, typer.Option(help="The drop rate to use for training", show_default=True)] = 0.0,
    use_eval_dropout: Annotated[bool, typer.Option(help="Use dropout for evaluation", show_default=True)] = False,
    trace_device: Annotated[int, typer.Option(help="The device to use for data processing", show_default=True)] = -1,
    window_size: Annotated[str, typer.Option(help="The window to use for prediction", show_default=True)] = "1m",
    train_data_dir: Annotated[
        Optional[Path], typer.Argument(help="The train data directory to use for training", exists=False, file_okay=False, dir_okay=True, resolve_path=True)
    ] = None,
    filter_predict: Annotated[bool, typer.Option(help="Filter the prediction data", show_default=True)] = False,
):
    args = locals()

    global_start_time = default_timer()

    output.mkdir(parents=True, exist_ok=True)
    log = log_global_setup(output / "log.txt", level=log_level)

    window_size: int = parse_time(window_size)
    # duration = parse_time(duration)

    log.info("Args", tab=0)
    for arg in args:
        log.info("%s: %s", arg, args[arg], tab=1)

    def sort_fn(x: Path) -> int:
        str_p = str(x)
        if "device" not in str_p:
            return int(x.name.split(".")[0])

        # log.info("%s", x.parent.parent.name.split("_")[1])
        return int(x.parent.parent.name.split("_")[1])

    def filter_fn(x: Path) -> bool:
        # filter to include specified device
        if trace_device < 0:
            return True

        str_p = str(x)
        if "device" not in str_p:
            return True

        return f"device_{trace_device}" in str_p

    data_paths = trace_get_dataset_paths(data_dir, profile_name=profile_name, feat_name=feat_name, readonly_data=True, sort_fn=sort_fn, filter_fn=filter_fn)
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
    prediction_results: PredictionResults = PredictionResults()

    torch_set_seed(seed)
    device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() and cuda >= 0 else "cpu")

    base_model_dir = output / "models"
    # NOTE: Remove the base model directory if it exists
    rmdir(base_model_dir)
    base_model_dir.mkdir(parents=True, exist_ok=True)

    ifile = IndentedFile(output / "stats.txt")
    current_group_key = suid.uuid()
    model: torch.nn.Module | torch.ScriptModule | None = None
    ctx = TraceWindowGeneratorContext()
    initial_df = pd.read_csv(data_paths[0])
    reference = pd.DataFrame()

    # log.info("Window Size: %s", window_size, tab=0)
    # log.info("Data paths", tab=0)
    # for data_path in data_paths:
    #     log.info("%s", data_path, tab=1)

    #######################
    ##     TRAIN DATA    ##
    #######################

    if train_data_dir and not train_data_dir.exists():
        raise FileNotFoundError(f"Train data directory does not exist: {train_data_dir}")

    no_train_data = True
    if train_data_dir and train_data_dir.exists():
        no_train_data = False
        train_data_paths = trace_get_dataset_paths(
            train_data_dir, profile_name=profile_name, feat_name=feat_name, readonly_data=True, sort_fn=sort_fn, filter_fn=filter_fn
        )
        if len(train_data_paths) == 0:
            raise ValueError(f"No dataset found in {train_data_dir}")

        # log.info("Train Data Paths", tab=0)
        # for train_data_path in train_data_paths:
        #     log.info("%s", train_data_path, tab=1)

        train_data = pd.concat([pd.read_csv(train_data_path) for train_data_path in train_data_paths])
        # train_data = add_filter_v2(train_data)

        log.info("Train Data", tab=1)
        log.info("Length: %s", len(train_data), tab=2)
        log.info("Columns: %s", train_data.columns, tab=2)

        log.info("Training", tab=1)

        train_cpu_usage = CPUUsage()
        model_id = suid.uuid()
        train_cpu_usage.update()
        model_dir = base_model_dir / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "model.pt"

        with Timer(name="Pipeline -- Initial Model Training") as timer:
            train_result = flashnet_simple.flashnet_train(
                model_path=model_path,
                dataset=train_data,
                retrain=False,
                batch_size=batch_size,
                prediction_batch_size=prediction_batch_size,
                lr=learning_rate,
                epochs=epochs,
                norm_mean=None,
                norm_std=None,
                n_data=None,
                device=device,
                drop_rate=drop_rate,
                use_eval_dropout=use_eval_dropout,
                layers=[512, 512],
            )
        train_cpu_usage.update()
        log.info("Pipeline Initial Model")
        log.info("Elapsed time: %s", timer.elapsed, tab=2)
        log.info("CPU Usage: %s", train_cpu_usage.result, tab=2)
        log.info("AUC: %s", train_result.auc, tab=2)
        # log.info("Train Result: %s", train_result, tab=2)

        assert len(train_data) == train_result.num_io, "sanity check, number of training data should be the same as the number of input/output"

        prediction_results.append(train_result.prediction_result)

        confidence_result = get_confidence_cases(
            labels=train_result.labels,
            predictions=train_result.predictions,
            probabilities=train_result.probabilities,
            threshold=threshold,
            confidence_threshold=eval_confidence_threshold,
        )

        log.info("Confidence", tab=2)
        log.info("Best Case: %s", ratio_to_percentage_str(confidence_result.best_case_ratio), tab=3)
        log.info("Worst Case: %s", ratio_to_percentage_str(confidence_result.worst_case_ratio), tab=3)
        log.info("Clueless Case: %s", ratio_to_percentage_str(confidence_result.clueless_case_ratio), tab=3)
        log.info("Lucky Case: %s", ratio_to_percentage_str(confidence_result.lucky_case_ratio), tab=3)

        uncertainty_result = get_uncertainty_result(labels=train_result.labels, predictions=train_result.predictions, probabilities=train_result.probabilities)

        log.info("Uncertainty", tab=2)
        log.info("Mean: %s", uncertainty_result.statistic.avg, tab=3)
        log.info("Median: %s", uncertainty_result.statistic.median, tab=3)
        log.info("P90: %s", uncertainty_result.statistic.p90, tab=3)

        entropy_result = get_entropy_result(labels=train_result.labels, predictions=train_result.predictions, probabilities=train_result.probabilities)

        log.info("Entropy", tab=2)
        log.info("Mean: %s", entropy_result.statistic.avg, tab=3)
        log.info("Median: %s", entropy_result.statistic.median, tab=3)
        log.info("P90: %s", entropy_result.statistic.p90, tab=3)

        results = append_to_df(
            df=results,
            data={
                **train_result.eval_dict(),
                "num_io": len(train_data),
                "num_reject": len(train_data[train_data["reject"] == 1]),
                "elapsed_time": timer.elapsed,
                "train_time": train_result.train_time,
                "train_data_size": len(train_data),
                "prediction_time": train_result.prediction_time,
                "type": "train",
                "window_id": -1,
                "cpu_usage": train_cpu_usage.result,
                "model_selection_time": 0.0,
                "group": current_group_key,
                "dataset": str(train_data_dir),
                **confidence_result.as_dict(),
                **uncertainty_result.as_dict(),
                **entropy_result.as_dict(),
            },
        )

        assert train_result.model_path == model_path, "sanity check, model path should be the same as the initial model path"
        # model = flashnet_simple.load_model(model_path, device=device)
        model = flashnet_simple.load_model(train_result.model_path, device=device)

        with ifile.section("Window 0"):
            with ifile.section("Evaluation"):
                train_result.to_indented_file(ifile)
            with ifile.section("Confidence Analysis"):
                confidence_result.to_indented_file(ifile)

    #######################
    ## PREDICTION WINDOW ##
    #######################

    # for i, data_path in enumerate(data_paths):
    for i, ctx, curr_path, reference, window, is_interval_valid, is_last in trace_time_window_generator(
        ctx=ctx,
        window_size=window_size * 60,
        trace_paths=data_paths,
        n_data=len(data_paths),
        current_trace=initial_df,
        reference=reference,
        return_last_remaining_data=True,
        curr_count=0,
        curr_ts_record=0,
        end_ts=-1,
    ):
        log.info("Length of window: %s", len(window), tab=2)
        if i == 0 and no_train_data:
            log.info("Training", tab=1)

            train_cpu_usage = CPUUsage()
            model_id = suid.uuid()
            train_cpu_usage.update()
            model_dir = base_model_dir / model_id
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / "model.pt"

            with Timer(name="Pipeline -- Initial Model Training -- Window %d" % i) as timer:
                train_result = flashnet_simple.flashnet_train(
                    model_path=model_path,
                    dataset=window,
                    retrain=False,
                    batch_size=batch_size,
                    prediction_batch_size=prediction_batch_size,
                    lr=learning_rate,
                    epochs=epochs,
                    norm_mean=None,
                    norm_std=None,
                    n_data=None,
                    device=device,
                    drop_rate=drop_rate,
                    use_eval_dropout=use_eval_dropout,
                )
            train_cpu_usage.update()
            log.info("Pipeline Initial Model")
            log.info("Elapsed time: %s", timer.elapsed, tab=2)
            log.info("CPU Usage: %s", train_cpu_usage.result, tab=2)
            log.info("AUC: %s", train_result.auc, tab=2)
            # log.info("Train Result: %s", train_result, tab=2)

            assert len(window) == train_result.num_io, "sanity check, number of data in window should be the same as the number of input/output"

            prediction_results.append(train_result.prediction_result)

            confidence_result = get_confidence_cases(
                labels=train_result.labels,
                predictions=train_result.predictions,
                probabilities=train_result.probabilities,
                threshold=threshold,
                confidence_threshold=eval_confidence_threshold,
            )

            log.info("Confidence", tab=2)
            log.info("Best Case: %s", ratio_to_percentage_str(confidence_result.best_case_ratio), tab=3)
            log.info("Worst Case: %s", ratio_to_percentage_str(confidence_result.worst_case_ratio), tab=3)
            log.info("Clueless Case: %s", ratio_to_percentage_str(confidence_result.clueless_case_ratio), tab=3)
            log.info("Lucky Case: %s", ratio_to_percentage_str(confidence_result.lucky_case_ratio), tab=3)

            uncertainty_result = get_uncertainty_result(
                labels=train_result.labels, predictions=train_result.predictions, probabilities=train_result.probabilities
            )

            log.info("Uncertainty", tab=2)
            log.info("Mean: %s", uncertainty_result.statistic.avg, tab=3)
            log.info("Median: %s", uncertainty_result.statistic.median, tab=3)
            log.info("P90: %s", uncertainty_result.statistic.p90, tab=3)

            entropy_result = get_entropy_result(labels=train_result.labels, predictions=train_result.predictions, probabilities=train_result.probabilities)

            log.info("Entropy", tab=2)
            log.info("Mean: %s", entropy_result.statistic.avg, tab=3)
            log.info("Median: %s", entropy_result.statistic.median, tab=3)
            log.info("P90: %s", entropy_result.statistic.p90, tab=3)

            results = append_to_df(
                df=results,
                data={
                    **train_result.eval_dict(),
                    "num_io": len(window),
                    "num_reject": len(window[window["reject"] == 1]),
                    "elapsed_time": timer.elapsed,
                    "train_time": train_result.train_time,
                    "train_data_size": len(window),
                    "prediction_time": train_result.prediction_time,
                    "type": "window",
                    "window_id": i,
                    "cpu_usage": train_cpu_usage.result,
                    "model_selection_time": 0.0,
                    "group": current_group_key,
                    "dataset": str(curr_path),
                    **confidence_result.as_dict(),
                    **uncertainty_result.as_dict(),
                    **entropy_result.as_dict(),
                },
            )

            assert train_result.model_path == model_path, "sanity check, model path should be the same as the initial model path"
            # model = flashnet_simple.load_model(model_path, device=device)
            model = flashnet_simple.load_model(train_result.model_path, device=device)

            with ifile.section("Window 0"):
                with ifile.section("Evaluation"):
                    train_result.to_indented_file(ifile)
                with ifile.section("Confidence Analysis"):
                    confidence_result.to_indented_file(ifile)

            continue

        #######################
        ## PREDICTION WINDOW ##
        #######################

        log.info("Predicting", tab=1)

        predict_data = window.copy()
        if filter_predict:
            predict_data = add_filter_v2(predict_data)

        with Timer(name="Pipeline -- Window %s" % i) as window_timer:
            #######################
            ##     PREDICTION    ##
            #######################

            predict_cpu_usage = CPUUsage()
            predict_cpu_usage.update()
            with Timer(name="Pipeline -- Prediction -- Window %s" % i) as pred_timer:
                prediction_result = flashnet_simple.flashnet_predict(
                    model=model, dataset=predict_data, device=device, batch_size=prediction_batch_size, threshold=threshold, use_eval_dropout=use_eval_dropout
                )
            predict_cpu_usage.update()
            prediction_time = pred_timer.elapsed
            log.info("Prediction", tab=2)
            log.info("Time elapsed: %s", prediction_time, tab=3)
            log.info("CPU Usage: %s", predict_cpu_usage.result, tab=3)
            prediction_results.append(prediction_result)

            #######################
            ## ASSESS CONFIDENCE ##
            #######################

            confidence_result = get_confidence_cases(
                labels=prediction_result.labels,
                predictions=prediction_result.predictions,
                probabilities=prediction_result.probabilities,
                confidence_threshold=eval_confidence_threshold,
                threshold=threshold,
            )

            log.info("Confidence", tab=2)
            log.info("Best Case: %s", ratio_to_percentage_str(confidence_result.best_case_ratio), tab=3)
            log.info("Worst Case: %s", ratio_to_percentage_str(confidence_result.worst_case_ratio), tab=3)
            log.info("Clueless Case: %s", ratio_to_percentage_str(confidence_result.clueless_case_ratio), tab=3)
            log.info("Lucky Case: %s", ratio_to_percentage_str(confidence_result.lucky_case_ratio), tab=3)

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
            log.info("Data", tab=3)
            log.info("Total: %s", len(window), tab=4)
            log.info("Num Reject: %s", len(window[window["reject"] == 1]), tab=4)
            log.info("Num Accept: %s", len(window[window["reject"] == 0]), tab=4)
            log.info("Accuracy: %s", eval_result.accuracy, tab=3)
            log.info("AUC: %s", eval_result.auc, tab=3)
            log.info("Time elapsed: %s", eval_timer.elapsed, tab=3)
            log.info("CPU Usage: %s", eval_cpu_usage.result, tab=3)

            with ifile.section(f"Window {i}"):
                with ifile.section("Evaluation"):
                    with ifile.section("Model Performance"):
                        eval_result.to_indented_file(ifile)
                    with ifile.section("Confidence Analysis"):
                        confidence_result.to_indented_file(ifile)

        #######################
        ##    SAVE RESULTS   ##
        #######################

        results = append_to_df(
            df=results,
            data={
                **eval_result.as_dict(),
                "num_io": len(predict_data),
                "num_reject": len(predict_data[predict_data["reject"] == 1]),
                "elapsed_time": window_timer.elapsed,
                "train_time": 0.0,
                "train_data_size": 0,
                "prediction_time": pred_timer.elapsed,
                "type": "window",
                "window_id": i,
                "cpu_usage": predict_cpu_usage.result,
                "model_selection_time": 0.0,
                "group": current_group_key,
                "dataset": str(curr_path),
                **confidence_result.as_dict(),
                **uncertainty_result.as_dict(),
                **entropy_result.as_dict(),
            },
        )

        if i % 4 == 0:
            results.to_csv(output / "results.csv", index=False)
            prediction_results.to_msgpack(output / "prediction_results.msgpack")
            ifile.flush()

    results.to_csv(output / "results.csv", index=False)
    prediction_results.to_msgpack(output / "prediction_results.msgpack")
    ifile.close()

    global_end_time = default_timer()
    log.info("Total elapsed time: %s s", global_end_time - global_start_time, tab=0)


if __name__ == "__main__":
    app()
