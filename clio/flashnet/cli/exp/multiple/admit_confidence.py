import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd

import torch

import shortuuid as suid
import typer

import clio.flashnet.training.ensemble as flashnet_ensemble
import clio.flashnet.training.simple as flashnet_simple
from clio.flashnet.confidence import get_confidence_cases
from clio.flashnet.constants import FEATURE_COLUMNS
from clio.flashnet.entropy import get_entropy_result
from clio.flashnet.eval import PredictionResults, flashnet_evaluate
from clio.flashnet.uncertainty import get_uncertainty_result

from clio.utils.cpu_usage import CPUUsage
from clio.utils.dataframe import append_to_df
from clio.utils.general import ratio_to_percentage_str, torch_set_seed
from clio.utils.indented_file import IndentedFile
from clio.utils.logging import LogLevel, log_global_setup
from clio.utils.path import rmdir
from clio.utils.timer import Timer, default_timer
from clio.utils.trace_pd import trace_get_dataset_paths

app = typer.Typer(name="Exp -- Multiple -- Admit -- Confidence-based")


@dataclass(kw_only=True)
class ModelGroup:
    _models: list[str | Path] = field(default_factory=list)

    def add_model(self, model: str | Path):
        self._models.append(model)

    @property
    def models(self):
        return self._models

    def __len__(self):
        return len(self._models)

    def __getitem__(self, idx):
        return self._models[idx]

    def __iter__(self):
        return iter(self._models)

    def __repr__(self):
        return f"ModelGroup(num_models={len(self)})"

    def __str__(self):
        return self.__repr__()


@app.command()
def exp_admit_uncertain(
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
    uncertainty_threshold_data: Annotated[float, typer.Option(help="Retrain data threshold", show_default=True)] = 0.85,
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
    model_group = ModelGroup()

    #######################
    ## PREDICTION WINDOW ##
    #######################

    torch_set_seed(seed)
    device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() and cuda >= 0 else "cpu")

    base_model_dir = output / "models"
    # NOTE: Remove the base model directory if it exists
    rmdir(base_model_dir)
    base_model_dir.mkdir(parents=True, exist_ok=True)

    ifile = IndentedFile(output / "stats.txt")
    current_group_key = suid.uuid()
    # model: torch.nn.Module | torch.ScriptModule | None = None
    model_path: Path | str = ""
    prediction_results: PredictionResults = PredictionResults()

    for i, data_path in enumerate(data_paths):
        log.info("Processing dataset: %s", data_path, tab=1)
        data = pd.read_csv(data_path)
        log.info("Length of data: %s", len(data), tab=2)
        if i == 0:
            log.info("Training", tab=1)

            model_id = suid.uuid()
            model_dir = base_model_dir / model_id
            model_dir.mkdir(parents=True, exist_ok=True)
            model_path = model_dir / "model.pt"

            train_cpu_usage = CPUUsage()
            train_cpu_usage.update()

            with Timer(name="Pipeline -- Initial Model Training -- Window %d" % i) as timer:
                train_result = flashnet_simple.flashnet_train(
                    model_path=model_path,
                    dataset=data,
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

            assert len(data) == train_result.num_io, "sanity check, number of data should be the same as the number of input/output"

            prediction_results.append(train_result.prediction_result)

            confidence_result = get_confidence_cases(
                labels=train_result.labels,
                predictions=train_result.predictions,
                probabilities=train_result.probabilities,
                threshold=threshold,
                confidence_threshold=eval_confidence_threshold,
            )
            uncertainty_result = get_uncertainty_result(
                labels=train_result.labels, predictions=train_result.predictions, probabilities=train_result.probabilities
            )
            entropy_result = get_entropy_result(labels=train_result.labels, predictions=train_result.predictions, probabilities=train_result.probabilities)

            # -- Confidence
            log.info("Confidence", tab=2)
            log.info("Best Case: %s", ratio_to_percentage_str(confidence_result.best_case_ratio), tab=3)
            log.info("Worst Case: %s", ratio_to_percentage_str(confidence_result.worst_case_ratio), tab=3)
            log.info("Clueless Case: %s", ratio_to_percentage_str(confidence_result.clueless_case_ratio), tab=3)
            log.info("Lucky Case: %s", ratio_to_percentage_str(confidence_result.lucky_case_ratio), tab=3)
            # -- Uncertainty
            log.info("Uncertainty", tab=2)
            log.info("Mean: %s", uncertainty_result.statistic.avg, tab=3)
            log.info("Median: %s", uncertainty_result.statistic.median, tab=3)
            log.info("P90: %s", uncertainty_result.statistic.p90, tab=3)
            # -- Entropy
            log.info("Entropy", tab=2)
            log.info("Mean: %s", entropy_result.statistic.avg, tab=3)
            log.info("Median: %s", entropy_result.statistic.median, tab=3)
            log.info("P90: %s", entropy_result.statistic.p90, tab=3)

            results = append_to_df(
                df=results,
                data={
                    **train_result.eval_dict(),
                    "num_io": len(data),
                    "num_reject": len(data[data["reject"] == 1]),
                    "elapsed_time": timer.elapsed,
                    "train_data_size": len(data),
                    "train_time": train_result.train_time,
                    "prediction_time": train_result.prediction_time,
                    "type": "window",
                    "window_id": i,
                    "cpu_usage": train_cpu_usage.result,
                    "model_selection_time": 0.0,
                    "group": current_group_key,
                    "dataset": data_path.name,
                    **confidence_result.as_dict(),
                    **uncertainty_result.as_dict(),
                    **entropy_result.as_dict(),
                },
            )

            assert train_result.model_path == model_path, "sanity check, model path should be the same as the initial model path"
            # model = flashnet_simple.load_model(model_path, device=device)
            model_path = train_result.model_path
            # model = flashnet_simple.load_model(model_path, device=device)
            model_group.add_model(model_path)

            with ifile.section("Window 0"):
                with ifile.section("Evaluation"):
                    train_result.to_indented_file(ifile)
                with ifile.section("Confidence Analysis"):
                    confidence_result.to_indented_file(ifile)

            continue

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
                prediction_result = flashnet_ensemble.flashnet_ensemble_predict_p(
                    models=model_group.models,
                    dataset=data,
                    device=device,
                    batch_size=prediction_batch_size,
                    threshold=threshold,
                    use_eval_dropout=use_eval_dropout,
                )
            predict_cpu_usage.update()
            prediction_time = pred_timer.elapsed
            log.info("Prediction", tab=2)
            log.info("Time elapsed: %s", prediction_time, tab=3)
            log.info("CPU Usage: %s", predict_cpu_usage.result, tab=3)
            prediction_results.append(prediction_result)

            #######################
            ##   ASSESS QUALITY  ##
            #######################

            confidence_result = get_confidence_cases(
                labels=prediction_result.labels,
                predictions=prediction_result.predictions,
                probabilities=prediction_result.probabilities,
                confidence_threshold=eval_confidence_threshold,
                threshold=threshold,
            )
            uncertainty_result = get_uncertainty_result(
                labels=prediction_result.labels, predictions=prediction_result.predictions, probabilities=prediction_result.probabilities
            )
            entropy_result = get_entropy_result(
                labels=prediction_result.labels, predictions=prediction_result.predictions, probabilities=prediction_result.probabilities
            )

            # -- Confidence
            log.info("Confidence", tab=2)
            log.info("Best Case: %s", ratio_to_percentage_str(confidence_result.best_case_ratio), tab=3)
            log.info("Worst Case: %s", ratio_to_percentage_str(confidence_result.worst_case_ratio), tab=3)
            log.info("Clueless Case: %s", ratio_to_percentage_str(confidence_result.clueless_case_ratio), tab=3)
            log.info("Lucky Case: %s", ratio_to_percentage_str(confidence_result.lucky_case_ratio), tab=3)
            # -- Uncertainty
            log.info("Uncertainty", tab=2)
            log.info("Mean: %s", uncertainty_result.statistic.avg, tab=3)
            log.info("Median: %s", uncertainty_result.statistic.median, tab=3)
            log.info("P90: %s", uncertainty_result.statistic.p90, tab=3)
            # -- Entropy
            log.info("Entropy", tab=2)
            log.info("Mean: %s", entropy_result.statistic.avg, tab=3)
            log.info("Median: %s", entropy_result.statistic.median, tab=3)
            log.info("P90: %s", entropy_result.statistic.p90, tab=3)

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
            log.info("Total: %s", len(data), tab=4)
            log.info("Num Reject: %s", len(data[data["reject"] == 1]), tab=4)
            log.info("Num Accept: %s", len(data[data["reject"] == 0]), tab=4)
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
        ##   RETRAIN MODEL   ##
        #######################

        log.info("Retrain", tab=1)

        retrain_data_indices = (
            confidence_result.clueless_case_indices.tolist() + confidence_result.worst_case_indices.tolist() + confidence_result.lucky_case_indices.tolist()
        )
        # select the retrain data
        retrain_data = data.iloc[retrain_data_indices]

        if len(retrain_data) < 100:
            log.info("Retrain data is less than 100, skipping retrain", tab=2)
            #######################
            ##    SAVE RESULTS   ##
            #######################

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
                    "group": current_group_key,
                    "dataset": data_path.name,
                    **confidence_result.as_dict(),
                    **uncertainty_result.as_dict(),
                    **entropy_result.as_dict(),
                },
            )
            continue

        num_reject = len(retrain_data[retrain_data["reject"] == 1])
        num_accept = len(retrain_data) - num_reject
        original_num_reject = len(data[data["reject"] == 1])
        original_num_accept = len(data) - original_num_reject
        if num_accept == 0:
            log.info("No accept data, sampling from all data", tab=2)
            num_sample = num_reject // 4
            if num_sample > original_num_accept:
                num_sample = original_num_accept // 4
            retrain_data = retrain_data.append(data[data["reject"] == 0].sample(n=num_sample))
        elif num_reject == 0:
            log.info("No reject data, sampling from all data", tab=2)
            num_sample = num_accept // 4
            if num_sample > original_num_reject:
                num_sample = original_num_reject // 4
            retrain_data = retrain_data.append(data[data["reject"] == 1].sample(n=num_sample))

        log.info("Data", tab=2)
        log.info("Retrain Data Size: %s", len(retrain_data), tab=3)
        log.info("Original Data Size: %s", len(data), tab=3)
        log.info("Reduced by: %s", ratio_to_percentage_str(len(retrain_data) / len(data)), tab=3)
        assert len(retrain_data) > 0, "sanity check, retrain data should not be empty"
        assert len(retrain_data) != len(data), "sanity check, retrain data should not be the same as the original data"

        retrain_cpu_usage = CPUUsage()

        with Timer(name="Pipeline -- Retrain -- Window %d" % i) as timer:
            new_model_id = suid.uuid()
            new_model_dir = base_model_dir / new_model_id
            new_model_dir.mkdir(parents=True, exist_ok=True)
            new_model_path = new_model_dir / "model.pt"

            retrain_result = flashnet_simple.flashnet_train(
                model_path=new_model_path,
                dataset=retrain_data,
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

        retrain_cpu_usage.update()
        log.info("Performance", tab=2)
        log.info("Elapsed time: %s", timer.elapsed, tab=2)
        log.info("CPU Usage: %s", retrain_cpu_usage.result, tab=2)
        log.info("AUC: %s", retrain_result.auc, tab=2)

        assert len(retrain_data) == retrain_result.num_io, "sanity check, number of data should be the same as the number of input/output"

        model_path = retrain_result.model_path
        # model = flashnet_simple.load_model(model_path, device=device)
        model_group.add_model(model_path)

        #######################
        ##    SAVE RESULTS   ##
        #######################

        results = append_to_df(
            df=results,
            data={
                **eval_result.as_dict(),
                "num_io": len(data),
                "num_reject": len(data[data["reject"] == 1]),
                "elapsed_time": window_timer.elapsed + retrain_result.train_time,
                "train_time": retrain_result.train_time,
                "train_data_size": len(retrain_data),
                "prediction_time": pred_timer.elapsed,
                "type": "window",
                "window_id": i,
                "cpu_usage": predict_cpu_usage.result,
                "model_selection_time": 0.0,
                "group": current_group_key,
                "dataset": data_path.name,
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
