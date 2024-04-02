import io
import json
import shutil
import sys
from collections import UserDict
from dataclasses import dataclass, field
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

import shortuuid as suid
import typer
from serde import serde
from serde.json import from_json, to_json
from serde.msgpack import from_msgpack, to_msgpack

import clio.flashnet.training.ensemble as flashnet_ensemble
import clio.flashnet.training.simple as flashnet_simple
from clio.flashnet.confidence import get_confidence_cases
from clio.flashnet.constants import FEATURE_COLUMNS
from clio.flashnet.eval import flashnet_evaluate

from clio.utils.characteristic import Characteristic, characteristic_from_df
from clio.utils.cpu_usage import CPUUsage
from clio.utils.dataframe import append_to_df
from clio.utils.general import parse_time, ratio_to_percentage_str, torch_set_seed
from clio.utils.indented_file import IndentedFile
from clio.utils.logging import LogLevel, log_get, log_global_setup
from clio.utils.path import rmdir
from clio.utils.timer import Timer, default_timer
from clio.utils.trace_pd import trace_get_dataset_paths

app = typer.Typer(name="Student Management", pretty_exceptions_enable=False)


@serde
@dataclass
class Model:
    id: str
    path: Path

    def __str__(self):
        return f"Model(id={self.id}, path={self.path})"


@serde
@dataclass
class ModelGroup:
    characteristic: Characteristic
    models: list[Model] = field(default_factory=list)
    confidences: list[float] = field(default_factory=list)

    def add_model(self, model: Model, confidence: float):
        self.models.append(model)
        self.confidences.append(confidence)

    def to_indented_file(self, file: IndentedFile):
        with file.section("Models"):
            for i, model in enumerate(self.models):
                with file.section("Model %s", i):
                    file.writeln("Path: %s", model.path)
                    file.writeln("Confidence: %s", self.confidences[i])

    def to_msgpack(self, path: Path | str | io.BufferedIOBase):
        if isinstance(path, (str, Path)):
            path = Path(path)
            with path.open("wb") as f:
                f.write(to_msgpack(self))
            return

        path.write(to_msgpack(self))

    @staticmethod
    def from_msgpack(path: Path | str | io.BufferedIOBase) -> "ModelGroup":
        if isinstance(path, (str, Path)):
            path = Path(path)
            with path.open("rb") as f:
                return from_msgpack(f.read())
        return from_msgpack(path)


@serde
@dataclass
class ModelZoo(UserDict[str, ModelGroup]):
    data: dict[str, ModelGroup] = field(default_factory=dict)

    def to_msgpack(self, path: Path | str | io.BufferedIOBase):
        if isinstance(path, (str, Path)):
            path = Path(path)
            with path.open("wb") as f:
                f.write(to_msgpack(self))
            return

        path.write(to_msgpack(self))

    @staticmethod
    def from_msgpack(path: Path | str | io.BufferedIOBase) -> "ModelZoo":
        if isinstance(path, (str, Path)):
            path = Path(path)
            with path.open("rb") as f:
                return from_msgpack(f.read())
        return from_msgpack(path)


def group_prediction(group: ModelGroup, data: pd.DataFrame, batch_size: int, device: torch.device | None = None):
    results: list[flashnet_simple.PredictionResult] = []
    for model in group.models:
        model = flashnet_simple.load_model(model.path, device=device)
        result = flashnet_simple.flashnet_predict(model, data, batch_size=batch_size, device=device)
        results.append(result)

    if len(results) == 1:
        return results[0]

    final_probs = np.zeros(len(data), dtype=np.float32)
    for i, (confidence, result) in enumerate(zip(group.confidences, results)):
        final_probs += confidence * result.probabilities

    final_pred = (final_probs > 0.5).astype(np.int32)

    return flashnet_simple.PredictionResult(
        labels=results[0].labels,
        predictions=final_pred,
        probabilities=final_probs,
    )


def zoo_prediction(zoo: ModelZoo, data: pd.DataFrame, batch_size: int, device: torch.device | None = None):
    # choose group based on characteristic of the instance of data

    data = data.copy()
    final_data = data.copy()
    final_data["prediction"] = 0

    # iterate through the data
    # for each data, get the characteristic
    # when doing prediction, we only know FEATURE_COLUMNS

    # data = data[FEATURE_COLUMNS]

    def choose_group(features: pd.Series):
        # _log.info("Choosing group based on features: %s", features)
        selected_groups: list[ModelGroup] = []
        for group in zoo.values():
            # if group.characteristic is closer to the instance size, then use that group
            # if np.isclose(group.characteristic.size.avg, features["size"], rtol=500):
            size_upper_iqr = group.characteristic.size.upper_iqr_bound
            size_lower_iqr = group.characteristic.size.lower_iqr_bound
            if size_lower_iqr <= features["size"] <= size_upper_iqr:
                selected_groups.append(group)

        if len(selected_groups) == 0:
            # randomly pick a group
            return np.random.choice(list(zoo.values()))

        if len(selected_groups) == 1:
            return selected_groups[0]

        # if there are multiple groups, then choose the one with the closest size
        selected_group = selected_groups[0]
        for group in selected_groups[1:]:
            if np.abs(group.characteristic.size.avg - features["size"]) < np.abs(selected_group.characteristic.size.avg - features["size"]):
                selected_group = group

        return selected_group

    data["group"] = data[FEATURE_COLUMNS].apply(lambda x: choose_group(x), axis=1)

    # for each group, do prediction
    results: list[flashnet_simple.PredictionResult] = []
    labels = np.zeros(len(data), dtype=np.int32)
    predictions = np.zeros(len(data), dtype=np.int32)
    probabilities = np.zeros(len(data), dtype=np.float32)
    # indices: list[list[int]] = []
    for group in zoo.values():
        group_data = data[data["group"] == group]
        # indices.append(group_data.index)
        result = group_prediction(group, group_data, batch_size=batch_size, device=device)
        final_data.loc[group_data.index, "prediction"] = result.predictions
        labels[group_data.index] = result.labels
        predictions[group_data.index] = result.predictions
        probabilities[group_data.index] = result.probabilities
        # results.append(result)

    # reconstruct the results to data back
    # data["reject"] = 0
    # for i, group_data in enumerate(data["group"]):
    #     data.loc[indices[i], "reject"] = results[i].predictions

    if len(results) == 1:
        return results[0]

    return flashnet_simple.PredictionResult(
        labels=labels,
        predictions=predictions,
        probabilities=probabilities,
    )


@app.command()
def exp_initial_only(
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
    admission_confidence_threshold: Annotated[float, typer.Option(help="The confidence threshold to for admission", show_default=True)] = 0.7,
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

    base_model_dir = output / "models"
    # NOTE: Remove the base model directory if it exists
    rmdir(base_model_dir)
    base_model_dir.mkdir(parents=True, exist_ok=True)

    ifile = IndentedFile(output / "stats.txt")
    model_zoo: ModelZoo = ModelZoo()
    current_group_key = suid.uuid()

    for i, data_path in enumerate(data_paths):
        log.info("Processing dataset: %s", data_path, tab=1)
        data = pd.read_csv(data_path)
        log.info("Length of data: %s", len(data), tab=2)
        if i == 0:
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
                confidence_threshold=eval_confidence_threshold,
            )

            log.info("Confidence", tab=2)
            log.info("Best Case: %s", ratio_to_percentage_str(confidence_result.best_case_ratio), tab=3)
            log.info("Worst Case: %s", ratio_to_percentage_str(confidence_result.worst_case_ratio), tab=3)
            log.info("Clueless Case: %s", ratio_to_percentage_str(confidence_result.clueless_case_ratio), tab=3)
            log.info("Lucky Case: %s", ratio_to_percentage_str(confidence_result.lucky_case_ratio), tab=3)

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
                    "group": current_group_key,
                    "dataset": data_path.name,
                    **confidence_result.as_dict(),
                },
            )

            assert train_result.model_path == model_path, "sanity check, model path should be the same as the initial model path"
            # model = flashnet_simple.load_model(model_path, device=device)
            if current_group_key not in model_zoo:
                model_zoo[current_group_key] = ModelGroup(characteristic=characteristic_from_df(data))
            model_zoo[current_group_key].add_model(Model(id=model_id, path=model_path), confidence=confidence_result.best_case_ratio)

            with ifile.section("Window 0"):
                with ifile.section("Evaluation"):
                    train_result.to_indented_file(ifile)
                with ifile.section("Confidence Analysis"):
                    confidence_result.to_indented_file(ifile)
                with ifile.section("Model Group"):
                    model_zoo[current_group_key].to_indented_file(ifile)

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
                # prediction_result = group_prediction(model_zoo[current_group_key], data=data, device=device, batch_size=prediction_batch_size)
                prediction_result = zoo_prediction(zoo=model_zoo, data=data, device=device, batch_size=prediction_batch_size)
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
                    with ifile.section("Model Group"):
                        model_zoo[current_group_key].to_indented_file(ifile)

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
            },
        )

        #######################
        ##     ADMISSIONS    ##
        #######################

        # THERE ARE 2 CASES:
        #   CASE 1. If MODEL PERFORMANCE is low, then we CREATE a new group and admit the NEW student to the new group
        #   CASE 2. If the confidence is low BUT MODEL PERFORMANCE is high, then we admit the NEW student to the current group
        do_admission = True
        if eval_result.accuracy < 0.8:
            # CASE 1
            # use the whole data for training
            training_data = data
            log.info("CASE 1 -- Admissions", tab=1)

            current_group_key = suid.uuid()
            model_zoo[current_group_key] = ModelGroup(characteristic=characteristic_from_df(training_data))
            log.info("Creating new group: %s", current_group_key, tab=2)
            data_ratio = 1.0

            log.info("Training Data", tab=2)
            log.info("Length of data: %s", len(training_data), tab=3)
            log.info("Data Ratio: %s", data_ratio, tab=3)
            log.info("Num reject in training data: %s", len(training_data[training_data["reject"] == 1]), tab=3)
            log.info("Num accept in training data: %s", len(training_data[training_data["reject"] == 0]), tab=3)
        elif confidence_result.best_case_ratio < admission_confidence_threshold:
            # CASE 2

            log.info("CASE 2 -- Admissions", tab=1)
            worst_case_indices = confidence_result.worst_case_indices
            worst_case_data = data.iloc[worst_case_indices]
            clueless_case_data = data.iloc[confidence_result.clueless_case_indices]
            lucky_case_data = data.iloc[confidence_result.lucky_case_indices]
            training_data = pd.concat([worst_case_data, clueless_case_data, lucky_case_data], ignore_index=True)
            num_reject = len(training_data[training_data["reject"] == 1])
            num_accept = len(training_data) - num_reject
            original_num_reject = len(data[data["reject"] == 1])
            original_num_accept = len(data) - original_num_reject

            if num_accept == 0:
                log.info("Sampling Accept Data", tab=1)
                # sample accept data from the rest of the data
                num_sample = num_reject // 4
                if num_sample > original_num_accept:
                    num_sample = original_num_accept // 4
                accept_data = data[data["reject"] == 0].sample(n=num_sample, random_state=seed)
                training_data = pd.concat([worst_case_data, clueless_case_data, lucky_case_data, accept_data], ignore_index=True)
            elif num_reject == 0:
                log.info("Sampling Reject Data", tab=2)
                # sample reject data from the rest of the data
                num_sample = num_accept // 4
                if num_sample > original_num_reject:
                    num_sample = original_num_reject // 4
                reject_data = data[data["reject"] == 1].sample(n=num_sample, random_state=seed)
                training_data = pd.concat([worst_case_data, clueless_case_data, lucky_case_data, reject_data], ignore_index=True)

                data_ratio = (len(worst_case_indices) + len(clueless_case_data) + len(lucky_case_data)) / len(data)
                log.info("Training Data", tab=2)
                log.info("Length of data: %s", len(training_data), tab=3)
                log.info("Data Ratio: %s", data_ratio, tab=3)
                log.info("Num reject in training data: %s", num_reject, tab=3)
                log.info("Num accept in training data: %s", num_accept, tab=3)
        else:
            do_admission = False

        if do_admission:
            log.info("Admitting to group: %s", current_group_key, tab=2)

            num_reject = len(training_data[training_data["reject"] == 1])
            num_accept = len(training_data) - num_reject
            assert num_reject != 0 and num_accept != 0, "sanity check, there should be reject and accept data in the training data"

            with Timer(name="Pipeline -- Admissions -- Window %s" % i) as admission_timer:
                model_id = suid.uuid()
                model_dir = base_model_dir / model_id
                model_dir.mkdir(parents=True, exist_ok=True)
                model_path = model_dir / "model.pt"
                train_result = flashnet_simple.flashnet_train(
                    model_path=model_path,
                    dataset=training_data,
                    retrain=True,
                    batch_size=batch_size,
                    prediction_batch_size=prediction_batch_size,
                    # tqdm=True,
                    lr=learning_rate,
                    epochs=epochs,
                    norm_mean=None,
                    norm_std=None,
                    n_data=None,
                    device=device,
                )
                if current_group_key not in model_zoo:
                    model_zoo[current_group_key] = ModelGroup()
                model_zoo[current_group_key].add_model(Model(id=model_id, path=model_path), confidence=data_ratio * confidence_result.best_case_ratio)

            log.info("Training", tab=2)
            log.info("Elapsed time: %s", admission_timer.elapsed, tab=3)
            log.info("AUC: %s", train_result.auc, tab=3)

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

            with ifile.block():
                with ifile.section("Admissions"):
                    with ifile.section("Evaluation"):
                        with ifile.section("Model Performance"):
                            train_result.to_indented_file(ifile)
                        with ifile.section("Confidence Analysis"):
                            confidence_result.to_indented_file(ifile)
                with ifile.section("Model Group"):
                    model_zoo[current_group_key].to_indented_file(ifile)

        results.to_csv(output / "results.csv", index=False)
        ifile.flush()
        with open(output / "model_zoo.json", "w") as f:
            f.write(to_json(model_zoo))

        # current_group_key += 1

        # if i == 1:
        #     break

    results.to_csv(output / "results.csv", index=False)
    ifile.close()
    with open(output / "model_zoo.json", "w") as f:
        f.write(to_json(model_zoo))

    global_end_time = default_timer()
    log.info("Total elapsed time: %s s", global_end_time - global_start_time, tab=0)


@app.command()
def exp_recent(
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
    admission_confidence_threshold: Annotated[float, typer.Option(help="The confidence threshold to for admission", show_default=True)] = 0.7,
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

    base_model_dir = output / "models"
    # NOTE: Remove the base model directory if it exists
    rmdir(base_model_dir)
    base_model_dir.mkdir(parents=True, exist_ok=True)

    ifile = IndentedFile(output / "stats.txt")
    model_zoo: ModelZoo = ModelZoo()
    current_group_key = suid.uuid()

    for i, data_path in enumerate(data_paths):
        log.info("Processing dataset: %s", data_path, tab=1)
        data = pd.read_csv(data_path)
        log.info("Length of data: %s", len(data), tab=2)
        if i == 0:
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
                confidence_threshold=eval_confidence_threshold,
            )

            log.info("Confidence", tab=2)
            log.info("Best Case: %s", ratio_to_percentage_str(confidence_result.best_case_ratio), tab=3)
            log.info("Worst Case: %s", ratio_to_percentage_str(confidence_result.worst_case_ratio), tab=3)
            log.info("Clueless Case: %s", ratio_to_percentage_str(confidence_result.clueless_case_ratio), tab=3)
            log.info("Lucky Case: %s", ratio_to_percentage_str(confidence_result.lucky_case_ratio), tab=3)

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
                    "group": current_group_key,
                    "dataset": data_path.name,
                    **confidence_result.as_dict(),
                },
            )

            assert train_result.model_path == model_path, "sanity check, model path should be the same as the initial model path"
            # model = flashnet_simple.load_model(model_path, device=device)
            if current_group_key not in model_zoo:
                model_zoo[current_group_key] = ModelGroup(characteristic=characteristic_from_df(data))
            model_zoo[current_group_key].add_model(Model(id=model_id, path=model_path), confidence=confidence_result.best_case_ratio)

            with ifile.section("Window 0"):
                with ifile.section("Evaluation"):
                    train_result.to_indented_file(ifile)
                with ifile.section("Confidence Analysis"):
                    confidence_result.to_indented_file(ifile)
                with ifile.section("Model Group"):
                    model_zoo[current_group_key].to_indented_file(ifile)

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
                prediction_result = group_prediction(model_zoo[current_group_key], data=data, device=device, batch_size=prediction_batch_size)
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
                    with ifile.section("Model Group"):
                        model_zoo[current_group_key].to_indented_file(ifile)

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
            },
        )

        #######################
        ##     ADMISSIONS    ##
        #######################

        # THERE ARE 2 CASES:
        #   CASE 1. If MODEL PERFORMANCE is low, then we CREATE a new group and admit the NEW student to the new group
        #   CASE 2. If the confidence is low BUT MODEL PERFORMANCE is high, then we admit the NEW student to the current group
        do_admission = True
        if eval_result.accuracy < 0.8:
            # CASE 1
            # use the whole data for training
            training_data = data
            log.info("CASE 1 -- Admissions", tab=1)

            current_group_key = suid.uuid()
            model_zoo[current_group_key] = ModelGroup(characteristic=characteristic_from_df(training_data))
            log.info("Creating new group: %s", current_group_key, tab=2)
            data_ratio = 1.0

            log.info("Training Data", tab=2)
            log.info("Length of data: %s", len(training_data), tab=3)
            log.info("Data Ratio: %s", data_ratio, tab=3)
            log.info("Num reject in training data: %s", len(training_data[training_data["reject"] == 1]), tab=3)
            log.info("Num accept in training data: %s", len(training_data[training_data["reject"] == 0]), tab=3)
        elif confidence_result.best_case_ratio < admission_confidence_threshold:
            # CASE 2

            log.info("CASE 2 -- Admissions", tab=1)
            worst_case_indices = confidence_result.worst_case_indices
            worst_case_data = data.iloc[worst_case_indices]
            clueless_case_data = data.iloc[confidence_result.clueless_case_indices]
            lucky_case_data = data.iloc[confidence_result.lucky_case_indices]
            training_data = pd.concat([worst_case_data, clueless_case_data, lucky_case_data], ignore_index=True)
            num_reject = len(training_data[training_data["reject"] == 1])
            num_accept = len(training_data) - num_reject
            original_num_reject = len(data[data["reject"] == 1])
            original_num_accept = len(data) - original_num_reject

            if num_accept == 0:
                log.info("Sampling Accept Data", tab=1)
                # sample accept data from the rest of the data
                num_sample = num_reject // 4
                if num_sample > original_num_accept:
                    num_sample = original_num_accept // 4
                accept_data = data[data["reject"] == 0].sample(n=num_sample, random_state=seed)
                training_data = pd.concat([worst_case_data, clueless_case_data, lucky_case_data, accept_data], ignore_index=True)
            elif num_reject == 0:
                log.info("Sampling Reject Data", tab=2)
                # sample reject data from the rest of the data
                num_sample = num_accept // 4
                if num_sample > original_num_reject:
                    num_sample = original_num_reject // 4
                reject_data = data[data["reject"] == 1].sample(n=num_sample, random_state=seed)
                training_data = pd.concat([worst_case_data, clueless_case_data, lucky_case_data, reject_data], ignore_index=True)

                data_ratio = (len(worst_case_indices) + len(clueless_case_data) + len(lucky_case_data)) / len(data)
                log.info("Training Data", tab=2)
                log.info("Length of data: %s", len(training_data), tab=3)
                log.info("Data Ratio: %s", data_ratio, tab=3)
                log.info("Num reject in training data: %s", num_reject, tab=3)
                log.info("Num accept in training data: %s", num_accept, tab=3)
        else:
            do_admission = False

        if do_admission:
            log.info("Admitting to group: %s", current_group_key, tab=2)

            num_reject = len(training_data[training_data["reject"] == 1])
            num_accept = len(training_data) - num_reject
            assert num_reject != 0 and num_accept != 0, "sanity check, there should be reject and accept data in the training data"

            with Timer(name="Pipeline -- Admissions -- Window %s" % i) as admission_timer:
                model_id = suid.uuid()
                model_dir = base_model_dir / model_id
                model_dir.mkdir(parents=True, exist_ok=True)
                model_path = model_dir / "model.pt"
                train_result = flashnet_simple.flashnet_train(
                    model_path=model_path,
                    dataset=training_data,
                    retrain=True,
                    batch_size=batch_size,
                    prediction_batch_size=prediction_batch_size,
                    # tqdm=True,
                    lr=learning_rate,
                    epochs=epochs,
                    norm_mean=None,
                    norm_std=None,
                    n_data=None,
                    device=device,
                )
                if current_group_key not in model_zoo:
                    model_zoo[current_group_key] = ModelGroup()
                model_zoo[current_group_key].add_model(Model(id=model_id, path=model_path), confidence=data_ratio * confidence_result.best_case_ratio)

            log.info("Training", tab=2)
            log.info("Elapsed time: %s", admission_timer.elapsed, tab=3)
            log.info("AUC: %s", train_result.auc, tab=3)

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

            with ifile.block():
                with ifile.section("Admissions"):
                    with ifile.section("Evaluation"):
                        with ifile.section("Model Performance"):
                            train_result.to_indented_file(ifile)
                        with ifile.section("Confidence Analysis"):
                            confidence_result.to_indented_file(ifile)
                with ifile.section("Model Group"):
                    model_zoo[current_group_key].to_indented_file(ifile)

        results.to_csv(output / "results.csv", index=False)
        ifile.flush()
        with open(output / "model_zoo.json", "w") as f:
            f.write(to_json(model_zoo))

    results.to_csv(output / "results.csv", index=False)
    ifile.close()
    with open(output / "model_zoo.json", "w") as f:
        f.write(to_json(model_zoo))

    global_end_time = default_timer()
    log.info("Total elapsed time: %s s", global_end_time - global_start_time, tab=0)


if __name__ == "__main__":
    app()
