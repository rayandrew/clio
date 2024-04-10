import shutil
from collections import defaultdict
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd

from sklearn import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import torch

import shortuuid as suid
import typer

from clio.flashnet.preprocessing.add_filter import add_filter_v2
from clio.flashnet.training.shared import FlashnetTrainResult
import clio.flashnet.training.simple as flashnet_simple
from clio.flashnet.confidence import get_confidence_cases
from clio.flashnet.constants import FEATURE_COLUMNS
from clio.flashnet.entropy import get_entropy_result
from clio.flashnet.eval import PredictionResult, PredictionResults, flashnet_evaluate
from clio.flashnet.uncertainty import get_uncertainty_result

from clio.utils.cpu_usage import CPUUsage
from clio.utils.dataframe import append_to_df
from clio.utils.general import ratio_to_percentage_str, torch_set_seed
from clio.utils.indented_file import IndentedFile
from clio.utils.logging import LogLevel, log_global_setup
from clio.utils.path import rmdir
from clio.utils.timer import Timer, default_timer
from clio.utils.tqdm import tqdm
from clio.utils.trace_pd import trace_get_dataset_paths

app = typer.Typer(name="Exp -- Single -- Retrain -- Uncertainty Based")


class Matchmaker:

    def __init__(self):
        self.target = "reject"
        self.models: list[torch.nn.Module] = []
        self.datasets: list[pd.DataFrame] = []
        ## HIGH score -> GOOD.
        ## Covariate will be calculated inference time
        self.concept_score: list[float] = []

        self.forest_dict = None

    def add(
        self,
        model: torch.nn.Module,
        data: pd.DataFrame,
        prediction_batch_size: int | None = None,
        device: torch.device | None = None,
        prediction_threshold: float = 0.5,
        use_eval_dropout: bool = False,
    ):
        self.models.append(model)
        self.datasets.append(data)
        self.rank_concept(
            data, prediction_batch_size=prediction_batch_size, device=device, prediction_threshold=prediction_threshold, use_eval_dropout=use_eval_dropout
        )
        self.build_tree()

    def rank_concept(
        self,
        data,
        prediction_batch_size: int | None = None,
        device: torch.device | None = None,
        prediction_threshold: float = 0.5,
        use_eval_dropout: bool = False,
    ):
        # ranking the models based on ACC
        self.concept_score = [0] * len(self.models)
        for idx, model in tqdm(enumerate(self.models), desc="Ranking Models", leave=False):
            prediction_result = flashnet_simple.flashnet_predict(
                model,
                dataset=data,
                batch_size=prediction_batch_size,
                device=device,
                threshold=prediction_threshold,
                use_eval_dropout=use_eval_dropout,
                disable_tqdm=True,
            )
            eval_result = flashnet_evaluate(
                labels=prediction_result.labels,
                predictions=prediction_result.predictions,
                probabilities=prediction_result.probabilities,
            )
            self.concept_score[idx] = eval_result.accuracy

    def build_tree(self):
        whole_datasets = pd.concat(self.datasets)
        X = whole_datasets[FEATURE_COLUMNS]
        y = whole_datasets[self.target]

        from sklearn.ensemble import RandomForestClassifier

        forest = RandomForestClassifier(n_estimators=5, max_depth=10, random_state=42)
        # fit on whole data to create the tree
        self.forest = forest.fit(X, y)
        covariate_forest_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))

        # then loop for each model's dataset.
        # how much did this model's data end up at every leaf?
        for model_idx, dataset in enumerate(self.datasets):

            X_local = dataset[FEATURE_COLUMNS]
            X_leaf_forest = forest.apply(X_local)

            for instance_idx, instance_leaves in enumerate(X_leaf_forest):
                for tree_idx, tree_leaf_idx in enumerate(instance_leaves):
                    ## First index is the tree ID,
                    ## second index is the leaf ID
                    ## third index is the model ID -> A leaf will be populated by multiple models
                    covariate_forest_dict[tree_idx][tree_leaf_idx][model_idx] += 1
        self.forest_dict = covariate_forest_dict

    def calculate_borda_count(self, ranking):
        # Given an array of scores, will return an array containing ranking of each indices
        ## eg: [3,2,1] means that model 0 is the highest ranked (3 point), followed by 2 and 1.
        ## Will return [1,2,3] -> Means that model 0 is ranked 1, index 1 is ranked 2, index 2 is ranked 3
        n = len(ranking)
        borda_count = [0] * n

        sorted_indices = sorted(range(n), key=lambda i: ranking[i], reverse=True)

        for i, index in enumerate(sorted_indices):
            borda_count[index] += i + 1

        return borda_count

    def get_model_borda_count(self, covariate_score):
        concept_borda_count = self.calculate_borda_count(self.concept_score)

        covariate_borda_count = self.calculate_borda_count(covariate_score)

        combined_borda_count = [concept + covariate for concept, covariate in zip(concept_borda_count, covariate_borda_count)]
        # print("Covariate ranking " + str(covariate_score))
        # print("Covariate borda count" + str(covariate_borda_count))
        # print("Concept ranking " + str(self.concept_score))
        # print("Concept borda count" + str(concept_borda_count))
        # print("Combined borda count" + str(combined_borda_count))
        ## Return model with HIGHEST ranking.
        return combined_borda_count.index(min(combined_borda_count))

    def rank_covariate(self, data):
        X = data[FEATURE_COLUMNS]

        X_forest = self.forest.apply(X)
        model_weights = defaultdict(lambda: 0)

        for instance_idx, instance_forest in enumerate(X_forest):
            for tree_idx, tree_leaf_idx in enumerate(instance_forest):
                if tree_leaf_idx in self.forest_dict[tree_idx]:
                    datapoints_each_model = self.forest_dict[tree_idx][tree_leaf_idx]
                    for model_idx, datapoints in datapoints_each_model.items():
                        model_weights[model_idx] += datapoints

        model_rankings = []
        for i in range(len(self.concept_score)):
            if i in model_weights:
                model_rankings.append(model_weights[i])
            else:
                model_rankings.append(0)
        return model_rankings

    def get_model(self, data):
        # On inference time, concept will never change.
        # Covariate does.
        covariate_rank = self.rank_covariate(data)
        model_idx = self.get_model_borda_count(covariate_rank)
        model = self.models[model_idx]
        return model


@app.command()
def exp_matchmaker_all(
    data_dir: Annotated[
        Path, typer.Argument(help="The test data directory to use for prediction", exists=True, file_okay=False, dir_okay=True, resolve_path=True)
    ],
    output: Annotated[Path, typer.Option(help="The output path to write the results to")],
    # window_size: Annotated[str, typer.Option(help="The window size to use for prediction (in minute(s))", show_default=True)] = "10",
    log_level: Annotated[LogLevel, typer.Option(help="The log level to use")] = LogLevel.INFO,
    profile_name: Annotated[str, typer.Option(help="The profile name to use for prediction", show_default=True)] = "profile_v1",
    feat_name: Annotated[str, typer.Option(help="The feature name to use for prediction", show_default=True)] = "feat_v6_ts",
    learning_rate: Annotated[float, typer.Option(help="The learning rate to use for training", show_default=True)] = 0.0001,
    epochs: Annotated[int, typer.Option(help="The number of epochs to use for training", show_default=True)] = 5,
    batch_size: Annotated[int, typer.Option(help="The batch size to use for training", show_default=True)] = 32,
    prediction_batch_size: Annotated[int, typer.Option(help="The batch size to use for prediction", show_default=True)] = -1,
    # duration: Annotated[str, typer.Option(help="The duration to use for prediction (in minute(s))", show_default=True)] = "-1",
    seed: Annotated[int, typer.Option(help="The seed to use for random number generation", show_default=True)] = 3003,
    cuda: Annotated[int, typer.Option(help="Use CUDA for training and prediction", show_default=True)] = 0,
    threshold: Annotated[float, typer.Option(help="The threshold to use for prediction", show_default=True)] = 0.5,
    eval_confidence_threshold: Annotated[float, typer.Option(help="The confidence threshold to for evaluation", show_default=True)] = 0.1,
    drop_rate: Annotated[float, typer.Option(help="The drop rate to use for training", show_default=True)] = 0.0,
    use_eval_dropout: Annotated[bool, typer.Option(help="Use dropout for evaluation", show_default=True)] = False,
):
    print("MATCHMAKER ALL")
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
    current_group_key = suid.uuid()
    model: torch.nn.Module | torch.ScriptModule | None = None
    model_path: Path | str = ""
    prediction_results: PredictionResults = PredictionResults()

    ## To accomodate with the current pipeline, MATCHMAKER will function as follows:
    ## - An add function, to keep model_paths. Will also save the data used to train the model
    ## - A rerank function(newest data) that will rerank models ranking internally
    ## - A get_predictor that will return the most relevant model path to pass to built in function.

    matchmaker = Matchmaker()

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
                    drop_rate=drop_rate,
                    use_eval_dropout=use_eval_dropout,
                )
                model = flashnet_simple.load_model(model_path, device=device)
                matchmaker.add(
                    model,
                    data=data,
                    prediction_batch_size=prediction_batch_size,
                    device=device,
                    prediction_threshold=threshold,
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
            model = flashnet_simple.load_model(model_path, device=device)

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
                model = matchmaker.get_model(data)
                prediction_result = flashnet_simple.flashnet_predict(
                    model, dataset=data, batch_size=prediction_batch_size, device=device, threshold=threshold, use_eval_dropout=use_eval_dropout
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

        retrain_cpu_usage = CPUUsage()

        with Timer(name="Pipeline -- Retrain -- Window %d" % i) as timer:
            new_model_id = suid.uuid()
            new_model_dir = base_model_dir / new_model_id
            new_model_dir.mkdir(parents=True, exist_ok=True)
            new_model_path = new_model_dir / "model.pt"

            retrain_result = flashnet_simple.flashnet_train(
                model_path=new_model_path,
                dataset=data,
                retrain=True,
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
            model = flashnet_simple.load_model(new_model_path, device=device)
            matchmaker.add(
                model, data=data, prediction_batch_size=prediction_batch_size, device=device, prediction_threshold=threshold, use_eval_dropout=use_eval_dropout
            )

        retrain_cpu_usage.update()
        log.info("Pipeline Initial Model")
        log.info("Elapsed time: %s", timer.elapsed, tab=2)
        log.info("CPU Usage: %s", retrain_cpu_usage.result, tab=2)
        log.info("AUC: %s", retrain_result.auc, tab=2)

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
                "train_data_size": len(data),
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


@app.command()
def exp_matchmaker_batch(
    data_dir: Annotated[
        Path, typer.Argument(help="The test data directory to use for prediction", exists=True, file_okay=False, dir_okay=True, resolve_path=True)
    ],
    output: Annotated[Path, typer.Option(help="The output path to write the results to")],
    # window_size: Annotated[str, typer.Option(help="The window size to use for prediction (in minute(s))", show_default=True)] = "10",
    log_level: Annotated[LogLevel, typer.Option(help="The log level to use")] = LogLevel.INFO,
    profile_name: Annotated[str, typer.Option(help="The profile name to use for prediction", show_default=True)] = "profile_v1",
    feat_name: Annotated[str, typer.Option(help="The feature name to use for prediction", show_default=True)] = "feat_v6_ts",
    learning_rate: Annotated[float, typer.Option(help="The learning rate to use for training", show_default=True)] = 0.0001,
    epochs: Annotated[int, typer.Option(help="The number of epochs to use for training", show_default=True)] = 5,
    batch_size: Annotated[int, typer.Option(help="The batch size to use for training", show_default=True)] = 32,
    prediction_batch_size: Annotated[int, typer.Option(help="The batch size to use for prediction", show_default=True)] = -1,
    # duration: Annotated[str, typer.Option(help="The duration to use for prediction (in minute(s))", show_default=True)] = "-1",
    seed: Annotated[int, typer.Option(help="The seed to use for random number generation", show_default=True)] = 3003,
    cuda: Annotated[int, typer.Option(help="Use CUDA for training and prediction", show_default=True)] = 0,
    threshold: Annotated[float, typer.Option(help="The threshold to use for prediction", show_default=True)] = 0.5,
    eval_confidence_threshold: Annotated[float, typer.Option(help="The confidence threshold to for evaluation", show_default=True)] = 0.1,
    drop_rate: Annotated[float, typer.Option(help="The drop rate to use for training", show_default=True)] = 0.0,
    use_eval_dropout: Annotated[bool, typer.Option(help="Use dropout for evaluation", show_default=True)] = False,
):
    args = locals()

    global_start_time = default_timer()

    output.mkdir(parents=True, exist_ok=True)
    log = log_global_setup(output / "log.txt", level=log_level)

    if "single" in str(output):
        prediction_batch_size = 1

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

    log.info("Matchmaker prediction batch: %s", prediction_batch_size)

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
    current_group_key = suid.uuid()
    model: torch.nn.Module | torch.ScriptModule | None = None
    model_path: Path | str = ""
    prediction_results: PredictionResults = PredictionResults()

    ## To accomodate with the current pipeline, MATCHMAKER will function as follows:
    ## - An add function, to keep model_paths. Will also save the data used to train the model
    ## - A rerank function(newest data) that will rerank models ranking internally
    ## - A get_predictor that will return the most relevant model path to pass to built in function.

    matchmaker = Matchmaker()

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
                    drop_rate=drop_rate,
                    use_eval_dropout=use_eval_dropout,
                )
                model = flashnet_simple.load_model(model_path, device=device)
                matchmaker.add(
                    model,
                    data=data,
                    prediction_batch_size=prediction_batch_size,
                    device=device,
                    prediction_threshold=threshold,
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
            model = flashnet_simple.load_model(model_path, device=device)

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
                num_instances = len(data)
                num_iterations = (num_instances + prediction_batch_size - 1) // prediction_batch_size
                predictions = []
                labels = []
                probabilities = []

                for iter in tqdm(range(num_iterations), desc="Prediction", unit="batch", leave=False):
                    start_idx = iter * prediction_batch_size
                    end_idx = min((iter + 1) * prediction_batch_size, num_instances)

                    indices = range(start_idx, end_idx)

                    batch_data = data.loc[indices]

                    model = matchmaker.get_model(batch_data)
                    pred_temp = flashnet_simple.flashnet_predict(
                        model=model,
                        dataset=batch_data,
                        device=device,
                        batch_size=prediction_batch_size,
                        threshold=threshold,
                        use_eval_dropout=use_eval_dropout,
                        disable_tqdm=True,
                    )

                    predictions.extend(pred_temp.predictions)
                    labels.extend(pred_temp.labels)
                    probabilities.extend(pred_temp.probabilities)

                predictions = np.array(predictions)
                labels = np.array(labels)
                probabilities = np.array(probabilities)
                prediction_result = PredictionResult(predictions=predictions, labels=labels, probabilities=probabilities)

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

        retrain_cpu_usage = CPUUsage()

        with Timer(name="Pipeline -- Retrain -- Window %d" % i) as timer:
            new_model_id = suid.uuid()
            new_model_dir = base_model_dir / new_model_id
            new_model_dir.mkdir(parents=True, exist_ok=True)
            new_model_path = new_model_dir / "model.pt"

            retrain_result = flashnet_simple.flashnet_train(
                model_path=new_model_path,
                dataset=data,
                retrain=True,
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
            model = flashnet_simple.load_model(new_model_path, device=device)
            matchmaker.add(
                model, data=data, prediction_batch_size=prediction_batch_size, device=device, prediction_threshold=threshold, use_eval_dropout=use_eval_dropout
            )

        retrain_cpu_usage.update()
        log.info("Pipeline Initial Model")
        log.info("Elapsed time: %s", timer.elapsed, tab=2)
        log.info("CPU Usage: %s", retrain_cpu_usage.result, tab=2)
        log.info("AUC: %s", retrain_result.auc, tab=2)

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
                "train_data_size": len(data),
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


class MatchmakerScikit:

    def __init__(self, base_estimator):
        self.target = "reject"
        self.models: list[torch.nn.Module] = []
        self.datasets: list[pd.DataFrame] = []
        ## HIGH score -> GOOD.
        ## Covariate will be calculated inference time
        self.concept_score: list[float] = []

        self.forest_dict = None
        self.base_estimator = base_estimator

    def fit(
        self,
        data: pd.DataFrame,
    ):
        data_train = data.copy()
        data_train = add_filter_v2(data_train)
        ## clone base estimator, fit on data, and add to models
        model = clone(self.base_estimator)
        model.fit(data_train[FEATURE_COLUMNS], data_train[self.target])
        
        self.models.append(model)
        self.datasets.append(data)
        self.rank_concept(data)
        self.build_tree()

    def rank_concept(
        self,
        data,
    ):
        # ranking the models based on ACC
        self.concept_score = [0] * len(self.models)
        for idx, model in tqdm(enumerate(self.models), desc="Ranking Models", leave=False):
            prediction_result = model.predict(data[FEATURE_COLUMNS])
            
            self.concept_score[idx] = accuracy_score(data[self.target], prediction_result)

    def build_tree(self):
        whole_datasets = pd.concat(self.datasets)
        X = whole_datasets[FEATURE_COLUMNS]
        y = whole_datasets[self.target]

        from sklearn.ensemble import RandomForestClassifier

        forest = RandomForestClassifier(n_estimators=5, max_depth=10, random_state=42)
        # fit on whole data to create the tree
        self.forest = forest.fit(X, y)
        covariate_forest_dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: 0)))

        # then loop for each model's dataset.
        # how much did this model's data end up at every leaf?
        for model_idx, dataset in enumerate(self.datasets):

            X_local = dataset[FEATURE_COLUMNS]
            X_leaf_forest = forest.apply(X_local)

            for instance_idx, instance_leaves in enumerate(X_leaf_forest):
                for tree_idx, tree_leaf_idx in enumerate(instance_leaves):
                    ## First index is the tree ID,
                    ## second index is the leaf ID
                    ## third index is the model ID -> A leaf will be populated by multiple models
                    covariate_forest_dict[tree_idx][tree_leaf_idx][model_idx] += 1
        self.forest_dict = covariate_forest_dict

    def calculate_borda_count(self, ranking):
        # Given an array of scores, will return an array containing ranking of each indices
        ## eg: [3,2,1] means that model 0 is the highest ranked (3 point), followed by 2 and 1.
        ## Will return [1,2,3] -> Means that model 0 is ranked 1, index 1 is ranked 2, index 2 is ranked 3
        n = len(ranking)
        borda_count = [0] * n

        sorted_indices = sorted(range(n), key=lambda i: ranking[i], reverse=True)

        for i, index in enumerate(sorted_indices):
            borda_count[index] += i + 1

        return borda_count

    def get_model_borda_count(self, covariate_score):
        concept_borda_count = self.calculate_borda_count(self.concept_score)

        covariate_borda_count = self.calculate_borda_count(covariate_score)

        combined_borda_count = [concept + covariate for concept, covariate in zip(concept_borda_count, covariate_borda_count)]
        return combined_borda_count.index(min(combined_borda_count))

    def rank_covariate(self, data):
        X = data[FEATURE_COLUMNS]

        X_forest = self.forest.apply(X)
        model_weights = defaultdict(lambda: 0)

        for instance_idx, instance_forest in enumerate(X_forest):
            for tree_idx, tree_leaf_idx in enumerate(instance_forest):
                if tree_leaf_idx in self.forest_dict[tree_idx]:
                    datapoints_each_model = self.forest_dict[tree_idx][tree_leaf_idx]
                    for model_idx, datapoints in datapoints_each_model.items():
                        model_weights[model_idx] += datapoints

        model_rankings = []
        for i in range(len(self.concept_score)):
            if i in model_weights:
                model_rankings.append(model_weights[i])
            else:
                model_rankings.append(0)
        return model_rankings

    def get_model(self, data):
        # On inference time, concept will never change.
        # Covariate does.
        covariate_rank = self.rank_covariate(data)
        model_idx = self.get_model_borda_count(covariate_rank)
        model = self.models[model_idx]
        return model
    
    def predict(self, data):
        model = self.get_model(data)
        prediction_proba = model.predict_proba(data[FEATURE_COLUMNS])
        prediction_result = prediction_proba[:, 1] > 0.5
        prediction_proba_single = prediction_proba[:, 1]
        return prediction_result, prediction_proba_single

@app.command()
def exp_matchmaker_scikit(
    data_dir: Annotated[
        Path, typer.Argument(help="The test data directory to use for prediction", exists=True, file_okay=False, dir_okay=True, resolve_path=True)
    ],
    output: Annotated[Path, typer.Option(help="The output path to write the results to")],
    # window_size: Annotated[str, typer.Option(help="The window size to use for prediction (in minute(s))", show_default=True)] = "10",
    log_level: Annotated[LogLevel, typer.Option(help="The log level to use")] = LogLevel.INFO,
    profile_name: Annotated[str, typer.Option(help="The profile name to use for prediction", show_default=True)] = "profile_v1",
    feat_name: Annotated[str, typer.Option(help="The feature name to use for prediction", show_default=True)] = "feat_v6_ts",
    learning_rate: Annotated[float, typer.Option(help="The learning rate to use for training", show_default=True)] = 0.0001,
    epochs: Annotated[int, typer.Option(help="The number of epochs to use for training", show_default=True)] = 5,
    batch_size: Annotated[int, typer.Option(help="The batch size to use for training", show_default=True)] = 32,
    prediction_batch_size: Annotated[int, typer.Option(help="The batch size to use for prediction", show_default=True)] = -1,
    # duration: Annotated[str, typer.Option(help="The duration to use for prediction (in minute(s))", show_default=True)] = "-1",
    seed: Annotated[int, typer.Option(help="The seed to use for random number generation", show_default=True)] = 3003,
    cuda: Annotated[int, typer.Option(help="Use CUDA for training and prediction", show_default=True)] = 0,
    threshold: Annotated[float, typer.Option(help="The threshold to use for prediction", show_default=True)] = 0.5,
    eval_confidence_threshold: Annotated[float, typer.Option(help="The confidence threshold to for evaluation", show_default=True)] = 0.1,
    drop_rate: Annotated[float, typer.Option(help="The drop rate to use for training", show_default=True)] = 0.0,
    use_eval_dropout: Annotated[bool, typer.Option(help="Use dropout for evaluation", show_default=True)] = False,
):
    args = locals()

    global_start_time = default_timer()

    output.mkdir(parents=True, exist_ok=True)
    log = log_global_setup(output / "log.txt", level=log_level)


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
    current_group_key = suid.uuid()
    model: torch.nn.Module | torch.ScriptModule | None = None
    model_path: Path | str = ""
    prediction_results: PredictionResults = PredictionResults()

    ## AUE with scikit_learn's base estimator. Need to be able to fit(), partial_fit(), and predict_proba().
    ## No model saving is done
    
    matchmaker = MatchmakerScikit(base_estimator=RandomForestClassifier(random_state=42))

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

            with Timer(name="Pipeline -- Initial Model Training -- Window %d" % i) as timer_train_init:
                matchmaker.fit(data)
            with Timer(name="Pipeline -- Initial Model Prediction -- Window %d" % i) as timer_infer_init:
                temp_label, temp_proba = matchmaker.predict(data[FEATURE_COLUMNS])
                temp_label = np.array(temp_label)
                temp_proba = np.array(temp_proba)
                
            eval_result = flashnet_evaluate(labels=data["reject"], predictions=temp_label, probabilities=temp_proba)
            train_result = FlashnetTrainResult(
                **eval_result.as_dict(),
                stats=eval_result.stats,
                train_time=timer_train_init.elapsed,
                prediction_time=timer_infer_init.elapsed,
                model_path=model_path,
                norm_mean=None,
                norm_std=None,
                ip_threshold=None,
                confidence_threshold=None,
                labels=np.array(data["reject"]),
                predictions=temp_label,
                probabilities=temp_proba,
            )
            
            train_cpu_usage.update()
            log.info("Pipeline Initial Model")
            log.info("Elapsed time: %s", timer_train_init.elapsed, tab=2)
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
                    "elapsed_time": timer_train_init.elapsed,
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

            # model_path = train_result.model_path
            # model = flashnet_simple.load_model(model_path, device=device)

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
                predictions, probabilities = matchmaker.predict(data[FEATURE_COLUMNS])
                labels = data["reject"]

                predictions = np.array(predictions)
                labels = np.array(labels)
                probabilities = np.array(probabilities)
                prediction_result = PredictionResult(predictions=predictions, labels=labels, probabilities=probabilities)

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

        retrain_cpu_usage = CPUUsage()

        with Timer(name="Pipeline -- Retrain -- Window %d" % i) as timer:
            matchmaker.fit(data)
        
        ## TODO: Retrain metrics is nonsense, need to be fixed. 
        ## just to satisfy pipeline
        new_model_id = suid.uuid()
        new_model_dir = base_model_dir / new_model_id
        new_model_dir.mkdir(parents=True, exist_ok=True)
        new_model_path = new_model_dir / "model.pt"
        
        retrain_result = flashnet_simple.flashnet_train(
            model_path=new_model_path,
            dataset=data,
            retrain=True,
            batch_size=batch_size,
            prediction_batch_size=prediction_batch_size,
            lr=learning_rate,
            epochs=1,
            norm_mean=None,
            norm_std=None,
            n_data=None,
            device=device,
            drop_rate=drop_rate,
            use_eval_dropout=use_eval_dropout,
            disable_tqdm=True
        )

        retrain_cpu_usage.update()
        log.info("Pipeline Initial Model")
        log.info("Elapsed time: %s", timer.elapsed, tab=2)
        log.info("CPU Usage: %s", retrain_cpu_usage.result, tab=2)
        log.info("AUC: %s", retrain_result.auc, tab=2)

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
                "train_data_size": len(data),
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
