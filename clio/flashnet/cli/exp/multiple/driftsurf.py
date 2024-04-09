from collections import defaultdict
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd

import torch

import shortuuid as suid
import typer

app = typer.Typer()

import clio.flashnet.training.ensemble as flashnet_ensemble
import clio.flashnet.training.simple as flashnet_simple
from clio.flashnet.confidence import get_confidence_cases
from clio.flashnet.constants import FEATURE_COLUMNS
from clio.flashnet.entropy import get_entropy_result
from clio.flashnet.eval import PredictionResults, flashnet_evaluate
from clio.flashnet.uncertainty import get_uncertainty_result
from clio.flashnet.training.shared import FlashnetTrainResult

from clio.utils.cpu_usage import CPUUsage
from clio.utils.dataframe import append_to_df
from clio.utils.general import ratio_to_percentage_str, torch_set_seed
from clio.utils.indented_file import IndentedFile
from clio.utils.logging import LogLevel, log_global_setup
from clio.utils.path import rmdir
from clio.utils.timer import Timer, default_timer
from clio.utils.trace_pd import trace_get_dataset_paths


class DriftSurfState:
    def __init__(self, device, prediction_batch_size, threshold, use_eval_dropout, batch_size, base_model_dir, drop_rate = 0, delta=0.1, r=3, wl=2, epochs=1, learning_rate=0.0001):
        self.reac_len = r
        self.delta = delta
        self.win_len = wl
        self.models = {"pred": None, "stab": None, "reac": None}
        self.model_paths = {"pred": None, "stab": None, "reac": None}
        self.train_data_dict = {"pred": [0], "stab": [0], "reac": None}
        self.base_model_dir = base_model_dir
        
        # make this default dict, so auto initialize dict inside dict even if empty
        self.data_df_dict = defaultdict(dict)
        self.train_keys = ["pred", "stab"]
        self.acc_best = 0
        self.acc_dict = None
        self.reac_ctr = None
        self.state = "stab"
        self.model_key = "pred"  # Model used for prediction
        self.device = device
        self.prediction_batch_size = prediction_batch_size
        self.threshold = threshold
        self.use_eval_dropout = use_eval_dropout
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.drop_rate = drop_rate
        

    def _score(self, model_key, test_data):
        pred_result = flashnet_simple.flashnet_predict(                        
                        model=self.models[model_key],
                        dataset=test_data,
                        device=self.device,
                        batch_size=self.prediction_batch_size,
                        threshold=self.threshold,
                        use_eval_dropout=self.use_eval_dropout,
                        disable_tqdm=True
                    )
        eval_result = flashnet_evaluate(
            labels=pred_result.labels,
            predictions=pred_result.predictions,
            probabilities=pred_result.probabilities,
        )
        
        return eval_result.accuracy

    def eval(self, data):
        pred_result = flashnet_simple.flashnet_predict(
            model=self.models[self.model_key],
            dataset=data,
            device=self.device,
            batch_size=self.prediction_batch_size,
            threshold=self.threshold,
            use_eval_dropout=self.use_eval_dropout,
            disable_tqdm=True
        )
        
        return pred_result
    
    def initialize_model(self, key):
        new_model_id = suid.uuid()
        new_model_dir = self.base_model_dir / new_model_id
        new_model_dir.mkdir(parents=True, exist_ok=True)
        new_model_path = new_model_dir / "model.pt"
        
        self.model_paths[key] = new_model_path

    def train(self):
        if self.train_data_dict is None:
            raise ValueError("train_data_dict is None. Please set it first")

        num_train = 0
        for key in self.train_keys:
            if self.models[key] is None:
                self.initialize_model(key)
                
            for iter_id in self.train_data_dict[key]:
                print("Training model {} on iter {}".format(key, iter_id))
                retrain = self.models[key] is None
                flashnet_simple.flashnet_train(
                    model_path=self.model_paths[key],
                    dataset=self.data_df_dict[iter_id],
                    retrain=retrain,
                    batch_size=self.batch_size,
                    prediction_batch_size=self.prediction_batch_size,
                    lr=self.learning_rate,
                    epochs=self.epochs,
                    norm_mean=None,
                    norm_std=None,
                    n_data=None,
                    device=self.device,
                    drop_rate=self.drop_rate,
                    use_eval_dropout=self.use_eval_dropout,
                )
                model = flashnet_simple.load_model(self.model_paths[key], self.device)
                self.models[key] = model
                num_train += len(self.data_df_dict[iter_id])
        return num_train

    # IMPORTANT set Key DF for setting dataframes in drift surf
    def set_key_df(self, iter_id, data_df_dict):
        self.data_df_dict[iter_id] = data_df_dict

    def _append_train_data(self, model_key, iter_id):
        self.train_data_dict[model_key].append(iter_id)
        # Limit the number of data batches based on window length
        if len(self.train_data_dict[model_key]) > self.win_len:
            self.train_data_dict[model_key].pop(0)

    def _reset(self, key):
        self.models[key] = None
        self.model_paths[key] = None
        self.train_data_dict[key] = []


    def run_ds_algo(self, new_data, curr_iter):
        acc_pred = self._score("pred", new_data)
        print("DS Iteration {}, acc: {}".format(curr_iter, acc_pred))
        
        if acc_pred > self.acc_best:
            self.acc_best = acc_pred
            
        if self.state == "stab":
            if len(self.train_data_dict["stab"]) == 0:
                acc_stab = 0
            else:
                acc_stab = self._score("stab", new_data)
            
            # Enter reactive if current acc lower than best accuracy or stable accuracy
            if (acc_pred < self.acc_best - self.delta) or (acc_pred < acc_stab - self.delta / 2):
                # Enter reactive state
                print("Entering reactive state")
                self.state = "reac"
                self._reset("reac")
                self.reac_ctr = 0
                self.acc_dict = {"pred": np.zeros(self.reac_len), "reac": np.zeros(self.reac_len)}
            else:
                # Stay in stable state
                self._append_train_data("pred", curr_iter)
                self._append_train_data("stab", curr_iter)
                self.train_keys = ["pred", "stab"]

        if self.state == "reac":
            ## If reactive has been trained, 
            ## compare and see which model to use based on best acc
            if self.reac_ctr > 0:
                acc_reac = self._score("reac", new_data)
                print("acc_reac = {}".format(acc_reac))
                self.acc_dict["pred"][self.reac_ctr - 1] = acc_pred
                self.acc_dict["reac"][self.reac_ctr - 1] = acc_reac
                # Set key for next time step
                if acc_reac > acc_pred:
                    self.model_key = "reac"
                else:
                    self.model_key = "pred"
                    
            self._append_train_data("pred", curr_iter)
            self._append_train_data("reac", curr_iter)
            self.train_keys = ["pred", "reac"]
            self.reac_ctr += 1
            
            if self.reac_ctr == self.reac_len:
                # Exit Reactive State
                self.state = "stab"
                self._reset("stab")
                if np.mean(self.acc_dict["pred"]) < np.mean(self.acc_dict["reac"]):
                    self.models["pred"] = self.models["reac"]
                    self.model_paths["pred"] = self.model_paths["reac"]
                    self.train_data_dict["pred"] = self.train_data_dict["reac"]
                    self.acc_best = np.amax(self.acc_dict["reac"])
                    self.model_key = "pred"
                self.acc_dict = None
                self.reac_ctr = None
        # Debug
        print("DriftSurf State")
        print(self.state)
        print(self.train_data_dict)
        print(self.train_keys)
        print(self.acc_best)
        print(self.model_key)
        

@app.command()
def exp_driftsurf(
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

    log.info("Driftsurf prediction batch: %s", prediction_batch_size)

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

    ## This driftsurf uses our current flashnet models
    # To use this, call the predict(), run_ds_algo(), train() methods of drift surf.
    # Remember to set the data first for every iteration/window using set_key_df()
    driftsurf = DriftSurfState(
        device=device,
        prediction_batch_size=prediction_batch_size,
        threshold=threshold,
        use_eval_dropout=use_eval_dropout,
        batch_size=batch_size,
        base_model_dir=base_model_dir,
        delta=0.1,
        r=3,
        wl=2,
        epochs=epochs,
        learning_rate=learning_rate,
        drop_rate=drop_rate,
    )

    for i, data_path in enumerate(data_paths):
        log.info("Processing dataset: %s", data_path, tab=1)
        data = pd.read_csv(data_path)
        log.info("Length of data: %s", len(data), tab=2)
        driftsurf.set_key_df(i, data)
        if i == 0:
            log.info("Training", tab=1)

            train_cpu_usage = CPUUsage()
            train_cpu_usage.update()

            with Timer(name="Pipeline -- Initial Model Training -- Window %d" % i) as timer:
                driftsurf.train()
                train_result = driftsurf.eval(data)
                eval_result = flashnet_evaluate(labels=data["reject"], predictions=train_result.predictions, probabilities=train_result.probabilities)
                model_path = driftsurf.model_paths[driftsurf.model_key]
            
            train_result = FlashnetTrainResult(
                **eval_result.as_dict(),
                stats=eval_result.stats,
                train_time=timer.elapsed,
                prediction_time=timer.elapsed,
                model_path=model_path,
                norm_mean=None,
                norm_std=None,
                ip_threshold=None,
                confidence_threshold=None,
                labels=np.array(data["reject"]),
                predictions=train_result.predictions,
                probabilities=train_result.probabilities,
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
                prediction_result = driftsurf.eval(data)
                
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
            driftsurf.run_ds_algo(data, i)
            driftsurf.train()
            train_result = driftsurf.eval(data)
            eval_result = flashnet_evaluate(labels=data["reject"], predictions=train_result.predictions, probabilities=train_result.probabilities)
            model_path = driftsurf.model_paths[driftsurf.model_key]
            
        retrain_result = FlashnetTrainResult(
            **eval_result.as_dict(),
            stats=eval_result.stats,
            train_time=timer.elapsed,
            prediction_time=timer.elapsed,
            model_path=model_path,
            norm_mean=None,
            norm_std=None,
            ip_threshold=None,
            confidence_threshold=None,
            labels=np.array(data["reject"]),
            predictions=train_result.predictions,
            probabilities=train_result.probabilities,
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
