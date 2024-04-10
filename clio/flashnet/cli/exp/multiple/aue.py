import numpy as np

from clio.flashnet.preprocessing.add_filter import add_filter_v2
from clio.flashnet.training.shared import FlashnetTrainResult


## https://github.com/w4k2/stream-learn/tree/d3142a3b973e27141a0108f5fffabc7017222d31

np.float = float

from typing import Annotated
import typer
from clio.utils.logging import LogLevel, log_global_setup
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Annotated
from clio.utils.timer import default_timer as timer

import numpy as np
import pandas as pd

import torch

import shortuuid as suid
import typer

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


app = typer.Typer(name="Exp -- AUE")


from sklearn.base import ClassifierMixin, BaseEstimator
from sklearn.utils.validation import check_X_y, check_is_fitted, check_array
import numpy as np


class StreamingEnsemble(ClassifierMixin, BaseEstimator):
    """Abstract, base ensemble streaming class"""

    def __init__(self, base_estimator, n_estimators, weighted=False):
        self.base_estimator = base_estimator
        self.n_estimators = n_estimators
        self.weighted = weighted

    def fit(self, X, y):
        """Fitting."""
        self.partial_fit(X, y)
        return self

    def partial_fit(self, X, y, classes=None):
        """Partial fitting"""
        X, y = check_X_y(X, y)
        if not hasattr(self, "ensemble_"):
            self.ensemble_ = []

        self.green_light = True

        # Check feature consistency
        if hasattr(self, "X_"):
            if self.X_.shape[1] != X.shape[1]:
                raise ValueError("number of features does not match")
        self.X_, self.y_ = X, y

        # Check classes
        self.classes_ = classes
        if self.classes_ is None:
            self.classes_, _ = np.unique(y, return_inverse=True)

        # Check label consistency
        if len(np.unique(y)) != len(np.unique(self.classes_)):
            y[: len(np.unique(self.classes_))] = np.copy(self.classes_)

        # Check if it is possible to train new estimator
        if len(np.unique(y)) != len(self.classes_):
            self.green_light = False

        return self

    def ensemble_support_matrix(self, X):
        """Ensemble support matrix."""
        # print('ESM')
        return np.nan_to_num(np.array([member_clf.predict_proba(X) for member_clf in self.ensemble_]))

    def predict_proba(self, X):
        """Predict proba."""
        esm = self.ensemble_support_matrix(X)
        if self.weighted:
            esm *= np.array(self.weights_)[:, np.newaxis, np.newaxis]

        average_support = np.mean(esm, axis=0)
        return average_support

    def predict(self, X):
        """
        Predict classes for X.

        :type X: array-like, shape (n_samples, n_features)
        :param X: The training input samples.

        :rtype: array-like, shape (n_samples, )
        :returns: The predicted classes.
        """

        # Check is fit had been called
        check_is_fitted(self, "classes_")
        X = check_array(X)
        if X.shape[1] != self.X_.shape[1]:
            raise ValueError("number of features does not match")
        proba = self.predict_proba(X)
        prediction = np.argmax(proba, axis=1)
        
        # probabilities or return, need to convert into 1d array with probability of label 1
        proba = proba[:, 1]

        # Return prediction
        return self.classes_[prediction], proba

    def msei(self, clf, X, y):
        """MSEi score from original AWE algorithm."""
        pprobas = clf.predict_proba(X)
        probas = np.zeros(len(y))
        for label in self.classes_:
            probas[y == label] = pprobas[y == label, label]
        return np.sum(np.power(1 - probas, 2)) / len(y)
    
    def prior_proba(self, y):
        """Calculate prior probability for given labels"""
        return np.unique(y, return_counts=True)[1] / len(y)

    def mser(self, y):
        """MSEr score from original AWE algorithm."""
        prior_proba = self.prior_proba(y)
        return np.sum(prior_proba * np.power((1 - prior_proba), 2))

    def minority_majority_split(self, X, y, minority_name, majority_name):
        """Returns minority and majority data

        :type X: array-like, shape (n_samples, n_features)
        :param X: The training input samples.
        :type y: array-like, shape  (n_samples)
        :param y: The target values.

        :rtype: tuple (array-like, shape = [n_samples, n_features], array-like, shape = [n_samples, n_features])
        :returns: Tuple of minority and majority class samples
        """

        minority_ma = np.ma.masked_where(y == minority_name, y)
        minority = X[minority_ma.mask]

        majority_ma = np.ma.masked_where(y == majority_name, y)
        majority = X[majority_ma.mask]

        return minority, majority

    def minority_majority_name(self, y):
        """Returns minority and majority data

        :type y: array-like, shape  (n_samples)
        :param y: The target values.

        :rtype: tuple (object, object)
        :returns: Tuple of minority and majority class names.
        """

        unique, counts = np.unique(y, return_counts=True)

        if counts[0] > counts[1]:
            majority_name = unique[0]
            minority_name = unique[1]
        else:
            majority_name = unique[1]
            minority_name = unique[0]

        return minority_name, majority_name


from sklearn.base import clone
from sklearn.model_selection import KFold
import numpy as np

class AUE(StreamingEnsemble):
    """Accuracy Updated Ensemble"""

    def __init__(self, base_estimator=None, n_estimators=10, n_splits=5, epsilon=0.00001):
        """Initialization."""
        super().__init__(base_estimator, n_estimators)
        self.n_splits = n_splits
        self.epsilon = epsilon

    def partial_fit(self, X, y, classes=None):
        y = np.array(y)
        X = np.array(X)
        super().partial_fit(X, y, classes)
        if not self.green_light:
            print("Green light not true! Cannot train new classifier")
            return self

        # Compute baseline
        mser = self.mser(y)

        # Train new estimator
        candidate = clone(self.base_estimator).fit(self.X_, self.y_)

        # Calculate its scores
        scores = []
        kf = KFold(n_splits=self.n_splits)
        for fold, (train, test) in enumerate(kf.split(X)):
            if len(np.unique(y[train])) != len(self.classes_):
                continue
            fold_candidate = clone(self.base_estimator).fit(self.X_[train], self.y_[train])
            msei = self.msei(fold_candidate, self.X_[test], self.y_[test])
            scores.append(msei)

        # Save scores
        candidate_msei = np.mean(scores)
        candidate_weight = 1 / (candidate_msei + self.epsilon)

        # Calculate weights of current ensemble
        self.weights_ = [1 / (self.msei(clf, self.X_, self.y_) + self.epsilon) for clf in self.ensemble_]

        # Add new model
        self.ensemble_.append(candidate)
        self.weights_.append(candidate_weight)

        # Remove the worst when ensemble becomes too large
        if len(self.ensemble_) > self.n_estimators:
            worst_idx = np.argmin(self.weights_)
            del self.ensemble_[worst_idx]
            del self.weights_[worst_idx]

        # AUE update procedure
        comparator = 1 / mser
        counter = 0
        for i, clf in enumerate(self.ensemble_):
            if i == len(self.ensemble_) - 1:
                break
            ## If current weight is > comparator, update by refitting
            if self.weights_[i] > comparator:
                counter += 1
                clf.partial_fit(X, y)
        print("Model has {} classifiers, with weights {}".format(len(self.ensemble_), str(self.weights_)))
        print("Retrained {} classifiers, comparator: {}".format(counter, comparator))
        return self
    


@app.command()
def exp_aue(
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

    ## AUE with scikit_learn's base estimator. Need to be able to fit(), partial_fit(), and predict_proba().
    ## No model saving is done
    
    from sklearn.linear_model import SGDClassifier
    aue = AUE(base_estimator=SGDClassifier(loss='log_loss'))

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
                # We filter with timer to follow flashnet convention
                data_train = data.copy()
                data_train = add_filter_v2(data_train)
                aue.fit(data_train[FEATURE_COLUMNS], data_train["reject"])
            with Timer(name="Pipeline -- Initial Model Prediction -- Window %d" % i) as timer_infer_init:
                temp_label, temp_proba = aue.predict(data[FEATURE_COLUMNS])
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
                predictions, probabilities = aue.predict(data[FEATURE_COLUMNS])
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
            # We filter with timer to follow flashnet convention
            data_train = data.copy()
            data_train = add_filter_v2(data_train)
            aue.fit(data_train[FEATURE_COLUMNS], data_train["reject"])

        ## TODO: This is nonsense, just to satisfy the pipeline.
        ## Dont quite know what to return on aue retrain
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

app = typer.Typer(name="Exp -- Multiple -- Admit -- All Window Data")


@dataclass(kw_only=True)
class AUEModelGroup:
    _models: list[str | Path] = field(default_factory=list)
    _weights: list[float] = field(default_factory=list)
    _model_paths: list[Path] = field(default_factory=list)
    
    def __init__(self, device, batch_size, prediction_batch_size, learning_rate, epochs, drop_rate, use_eval_dropout, threshold, epsilon = 0.00001, n_estimator=10):
        self._models = []
        self._weights = []
        self._model_paths = []
        self.epsilon = epsilon
        self.n_estimators = n_estimator
        self.device = device
        self.batch_size = batch_size
        self.prediction_batch_size = prediction_batch_size
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.drop_rate = drop_rate
        self.use_eval_dropout = use_eval_dropout
        self.threshold = threshold
        
        
    def prior_proba(self, y):
        """Calculate prior probability for given labels"""
        return np.unique(y, return_counts=True)[1] / len(y)

    def mser(self, y):
        """MSEr score from original AWE algorithm."""
        prior_proba = self.prior_proba(y)
        return np.sum(prior_proba * np.power((1 - prior_proba), 2))


    def add_model(self, model_path, data):
        y = data["reject"]
        model = flashnet_simple.load_model(model_path, device=self.device)
        self._models.append(model)
        self._model_paths.append(model_path)
        ## Rerank models
        self._weights = [1 / (self.get_msei(clf, data) + self.epsilon) for clf in self._models]
        base_error = self.mser(y)
        
        # Remove the worst when ensemble becomes too large
        if len(self._models) > self.n_estimators:
            worst_idx = np.argmin(self._weights)
            del self._models[worst_idx]
            del self._weights[worst_idx]
            del self._model_paths[worst_idx]
        
        ## AUE update procedure
        comparator = 1 / base_error
        counter = 0
        for i, (clf_path) in enumerate(self._model_paths):
            if i == len(self._models) - 1:
                break
            ## If current weight is > comparator, update by refitting
            if self._weights[i] > comparator:
                counter += 1
                flashnet_simple.flashnet_train(
                    model_path=clf_path,
                    dataset=data,
                    retrain=True,
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
        
        print("Model has {} classifiers, with weights {}".format(len(self._models), str(self._weights)))
        print("Retrained {} classifiers, comparator: {}".format(counter, comparator))
    
        
    def get_msei(self, model, dataset):
        """MSEi score from original AWE algorithm."""
        
        X = dataset[FEATURE_COLUMNS]
        y = dataset["reject"]
        
        pred_result = flashnet_simple.flashnet_predict(                        
                        model=model,
                        dataset=dataset,
                        device=self.device,
                        batch_size=self.prediction_batch_size,
                        threshold=self.threshold,
                        use_eval_dropout=self.use_eval_dropout,
                        disable_tqdm=True
                    )
                
        pprobas = pred_result.probabilities
        # probas = np.zeros(len(y))
        # for label in [0,1]:
        #     probas[y == label] = pprobas[y == label, label]
        return np.sum(np.power(1 - pprobas, 2)) / len(y)

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
def exp_aue_adapted(
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
    model_group = AUEModelGroup(device = torch.device(f"cuda:{cuda}" if torch.cuda.is_available() and cuda >= 0 else "cpu"),
                                batch_size = batch_size,
                                prediction_batch_size = prediction_batch_size,
                                learning_rate = learning_rate,
                                epochs = epochs,
                                drop_rate = drop_rate,
                                use_eval_dropout = use_eval_dropout,
                                threshold = threshold)

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
            model_group.add_model(model_path, data)

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
                    models=model_group._model_paths,
                    dataset=data,
                    device=device,
                    batch_size=prediction_batch_size,
                    threshold=threshold,
                    use_eval_dropout=use_eval_dropout,
                    weights = model_group._weights
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
        log.info("Elapsed time: %s", timer.elapsed, tab=2)
        log.info("CPU Usage: %s", retrain_cpu_usage.result, tab=2)
        log.info("AUC: %s", retrain_result.auc, tab=2)

        assert len(data) == retrain_result.num_io, "sanity check, number of data should be the same as the number of input/output"

        model_path = retrain_result.model_path
        # model = flashnet_simple.load_model(model_path, device=device)
        model_group.add_model(model_path, data)

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
