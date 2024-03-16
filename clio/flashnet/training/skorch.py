from pathlib import Path

import numpy as np
import pandas as pd

import torch

from skorch import NeuralNetClassifier
from skorch.helper import predefined_split

import clio.flashnet.ip_finder as ip_finder
from clio.flashnet.constants import FEATURE_COLUMNS
from clio.flashnet.eval import flashnet_evaluate
from clio.flashnet.training.shared import FlashnetTrainResult
from clio.flashnet.training.simple import FlashnetDataset, FlashnetModel, prepare_data

from clio.utils.logging import log_get
from clio.utils.timer import default_timer as timer

log = log_get(__name__)


def flashnet_predict(
    model: NeuralNetClassifier,
    dataset: pd.DataFrame,
    norm_mean: np.ndarray | None = None,
    norm_std: np.ndarray | None = None,
    batch_size: int | None = -1,  # if None, then prediction_batch_size = 32
    device: torch.device | None = None,
    threshold: float = 0.5,
):
    dataset = dataset.copy(deep=True)

    # drop io_type == 0
    if "io_type" in dataset.columns:
        dataset = dataset[dataset["io_type"] != 0]

    x = dataset.drop(columns=dataset.columns.difference(FEATURE_COLUMNS), axis=1).values
    y = dataset["reject"].values
    x = x.astype(np.float32)
    y = y.astype(np.int64)

    eval_dataset = FlashnetDataset(x, y, norm_mean=norm_mean, norm_std=norm_std)
    y_prob = model.predict_proba(eval_dataset)
    y_pred = (y_prob > threshold).astype(int)
    return y, y_pred, y_prob


class CustomBCELoss(torch.nn.BCELoss):
    def forward(self, input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target = target.float().reshape(-1, 1)
        return super().forward(input, target)


def load_model(model_path: str | Path) -> FlashnetModel:
    # return torch.load(model_path)
    classifier = NeuralNetClassifier(
        FlashnetModel,
        max_epochs=1,
        lr=0.0001,
        batch_size=32,
        device="cpu",
        criterion=CustomBCELoss,
        train_split=None,
        module__input_size=len(FEATURE_COLUMNS),
        module__hidden_layers=2,
        module__hidden_size=512,
        module__output_size=1,
    )
    classifier.initialize()
    classifier.load_params(f_params=model_path)
    return classifier.module_


def save_model(model: FlashnetModel, model_path: str | Path) -> None:
    torch.save(model, model_path)


def flashnet_train(
    model_path: str | Path,
    dataset: pd.DataFrame,
    retrain: bool = False,
    batch_size: int | None = 32,  # if None, then batch_size = 32
    prediction_batch_size: int | None = -1,  # if None, then prediction_batch_size = 32
    lr: float = 0.0001,
    epochs: int = 20,
    norm_mean: np.ndarray | None = None,
    norm_std: np.ndarray | None = None,
    n_data: int | None = None,
    device: torch.device | None = None,
    threshold: float = 0.5,
) -> FlashnetTrainResult:
    assert (norm_mean is None) == (norm_std is None)

    model_path = Path(model_path)

    if prediction_batch_size < 0:
        prediction_batch_size = batch_size

    ########################################
    # Preparing data
    ########################################

    train_dataset, val_dataset, norm_mean, norm_std = prepare_data(dataset, n_data=n_data, norm_mean=norm_mean, norm_std=norm_std)

    ########################################
    # Model
    ########################################

    classifier = NeuralNetClassifier(
        FlashnetModel,
        max_epochs=epochs,
        lr=lr,
        batch_size=batch_size,
        device=device,
        criterion=CustomBCELoss,
        train_split=predefined_split(val_dataset),
        iterator_train__shuffle=True,
        iterator_train__num_workers=1,
        iterator_train__pin_memory=True,
        module__input_size=len(FEATURE_COLUMNS),
        module__hidden_layers=2,
        module__hidden_size=512,
        module__output_size=1,
        classes=[0, 1],
    )

    classifier.initialize()
    if retrain:
        classifier.load_params(f_params=model_path)

    ########################################
    # Training
    ########################################

    start_time = timer()
    classifier.partial_fit(train_dataset, y=None)
    train_time = timer() - start_time

    ########################################
    # Saving model
    ########################################

    classifier.save_params(
        f_params=model_path,
    )

    ########################################
    # Results
    ########################################

    y_true, y_pred, y_probs = flashnet_predict(
        classifier, dataset, norm_mean=norm_mean, norm_std=norm_std, batch_size=prediction_batch_size, device=device, threshold=threshold
    )

    start_time = timer()
    eval_result = flashnet_evaluate(y_true, y_pred)
    prediction_time = timer() - start_time

    ip_latency_threshold, _ = ip_finder.area_based(dataset["latency"])  # y_pred is array of predicted latencies

    result = FlashnetTrainResult(
        **eval_result.as_dict(),
        stats=eval_result.stats,
        train_time=train_time,
        prediction_time=prediction_time,
        model_path=model_path,
        norm_mean=norm_mean,
        norm_std=norm_std,
        ip_threshold=ip_latency_threshold,
    )

    return result
