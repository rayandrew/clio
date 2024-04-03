import copy
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader

import clio.flashnet.ip_finder as ip_finder
from clio.flashnet.constants import FEATURE_COLUMNS, LAYERS
from clio.flashnet.eval import flashnet_evaluate
from clio.flashnet.preprocessing.add_filter import add_filter_v2
from clio.flashnet.training.simple import FlashnetDataset, FlashnetModel, FlashnetTrainResult, PredictionResult, create_model
from clio.flashnet.training.simple import load_model as base_load_model
from clio.flashnet.training.simple import prepare_data
from clio.flashnet.training.simple import save_model as base_save_model

from clio.utils.general import enable_dropout
from clio.utils.logging import log_get
from clio.utils.timer import default_timer as timer
from clio.utils.tqdm import tqdm

log = log_get(__name__)


@dataclass(kw_only=True)
class FlashnetEnsembleTrainResult(FlashnetTrainResult):
    models: list[Path] = field(default_factory=list)


def flashnet_ensemble_train(
    model_path: str | Path,
    num_models: int,
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
    confidence_threshold: float = 0.1,
    layers: list[int] = LAYERS,
    drop_rate: float = 0.0,
    use_eval_dropout: bool = False,
) -> FlashnetEnsembleTrainResult:
    assert (norm_mean is None) == (norm_std is None)

    if prediction_batch_size < 0:
        prediction_batch_size = batch_size

    ########################################
    # Preparing data
    ########################################

    train_loaders: list[DataLoader] = []
    val_loaders: list[DataLoader] = []

    for i in range(num_models):
        _dataset = dataset.copy(deep=True)
        _dataset = add_filter_v2(_dataset)
        train_dataset, val_dataset = prepare_data(_dataset, n_data=n_data)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, persistent_workers=True, num_workers=1)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, persistent_workers=True, num_workers=1)
        train_loaders.append(train_loader)
        val_loaders.append(val_loader)

    # train_dataset, val_dataset = prepare_data(dataset, n_data=n_data)
    # train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, persistent_workers=True, num_workers=1)
    # val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, persistent_workers=True, num_workers=1)

    ########################################
    # Training
    ########################################

    if retrain and model_path.exists():
        models = load_model([model_path / f"model_{i}" for i in range(num_models)], device=device)
    else:
        models: list[FlashnetModel] = []
        for i in range(num_models):
            model = create_model(layers=layers, drop_rate=drop_rate)
            if norm_mean is not None and norm_std is not None:
                model.set_normalizer(norm_mean, norm_std)
            else:
                model.adapt(torch.tensor(train_dataset.x, dtype=torch.float32))

            models.append(model)

        norm_mean = models[-1].norm_mean
        norm_std = models[-1].norm_std

    criterion = torch.nn.BCELoss()
    optimizers = [torch.optim.Adam(model.parameters(), lr=lr) for model in models]

    start_time = timer()

    for i, (model, optimizer) in enumerate(zip(models, optimizers)):
        train_loader = train_loaders[i]
        val_loader = val_loaders[i]
        for epoch in range(epochs):
            if device is not None:
                model.to(device)

            model.train()
            for x, y in tqdm(train_loader, desc=f"Model {i} Epoch {epoch + 1}/{epochs}", unit="batch", leave=False, dynamic_ncols=True):
                if device is not None:
                    x = x.to(device)
                    y = y.to(device)
                optimizer.zero_grad()
                y_pred = model(x.float())
                loss = criterion(y_pred, y.float().view(-1, 1))
                loss.backward()
                optimizer.step()

            model.eval()
            if use_eval_dropout:
                enable_dropout(model)
            with torch.no_grad():
                val_loss = 0
                labels = []
                predictions = []
                probabilities = []
                for x, y in val_loader:
                    if device is not None:
                        x = x.to(device)
                        y = y.to(device)
                    y_pred = model(x.float())
                    val_loss += criterion(y_pred, y.float().view(-1, 1)).item()
                    y_pred_label = y_pred > threshold
                    probabilities.extend(y_pred.cpu().numpy())
                    labels.extend(y.cpu().numpy())
                    predictions.extend(y_pred_label.cpu().numpy())

                val_loss /= len(val_loader)
                val_loss = round(val_loss, 4)
                labels = np.array(labels)
                predictions = np.array(predictions)
                val_result = flashnet_evaluate(labels=labels, predictions=predictions, probabilities=probabilities)

            log.info(
                "Model %d, Epoch %d/%d, Loss: %.4f, Val Loss: %.4f, Val Acc: %.4f, Val AUC: %.4f",
                i,
                epoch + 1,
                epochs,
                loss.item(),
                val_loss,
                val_result.accuracy,
                val_result.auc,
                tab=1,
            )

            model.to("cpu")

    train_time = timer() - start_time

    save_model(models, model_path, norm_mean=norm_mean, norm_std=norm_std)

    result = flashnet_ensemble_predict(
        models=models,
        dataset=dataset,
        batch_size=prediction_batch_size,
        device=device,
        threshold=threshold,
    )
    start_time = timer()
    eval_result = flashnet_evaluate(labels=result.labels, predictions=result.predictions, probabilities=result.probabilities)
    prediction_time = timer() - start_time

    ip_latency_threshold, _ = ip_finder.area_based(dataset["latency"])  # y_pred is array of predicted latencies

    result = FlashnetEnsembleTrainResult(
        **eval_result.as_dict(),
        stats=eval_result.stats,
        train_time=train_time,
        prediction_time=prediction_time,
        model_path=model_path,
        norm_mean=norm_mean,
        norm_std=norm_std,
        ip_threshold=ip_latency_threshold,
        confidence_threshold=confidence_threshold,
        labels=result.labels,
        predictions=result.predictions,
        probabilities=result.probabilities,
        models=[model_path / f"model_{i}" for i in range(num_models)],
    )

    return result


def _ensemble_predict(
    models: list[torch.nn.Module | torch.ScriptModule],
    dataset: pd.DataFrame,
    batch_size: int,
    threshold: float,
    device: torch.device | None = None,
    use_eval_dropout: bool = False,
) -> PredictionResult:
    assert len(models) > 0, "At least one model is required"

    x = dataset.drop(columns=dataset.columns.difference(FEATURE_COLUMNS), axis=1).values
    y = dataset["reject"].values

    eval_dataset = FlashnetDataset(x, y)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, persistent_workers=True, num_workers=1)

    labels = np.zeros(len(dataset), dtype=np.int64)
    probabilities = np.zeros((len(models), len(dataset)), dtype=np.float32)

    # for model in models:
    #     model.eval()
    #     if device is not None:
    #         model.to(device)
    #     if use_eval_dropout:
    #         enable_dropout(model)

    # params, buffers = stack_module_state(models)

    # base_model = copy.deepcopy(models[0])
    # base_model = base_model.to("meta")

    # def fmodel(x: torch.Tensor) -> torch.Tensor:
    #     return functional_call(base_model, (params, buffers), (x.float(),))

    # with torch.no_grad():
    #     last_count = 0
    #     # log.info("1")
    #     # probs = torch.vmap(fmodel)(params, buffers, eval_loader)
    #     # log.info("2")
    #     for x, y in eval_loader:
    #         if device is not None:
    #             x = x.to(device)
    #             y = y.to(device)
    #         probs = torch.vmap(fmodel)(params, buffers, x)
    #         n_data = len(y)
    #         labels[last_count : last_count + n_data] = y.cpu()
    #         probabilities[:, last_count : last_count + n_data] = probs.cpu()
    #         last_count += n_data

    for i, model in enumerate(models):
        if device is not None:
            model.to(device)
        model.eval()
        if use_eval_dropout:
            enable_dropout(model)
        with torch.no_grad():
            last_count = 0
            for x, y in tqdm(eval_loader, desc=f"Model {i}", unit="batch", leave=False, dynamic_ncols=True):
                if device is not None:
                    x = x.to(device)
                    y = y.to(device)
                n_data = len(y)
                if i == 0:
                    labels[last_count : last_count + n_data] = y.cpu()
                probs = model(x.float()).reshape(-1)
                probabilities[i, last_count : last_count + n_data] = probs.cpu()
                last_count += n_data
        model.to("cpu")

    probabilities = np.mean(probabilities, axis=0)
    predictions = (probabilities > threshold).astype(int)

    # calculate accuracy
    correct = (predictions == labels).sum()
    accuracy = correct / len(labels)
    log.info("Accuracy: %.4f", accuracy)

    return PredictionResult(
        labels=labels,
        predictions=predictions,
        probabilities=probabilities,
    )


def flashnet_ensemble_predict(
    models: list[torch.nn.Module | torch.ScriptModule],
    dataset: pd.DataFrame,
    batch_size: int | None = -1,  # if None, then prediction_batch_size = 32
    device: torch.device | None = None,
    threshold: float = 0.5,
    use_eval_dropout: bool = False,
):
    if batch_size < 0:
        batch_size = 32

    dataset = dataset.copy(deep=True)

    if "io_type" not in dataset.columns:
        # assume that all data are read data
        return _ensemble_predict(
            models=models,
            dataset=dataset,
            batch_size=batch_size,
            device=device,
            threshold=threshold,
            use_eval_dropout=use_eval_dropout,
        )

    # NOTE: debug only
    # dataset = dataset[dataset["io_type"] != 0]

    # check if there are write data
    write_indexes = dataset.index[dataset["io_type"] == 0].tolist()
    if len(write_indexes) > 0:
        log.info("There are write data in the dataset")
        # dataset is not only read data
        # we need to PREDICT only the read data
        original_len = len(dataset)
        read_indexes = dataset.index[dataset["io_type"] == 1].tolist()
        readonly_dataset = dataset.iloc[read_indexes]
        result = _ensemble_predict(
            models=models,
            dataset=readonly_dataset,
            batch_size=batch_size,
            device=device,
            threshold=threshold,
            use_eval_dropout=use_eval_dropout,
        )

        # reconstruct the result
        labels = np.zeros(original_len, dtype=np.int64)  # set to 0 because we are sure that write data will be accepted (reject=0)
        predictions = np.zeros(original_len, dtype=np.int64)  # set to 0 because we are sure that write data will be accepted (reject=0)
        probabilities = np.ones(original_len, dtype=np.float32)  # set to 1 because we are sure that write data will be accepted (reject=0)
        labels[read_indexes] = result.labels
        predictions[read_indexes] = result.predictions
        probabilities[read_indexes] = result.probabilities
        return PredictionResult(
            labels=labels,
            predictions=predictions,
            probabilities=probabilities,
        )
    else:
        # dataset is not only read data
        return _ensemble_predict(
            models=models,
            dataset=dataset,
            batch_size=batch_size,
            device=device,
            threshold=threshold,
            use_eval_dropout=use_eval_dropout,
        )


def load_model(path: list[str | Path], device: torch.device | None = None):
    models: list[torch.nn.Module] = []
    for p in path:
        p = Path(p)
        model = base_load_model(p, device=device)
        models.append(model)
    return models


def save_model(models: list[torch.nn.Module], path: Path, norm_mean: np.ndarray | None = None, norm_std: np.ndarray | None = None):
    for i, model in enumerate(models):
        base_save_model(model, path / f"model_{i}", norm_mean=norm_mean, norm_std=norm_std)


def flashnet_ensemble_predict_p(
    models: list[str | Path],
    dataset: pd.DataFrame,
    batch_size: int | None = -1,  # if None, then prediction_batch_size = 32
    device: torch.device | None = None,
    threshold: float = 0.5,
    use_eval_dropout: bool = False,
):
    models: list[torch.nn.Module] = load_model(models, device=device)
    return flashnet_ensemble_predict(models, dataset, batch_size=batch_size, device=device, threshold=threshold, use_eval_dropout=use_eval_dropout)


__all__ = ["flashnet_ensemble_train", "flashnet_ensemble_predict", "flashnet_ensemble_predict_p"]
