from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, Dataset

import clio.flashnet.ip_finder as ip_finder
from clio.flashnet.constants import FEATURE_COLUMNS, HIDDEN_LAYERS, HIDDEN_SIZE
from clio.flashnet.eval import flashnet_evaluate
from clio.flashnet.normalization import norm_to_str, parse_norm
from clio.flashnet.training.shared import FlashnetTrainResult

from clio.layers.initialization import init_weights
from clio.layers.normalizer import NormalizerMixin
from clio.utils.logging import log_get
from clio.utils.timer import default_timer as timer

log = log_get(__name__)


class FlashnetDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        # , norm_mean: np.ndarray | None = None, norm_std: np.ndarray | None = None):
        assert len(x) == len(y)
        # assert (norm_mean is None) == (norm_std is None)

        self.x = x
        self.y = y

        # if norm_mean is not None and norm_std is not None:
        #     self._norm_mean = norm_mean
        #     self._norm_std = norm_std
        # else:
        #     # calculate using torch
        #     norm_std, norm_mean = torch.std_mean(torch.tensor(self.x, dtype=torch.float32), dim=0)
        #     self._norm_mean = norm_mean.numpy()
        #     self._norm_std = norm_std.numpy()
        #     self._norm_std = np.maximum(self._norm_std, 1e-7)

    def __len__(self):
        return len(self.x)

    # @property
    # def norm_mean(self) -> np.ndarray:
    #     return self._norm_mean

    # @property
    # def norm_std(self) -> np.ndarray:
    #     return self._norm_std

    # def normalize(self, x: np.ndarray) -> np.ndarray:
    #     if self.norm_mean is None or self.norm_std is None:
    #         return x
    #     return (x - self.norm_mean) / self.norm_std

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        # x = self.normalize(x)
        return x, y


class FlashnetModel(torch.nn.Module, NormalizerMixin):
    def __init__(self, input_size: int, hidden_layers: int, hidden_size: int, output_size: int, drop_rate: float = 0.0):
        torch.nn.Module.__init__(self=self)
        NormalizerMixin.__init__(self=self, input_size=input_size)

        layers = torch.nn.Sequential(
            *(
                torch.nn.Linear(input_size, hidden_size),
                torch.nn.ReLU(inplace=True),
                torch.nn.Dropout(p=drop_rate),
            )
        )

        for layer_idx in range(hidden_layers - 1):
            layers.add_module(
                f"fc{layer_idx + 1}",
                torch.nn.Sequential(
                    *(
                        torch.nn.Linear(hidden_size, hidden_size),
                        torch.nn.ReLU(inplace=True),
                        torch.nn.Dropout(p=drop_rate),
                    )
                ),
            )

        self.features = torch.nn.Sequential(*layers)
        if output_size == 1:
            self.classifier = torch.nn.Sequential(torch.nn.Linear(hidden_size, output_size), torch.nn.Sigmoid())
        else:
            self.classifier = torch.nn.Linear(hidden_size, output_size)

        # initialize weights like Keras (using Xavier)
        init_weights(self)

    def set_normalizer(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.set(mean, std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.normalize(x)
        x = self.features(x)
        x = self.classifier(x)
        return x

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.normalize(x)
        return self.features(x)


@dataclass(kw_only=True, frozen=True)
class PredictionResult:
    labels: np.ndarray
    predictions: np.ndarray
    probabilities: np.ndarray


def _predict(
    model: torch.nn.Module,
    dataset: pd.DataFrame,
    batch_size: int,
    device: torch.device,
    threshold: float,
) -> PredictionResult:
    x = dataset.drop(columns=dataset.columns.difference(FEATURE_COLUMNS), axis=1).values
    y = dataset["reject"].values

    eval_dataset = FlashnetDataset(x, y)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, persistent_workers=True, num_workers=1)

    labels = np.zeros(len(dataset), dtype=np.int64)
    predictions = np.zeros(len(dataset), dtype=np.int64)
    probabilities = np.zeros(len(dataset), dtype=np.float32)
    last_count = 0

    model.eval()
    with torch.no_grad():
        for x, y in eval_loader:
            if device is not None:
                x = x.to(device)
                y = y.to(device)
            n_data = len(y)
            labels[last_count : last_count + n_data] = y.cpu()
            probs = model(x.float()).reshape(-1)
            y_pred = (probs > threshold).int()
            # log.info("y_pred: %s", y_pred)
            predictions[last_count : last_count + n_data] = y_pred.cpu()
            probabilities[last_count : last_count + n_data] = probs.cpu()
            last_count += n_data

    return PredictionResult(
        labels=labels,
        predictions=predictions,
        probabilities=probabilities,
    )


def flashnet_predict(
    model: torch.nn.Module,
    dataset: pd.DataFrame,
    batch_size: int | None = -1,  # if None, then prediction_batch_size = 32
    device: torch.device | None = None,
    threshold: float = 0.5,
) -> PredictionResult:
    # if norm_mean is None or norm_std is None:
    #     log.warning("BE CAREFUL, norm_mean and norm_std are not provided! The model may not work properly.")

    if batch_size < 0:
        batch_size = 32

    dataset = dataset.copy(deep=True)

    if "io_type" not in dataset.columns:
        # assume that all data are read data
        return _predict(
            model,
            dataset,
            batch_size,
            device,
            threshold,
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
        result = _predict(
            model,
            readonly_dataset,
            batch_size,
            device,
            threshold,
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
        return _predict(
            model,
            dataset,
            # norm_mean, norm_std,
            batch_size,
            device,
            threshold,
        )


def prepare_data(
    dataset: pd.DataFrame,
    n_data: int | None = None,
    test_size: float = 0.2,
    seed: int | None = None,
):
    ########################################
    # Preparing data
    ########################################

    dataset = dataset.copy(deep=True)

    # Drop unnecessary columns
    start = timer()
    useless_columns = [
        "ts_record",
        "ts_submit",
        "size_after_replay",
        "offset",
        "io_type",
        "rw_ratio",
        "avg_iat",
        "avg_iops",
        "norm_ts_record",
        "original_ts_record",
    ]
    dataset = dataset.drop(columns=useless_columns, errors="ignore", axis=1)
    log.debug("Time to drop unnecessary columns: %f", timer() - start)

    # drop io_type == 0
    if "io_type" in dataset.columns:
        dataset = dataset[dataset["io_type"] != 0]

    if n_data is not None:
        dataset = dataset.tail(n_data)
        # dataset = dataset.sample(n=n_data)

    x_train = dataset.drop(columns=dataset.columns.difference(FEATURE_COLUMNS), axis=1)
    y_train = dataset["reject"]
    n_data = len(x_train)

    log.info("Training data size: %d", n_data)

    ########################################
    # Splitting data
    ########################################

    indexes = np.arange(n_data)
    x_train_indexes, x_val_indexes, final_y_train, final_y_val = train_test_split(indexes, y_train.values, test_size=test_size, random_state=seed)

    final_x_train = x_train.values[x_train_indexes].astype(np.float32)
    final_x_val = x_train.values[x_val_indexes].astype(np.float32)
    final_y_train = final_y_train.astype(np.int64)
    final_y_val = final_y_val.astype(np.int64)

    # Show sample of x_train
    # log.info("Sample of x_train: %s", final_x_train[:5], tab=1)

    train_dataset = FlashnetDataset(final_x_train, final_y_train)
    val_dataset = FlashnetDataset(final_x_val, final_y_val)

    log.info("Training the model with the following columns: %s", list(x_train.columns))

    return train_dataset, val_dataset


def create_model(hidden_layers: int = HIDDEN_LAYERS, hidden_size: int = HIDDEN_SIZE, drop_rate: float = 0.0) -> FlashnetModel:
    log.info("Creating model with %d hidden layers, hidden size %d, and drop rate %f", hidden_layers, hidden_size, drop_rate)
    model = FlashnetModel(
        input_size=len(FEATURE_COLUMNS),
        hidden_layers=hidden_layers,
        hidden_size=hidden_size,
        drop_rate=drop_rate,
        output_size=1,
    )
    return model


def load_model(path: str | Path, device: torch.device | None = None) -> torch.jit.ScriptModule:
    model = torch.jit.load(path, map_location=device)
    return model


def load_model_with_norm(path: str | Path, device: torch.device | None = None) -> tuple[torch.jit.ScriptModule, np.ndarray, np.ndarray]:
    extra_files = {
        "norm_mean.txt": "",
        "norm_std.txt": "",
    }
    model = torch.jit.load(path, map_location=device, _extra_files=extra_files)
    norm_mean = parse_norm(extra_files["norm_mean.txt"])
    norm_std = parse_norm(extra_files["norm_std.txt"])
    if len(norm_mean) == 0 or len(norm_std) == 0:
        raise ValueError("norm_mean is empty or norm_std is empty")
    return model, norm_mean, norm_std


def save_model(model: torch.nn.Module, path: str | Path, norm_mean: np.ndarray | None = None, norm_std: np.ndarray | None = None):
    extra_files = {}
    if norm_mean is not None and norm_std is not None:
        extra_files["norm_mean.txt"] = norm_to_str(norm_mean)
        extra_files["norm_std.txt"] = norm_to_str(norm_std)

    torch.jit.save(torch.jit.script(model), path, _extra_files=extra_files)


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
    confidence_threshold: float = 0.1,
    hidden_layers: int = HIDDEN_LAYERS,
    hidden_size: int = HIDDEN_SIZE,
) -> FlashnetTrainResult:
    assert (norm_mean is None) == (norm_std is None)

    model_path = Path(model_path)

    if prediction_batch_size < 0:
        prediction_batch_size = batch_size

    ########################################
    # Preparing data
    ########################################

    train_dataset, val_dataset = prepare_data(dataset, n_data=n_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, persistent_workers=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, persistent_workers=True, num_workers=1)

    ########################################
    # Training
    ########################################

    if retrain and model_path.exists():
        model = cast(FlashnetModel, load_model(model_path))
    else:
        model = create_model(hidden_layers=hidden_layers, hidden_size=hidden_size)
        if norm_mean is not None and norm_std is not None:
            model.set_normalizer(norm_mean, norm_std)
        else:
            model.adapt(torch.tensor(train_dataset.x, dtype=torch.float32))
        norm_mean = model.norm_mean
        norm_std = model.norm_std

    if device:
        model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.BCELoss()

    start_time = timer()
    for epoch in range(epochs):
        model.train()
        for i, (x, y) in enumerate(train_loader):
            if device is not None:
                x = x.to(device)
                y = y.to(device)
            optimizer.zero_grad()
            y_pred = model(x.float())
            loss = criterion(y_pred, y.float().view(-1, 1))
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = 0
            labels = []
            predictions = []
            for x, y in val_loader:
                if device is not None:
                    x = x.to(device)
                    y = y.to(device)
                y_pred = model(x.float())
                val_loss += criterion(y_pred, y.float().view(-1, 1)).item()
                y_pred_label = y_pred > threshold
                labels.extend(y.cpu().numpy())
                predictions.extend(y_pred_label.cpu().numpy())

            val_loss /= len(val_loader)
            val_loss = round(val_loss, 4)
            labels = np.array(labels)
            predictions = np.array(predictions)
            val_result = flashnet_evaluate(labels, predictions)

        log.info(
            "Epoch %d/%d, Train Loss: %f, Val Loss: %f, Val Acc: %f, Val AUC: %f",
            epoch + 1,
            epochs,
            loss.item(),
            val_loss,
            val_result.accuracy,
            val_result.auc,
            tab=1,
        )

    train_time = timer() - start_time

    save_model(model, path=model_path, norm_mean=norm_mean, norm_std=norm_std)

    result = flashnet_predict(
        model,
        dataset,
        # norm_mean=norm_mean, norm_std=norm_std,
        batch_size=prediction_batch_size,
        device=device,
        threshold=threshold,
    )
    start_time = timer()
    eval_result = flashnet_evaluate(result.labels, result.predictions)
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
        confidence_threshold=confidence_threshold,
        labels=result.labels,
        predictions=result.predictions,
        probabilities=result.probabilities,
    )

    return result


__all__ = [
    "FlashnetDataset",
    "FlashnetModel",
    "PredictionResult",
    "flashnet_predict",
    "prepare_data",
    "load_model",
    "load_model_with_norm",
    "save_model",
    "flashnet_train",
    "create_model",
]
