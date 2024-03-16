from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, Dataset

import clio.flashnet.ip_finder as ip_finder
from clio.flashnet.constants import FEATURE_COLUMNS
from clio.flashnet.eval import flashnet_evaluate, get_confidence_data
from clio.flashnet.training.shared import FlashnetTrainResult

from clio.layers.initialization import init_weights
from clio.utils.logging import log_get
from clio.utils.timer import default_timer as timer

# from clio.layers.normalizer import NormalizerMixin

log = log_get(__name__)


class FlashnetDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray, norm_mean: np.ndarray | None = None, norm_std: np.ndarray | None = None):
        assert len(x) == len(y)
        assert (norm_mean is None) == (norm_std is None)

        self.x = x
        self.y = y

        if norm_mean is not None and norm_std is not None:
            self._norm_mean = norm_mean
            self._norm_std = norm_std
        else:
            # calculate using torch
            norm_std, norm_mean = torch.std_mean(torch.tensor(self.x, dtype=torch.float32), dim=0)
            self._norm_mean = norm_mean.numpy()
            self._norm_std = norm_std.numpy()
            self._norm_std = np.maximum(self._norm_std, 1e-7)

    def __len__(self):
        return len(self.x)

    @property
    def norm_mean(self) -> np.ndarray:
        return self._norm_mean

    @property
    def norm_std(self) -> np.ndarray:
        return self._norm_std

    def normalize(self, x: np.ndarray) -> np.ndarray:
        if self.norm_mean is None or self.norm_std is None:
            return x
        return (x - self.norm_mean) / self.norm_std

    def __getitem__(self, idx):
        x = self.x[idx]
        y = self.y[idx]
        x = self.normalize(x)
        return x, y


class FlashnetModel(torch.nn.Module):
    def __init__(self, input_size: int, hidden_layers: int, hidden_size: int, output_size: int):
        torch.nn.Module.__init__(self=self)
        # NormalizerMixin.__init__(self=self)

        layers = torch.nn.Sequential(
            *(
                torch.nn.Linear(input_size, hidden_size),
                torch.nn.ReLU(inplace=True),
                # torch.nn.Dropout(p=drop_rate),
            )
        )

        for layer_idx in range(hidden_layers - 1):
            layers.add_module(
                f"fc{layer_idx + 1}",
                torch.nn.Sequential(
                    *(
                        torch.nn.Linear(hidden_size, hidden_size),
                        torch.nn.ReLU(inplace=True),
                        # torch.nn.Dropout(p=drop_rate),
                    )
                ),
            )

        self.features = torch.nn.Sequential(*layers)
        if output_size == 1:
            self.classifier = torch.nn.Sequential(torch.nn.Linear(hidden_size, output_size), torch.nn.Sigmoid())
        else:
            self.classifier = torch.nn.Linear(hidden_size, output_size)

        init_weights(self)  # initialize weights like Keras (using Xavier)

    # def set_normalizer(self, mean: np.ndarray, std: np.ndarray) -> None:
    #     self.set(mean, std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        # x = self.normalize(x)
        x = self.features(x)
        x = self.classifier(x)
        return x

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        x = self.normalize(x)
        return self.features(x)


def _predict(
    model: torch.nn.Module, dataset: pd.DataFrame, norm_mean: np.ndarray, norm_std: np.ndarray, batch_size: int, device: torch.device, threshold: float
):
    x = dataset.drop(columns=dataset.columns.difference(FEATURE_COLUMNS), axis=1).values
    y = dataset["reject"].values

    eval_dataset = FlashnetDataset(x, y, norm_mean=norm_mean, norm_std=norm_std)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, persistent_workers=True, num_workers=1)

    model.eval()

    y_trues = []
    y_preds = []
    y_probs = []
    with torch.no_grad():
        for x, y in eval_loader:
            if device is not None:
                x = x.to(device)
                y = y.to(device)
            y_trues.append(y.cpu())
            probs = model(x.float())
            y_pred = probs > threshold
            y_preds.append(y_pred.cpu())
            y_probs.append(probs.cpu())

    y_trues = np.concatenate(y_trues)
    y_preds = np.concatenate(y_preds)
    y_probs = np.concatenate(y_probs)

    return y_trues, y_preds, y_probs


def flashnet_predict(
    model: torch.nn.Module,
    dataset: pd.DataFrame,
    norm_mean: np.ndarray | None = None,
    norm_std: np.ndarray | None = None,
    batch_size: int | None = -1,  # if None, then prediction_batch_size = 32
    device: torch.device | None = None,
    threshold: float = 0.5,
):
    if norm_mean is None or norm_std is None:
        log.warning("BE CAREFUL, norm_mean and norm_std are not provided! The model may not work properly.")

    if batch_size < 0:
        batch_size = 32

    dataset = dataset.copy(deep=True)

    if "io_type" not in dataset.columns:
        # assume that all data are read data
        return _predict(model, dataset, norm_mean, norm_std, batch_size, device, threshold)

    # NOTE: debug only
    dataset = dataset[dataset["io_type"] != 0]

    # check if there are write data
    write_indexes = dataset.index[dataset["io_type"] == 0].tolist()
    if len(write_indexes) > 0:
        log.info("There are write data in the dataset")
        # dataset is not only read data
        # we need to PREDICT only the read data
        original_len = len(dataset)
        read_indexes = dataset.index[dataset["io_type"] == 1].tolist()
        readonly_dataset = dataset.iloc[read_indexes]
        read_y_trues, read_y_preds, read_y_probs = _predict(model, readonly_dataset, norm_mean, norm_std, batch_size, device, threshold)

        # reconstruct the result
        y_trues = np.zeros(original_len, dtype=np.int64)  # set to 0 because we are sure that write data will be accepted (reject=0)
        y_preds = np.zeros(original_len, dtype=np.int64)  # set to 0 because we are sure that write data will be accepted (reject=0)
        y_probs = np.ones(original_len, dtype=np.float32)  # set to 1 because we are sure that write data will be accepted (reject=0)
        y_trues[read_indexes] = read_y_trues
        y_preds[read_indexes] = read_y_preds
        y_probs[read_indexes] = read_y_probs
        return y_trues, y_preds, y_probs
    else:
        # dataset is not only read data
        return _predict(model, dataset, norm_mean, norm_std, batch_size, device, threshold)


def prepare_data(
    dataset: pd.DataFrame,
    n_data: int | None = None,
    norm_mean: np.ndarray | None = None,
    norm_std: np.ndarray | None = None,
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
    x_train_indexes, x_val_indexes, final_y_train, final_y_val = train_test_split(indexes, y_train.values, test_size=0.2)

    final_x_train = x_train.values[x_train_indexes].astype(np.float32)
    final_x_val = x_train.values[x_val_indexes].astype(np.float32)
    final_y_train = final_y_train.astype(np.int64)
    final_y_val = final_y_val.astype(np.int64)

    train_dataset = FlashnetDataset(final_x_train, final_y_train, norm_mean=norm_mean, norm_std=norm_std)
    norm_mean = train_dataset.norm_mean
    norm_std = train_dataset.norm_std

    val_dataset = FlashnetDataset(final_x_val, final_y_val, norm_mean=norm_mean, norm_std=norm_std)

    log.info("Training the model with the following columns: %s", list(x_train.columns))

    return train_dataset, val_dataset, norm_mean, norm_std


load_model = torch.jit.load


def save_model(model: torch.nn.Module, model_path: str | Path):
    torch.jit.save(torch.jit.script(model), model_path)


def flashnet_train(
    model_path: str,
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
) -> FlashnetTrainResult:
    assert (norm_mean is None) == (norm_std is None)

    model_path = Path(model_path)

    if prediction_batch_size < 0:
        prediction_batch_size = batch_size

    ########################################
    # Preparing data
    ########################################

    train_dataset, val_dataset, norm_mean, norm_std = prepare_data(dataset, n_data=n_data, norm_mean=norm_mean, norm_std=norm_std)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, persistent_workers=True, num_workers=1)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, persistent_workers=True, num_workers=1)

    ########################################
    # Training
    ########################################

    if retrain and model_path.exists():
        model = cast(FlashnetModel, load_model(model_path))
    else:
        model = FlashnetModel(
            input_size=len(FEATURE_COLUMNS),
            hidden_layers=2,
            hidden_size=512,
            output_size=1,
        )

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
            y_true = []
            y_preds = []
            for x, y in val_loader:
                if device is not None:
                    x = x.to(device)
                    y = y.to(device)
                y_pred = model(x.float())
                val_loss += criterion(y_pred, y.float().view(-1, 1)).item()
                y_pred_label = y_pred > threshold
                y_true.extend(y.cpu().numpy())
                y_preds.extend(y_pred_label.cpu().numpy())

            val_loss /= len(val_loader)
            val_loss = round(val_loss, 4)
            y_true = np.array(y_true)
            y_preds = np.array(y_preds)
            val_result = flashnet_evaluate(y_true, y_preds)

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

    save_model(model, model_path)

    y_true, y_pred, y_probs = flashnet_predict(
        model, dataset, norm_mean=norm_mean, norm_std=norm_std, batch_size=prediction_batch_size, device=device, threshold=threshold
    )
    high_confidence_indices, low_confidence_indices = get_confidence_data(y_probs, threshold=confidence_threshold)
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
        num_io=len(y_true),
        confidence_threshold=confidence_threshold,
        low_confidence_indices=low_confidence_indices.tolist(),
        high_confidence_indices=high_confidence_indices.tolist(),
    )

    return result
