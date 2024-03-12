from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import DataLoader, Dataset

import clio.flashnet.ip_finder as ip_finder
from clio.flashnet.constants import FEATURE_COLUMNS
from clio.flashnet.eval import FlashnetEvaluationResult, flashnet_evaluate
from clio.layers.initialization import init_weights
from clio.layers.normalizer import NormalizerMixin
from clio.utils.logging import log_get
from clio.utils.timer import default_timer as timer

log = log_get(__name__)


@dataclass(kw_only=True)
class FlashnetTrainResult(FlashnetEvaluationResult):
    train_time: float  # in seconds
    prediction_time: float  # in seconds
    model_path: str
    norm_mean: np.ndarray
    norm_std: np.ndarray
    ip_threshold: float

    def eval_dict(self) -> dict:
        return super().as_dict()

    def as_dict(self):
        return {
            **super().as_dict(),
            "train_time": self.train_time,
            "prediction_time": self.prediction_time,
            "model_path": self.model_path,
            "norm_mean": self.norm_mean,
            "norm_std": self.norm_std,
            "ip_threshold": self.ip_threshold,
        }

    def __str__(self):
        return f"{super().__str__()}, Train Time: {self.train_time}, Prediction Time: {self.prediction_time}, Model Path: {self.model_path}, IP Threshold: {self.ip_threshold}"


# class FlashnetDataset(Dataset):
#     def __init__(self, data: pd.DataFrame | tuple[np.ndarray, np.ndarray], norm_mean: np.ndarray | None = None, norm_std: np.ndarray | None = None):
#         assert (norm_mean is None) == (norm_std is None)
#         assert all(col in data.columns for col in self.FEATURE_COLUMNS)

#         if isinstance(data, tuple):
#             self.data =

#         elif isinstance(data, pd.DataFrame):
#             # self.data = data
#             self.data = data.to_numpy()
#             # save column position
#             self.col_pos: dict[str, int] = {col: i for i, col in enumerate(data.columns)}
#             self.feat_cols = [self.col_pos[col] for col in self.FEATURE_COLUMNS]

#         if norm_mean is not None and norm_std is not None:
#             self._norm_mean = norm_mean
#             self._norm_std = norm_std
#         else:
#             # calculate using torch
#             norm_std, norm_mean = torch.std_mean(torch.tensor(self.data[:, self.feat_cols], dtype=torch.float32), dim=0)
#             self._norm_mean = norm_mean.numpy()
#             self._norm_std = norm_std.numpy()
#             self._norm_std = np.max(self._norm_std, 1e-7)

#     @property
#     def norm_mean(self) -> np.ndarray:
#         return self._norm_mean

#     @property
#     def norm_std(self) -> np.ndarray:
#         return self._norm_std

#     def normalize(self, x: np.ndarray) -> np.ndarray:
#         if self.norm_mean is None or self.norm_std is None:
#             log.warning("Normalizer not initialized, returning input")
#             return x
#         return (x - self.norm_mean) / self.norm_std

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, idx):
#         row = self.data[idx]
#         x = row[self.feat_cols]
#         reject = row[self.col_pos["reject"]]
#         x = self.normalize(x)
#         return x, reject


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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        # x = self.normalize(x)
        x = self.features(x)
        x = self.classifier(x)
        return x

    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.view(x.size(0), -1)
        # x = self.normalize(x)
        return self.features(x)


def flashnet_predict(
    model: torch.nn.Module,
    dataset: pd.DataFrame,
    norm_mean: np.ndarray | None = None,
    norm_std: np.ndarray | None = None,
    batch_size: int | None = -1,  # if None, then prediction_batch_size = 32
    device: torch.device | None = None,
    threshold: float = 0.5,
):
    if batch_size < 0:
        batch_size = 32

    dataset = dataset.copy(deep=True)

    # drop io_type == 0
    if "io_type" in dataset.columns:
        dataset = dataset[dataset["io_type"] != 0]

    x = dataset.drop(columns=dataset.columns.difference(FEATURE_COLUMNS), axis=1).values
    y = dataset["reject"].values

    eval_dataset = FlashnetDataset(x, y, norm_mean=norm_mean, norm_std=norm_std)
    eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False)

    model.eval()

    y_trues = []
    y_preds = []
    with torch.no_grad():
        for x, y in eval_loader:
            if device is not None:
                x = x.to(device)
                y = y.to(device)
            y_trues.append(y.cpu())
            y_pred = model(x.float()) > threshold
            y_preds.append(y_pred.cpu())

    y_trues = np.concatenate(y_trues)
    y_preds = np.concatenate(y_preds)

    return y_trues, y_preds


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
) -> FlashnetTrainResult:
    assert (norm_mean is None) == (norm_std is None)

    model_path = Path(model_path)

    if prediction_batch_size < 0:
        prediction_batch_size = batch_size

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
    final_x_train = x_train.values[x_train_indexes]
    final_x_val = x_train.values[x_val_indexes]

    train_dataset = FlashnetDataset(final_x_train, final_y_train, norm_mean=norm_mean, norm_std=norm_std)
    norm_mean = train_dataset.norm_mean
    norm_std = train_dataset.norm_std
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    val_dataset = FlashnetDataset(final_x_val, final_y_val, norm_mean=norm_mean, norm_std=norm_std)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    ########################################
    # Training
    ########################################

    if retrain and model_path.exists():
        model = torch.jit.load(model_path)
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

    log.info("Training the model with the following columns: %s", list(x_train.columns))
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
            epoch,
            epochs,
            loss.item(),
            val_loss,
            val_result.accuracy,
            val_result.auc,
            tab=1,
        )

    train_time = timer() - start_time

    model_script = torch.jit.script(model)
    model_script.save(model_path)

    y_true, y_pred = flashnet_predict(
        model, dataset, norm_mean=norm_mean, norm_std=norm_std, batch_size=prediction_batch_size, device=device, threshold=threshold
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
