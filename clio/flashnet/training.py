from dataclasses import dataclass
from pathlib import Path
from timeit import default_timer as timer
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

import keras

import clio.flashnet.ip_finder as ip_finder
from clio.flashnet.eval import FlashnetEvaluationResult, flashnet_evaluate
from clio.utils.keras import load_model, save_model
from clio.utils.logging import log_get

log = log_get(__name__)


def batch_generator(x: np.ndarray, y: np.ndarray, batch_size: int):

    indices = np.arange(x.shape[0])
    # np.random.shuffle(indices)
    # for start in range(0, len(indices), batch_size):
    #     end = min(start + batch_size, len(indices))
    #     batch_idx = indices[start:end]
    #     yield x[batch_idx], y[batch_idx]

    while True:
        np.random.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            end = min(start + batch_size, len(indices))
            batch_idx = indices[start:end]
            yield x[batch_idx], y[batch_idx]


@dataclass(kw_only=True)
class FlashnetTrainResult(FlashnetEvaluationResult):
    train_time: float  # in seconds
    model_path: str
    norm_mean: np.ndarray
    norm_variance: np.ndarray
    ip_threshold: float

    def eval_dict(self) -> dict:
        return super().as_dict()

    def as_dict(self):
        return {
            **super().as_dict(),
            "train_time": self.train_time,
            "model_path": self.model_path,
            "norm_mean": self.norm_mean,
            "norm_variance": self.norm_variance,
            "ip_threshold": self.ip_threshold,
        }

    def __str__(self):
        return f"{super().__str__()}, Train Time: {self.train_time}, Model Path: {self.model_path}, Norm Mean: {self.norm_mean}, Norm Variance: {self.norm_variance}, IP Threshold: {self.ip_threshold}"


def flashnet_train(
    model_path: Path | str,
    dataset_ori: pd.DataFrame,
    retrain: bool = False,
    batch_size: int | None = 32,  # if None, then batch_size = 32
    prediction_batch_size: int | None = -1,  # if None, then prediction_batch_size = 32
    tqdm: bool = False,
    lr: float = 0.0001,
    epochs: int = 20,
    norm_mean: np.ndarray | None = None,
    norm_variance: np.ndarray | None = None,
    n_data: int | None = None,
) -> FlashnetTrainResult:
    # if norm_mean is None then norm_variance is None
    # and if norm_mean is not None then norm_variance is not None
    assert (norm_mean is None) == (norm_variance is None)

    model_path = Path(model_path)

    if prediction_batch_size < 0:
        prediction_batch_size = batch_size

    dataset = dataset_ori.copy(deep=True)

    # drop io_type == 0
    if "io_type" in dataset.columns:
        dataset = dataset[dataset["io_type"] != 0]

    start = timer()
    # Drop unnecessary columns
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

    # Put "latency" at the end
    start = timer()
    reordered_cols = [col for col in dataset.columns if col != "latency"] + ["latency"]
    dataset = dataset[reordered_cols]
    log.debug("Time to reorder columns: %f", timer() - start)

    if n_data is not None:
        if n_data > 0:
            # get final n_data rows as train and 0.2*n_data as validation
            # dataset = dataset.tail(n_data + int(0.2 * n_data))
            # get final n data rows
            dataset = dataset.tail(n_data)

    log.info("Train data size: %d", len(dataset))

    start = timer()
    x_train = dataset.drop(columns=["latency", "reject"], axis=1)
    y_train = dataset["reject"]
    log.debug("Time to copy and drop columns: %f", timer() - start)

    log.debug("Training the model with the following columns: %s", list(x_train.columns))
    log.debug("Training size: %d", len(x_train) * 0.8)
    log.debug("Validation size: %d", len(x_train) * 0.2)

    # Store the true/real value, for evaluating the ROC-AUC score
    true_reject = dataset["reject"].values.tolist()  # the rejection label

    # split data
    start = timer()
    indexes = np.arange(len(x_train))
    x_train_indexes, x_val_indexes, final_y_train, final_y_val = train_test_split(indexes, y_train.values, test_size=0.2, random_state=42)
    final_x_train = x_train.values[x_train_indexes]
    final_x_val = x_train.values[x_val_indexes]
    log.debug("Time to split data: %f", timer() - start)

    start = timer()
    train_generator = batch_generator(final_x_train, final_y_train, batch_size)
    log.debug("Time to create batch generator: %f", timer() - start)

    start = timer()
    val_generator = batch_generator(final_x_val, final_y_val, batch_size)
    log.debug("Time to create validation batch generator: %f", timer() - start)

    # Train the model
    if retrain:
        clf = load_model(model_path)
        # clf.fit(x_train, y_train, validation_split=0.2, verbose=0, epochs=20)A
        start_time = timer()
        clf.fit(
            train_generator,
            validation_data=val_generator,
            # validation_split=0.2,
            verbose=0,
            epochs=epochs,
            batch_size=batch_size,
            steps_per_epoch=len(final_x_train) // batch_size,
            validation_steps=len(final_x_val) // batch_size,
        )
        train_time = timer() - start_time
        log.info("Time to retrain model: %f", train_time)
        # Save the model
        save_model(clf, model_path)
    else:
        # Data normalization
        normalizer = keras.layers.Normalization(axis=-1, mean=norm_mean, variance=norm_variance)
        if norm_mean is None and norm_variance is None:
            start = timer()
            normalizer.adapt(np.array(x_train))
            log.debug("Time to normalize data: %f", timer() - start)
        else:
            normalizer.build(x_train.shape)

        # log_info("Normalizer")
        if norm_mean is not None and norm_variance is not None:
            # NOTE: DO NOT use print this unless needed (synchronization between CPU and GPU)
            # log.debug("Mean: %s", norm_mean, tab=1)
            # log.debug("Variance: %s", norm_variance, tab=1)
            ...
        else:
            norm_mean = normalizer.mean.numpy()
            norm_variance = normalizer.variance.numpy()
            # NOTE: DO NOT use print this unless needed (synchronization between CPU and GPU)
            # log.debug("Variance: %s", normalizer.variance.numpy(), tab=1)
            # log.debug("Mean: %s", normalizer.mean.numpy(), tab=1)

        # start = timer()
        clf = keras.Sequential(
            [
                keras.layers.Input(shape=(x_train.shape[1],)),
                normalizer,
                keras.layers.Dense(512, activation="relu"),
                keras.layers.Dense(512, activation="relu"),
                keras.layers.Dense(1, activation="sigmoid"),
            ]
        )
        clf.compile(loss="binary_crossentropy", optimizer=keras.optimizers.Adam(learning_rate=lr), metrics=["accuracy"])
        callbacks = []
        if tqdm:
            from tqdm.keras import TqdmCallback

            callbacks.append(TqdmCallback(verbose=2))
        # log_info("Time to create model: %f", timer() - start)

        start = timer()
        # clf.fit(x_train, y_train, validation_split=0.2, verbose=0, epochs=epochs, callbacks=callbacks, batch_size=batch_size)
        clf.fit(
            train_generator,
            validation_data=val_generator,
            verbose=0,
            epochs=epochs,
            callbacks=callbacks,
            batch_size=batch_size,
            steps_per_epoch=len(final_x_train) // batch_size,
            validation_steps=len(final_x_val) // batch_size,
        )
        train_time = timer() - start
        log.info("Time to train model: %f", train_time)
        save_model(clf, model_path)

    # Evaluation
    start = timer()
    y_pred = (clf.predict(x_train, verbose=0, batch_size=prediction_batch_size) > 0.5).flatten()
    log.debug("Time to predict data: %f", timer() - start)

    dataset["pred_reject"] = y_pred

    # Calculate IP_Finder threshold
    # IP threshold is used for getting reject/accept decision based on the predicted latency
    ip_latency_threshold, _ = ip_finder.area_based(dataset["latency"])  # y_pred is array of predicted latencies

    # Print confusion matrix and stats
    eval_result = flashnet_evaluate(true_reject, y_pred)

    result = FlashnetTrainResult(
        **eval_result.as_dict(),
        stats=eval_result.stats,
        train_time=train_time,
        model_path=model_path,
        norm_mean=norm_mean,
        norm_variance=norm_variance,
        ip_threshold=ip_latency_threshold,
    )

    return result


__all__ = ["flashnet_train", "FlashnetTrainResult"]
