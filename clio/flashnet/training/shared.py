from dataclasses import dataclass, field

import numpy as np

from clio.flashnet.eval import FlashnetEvaluationResult, PredictionResult

from clio.utils.indented_file import IndentedFile


@dataclass(kw_only=True)
class FlashnetTrainResult(FlashnetEvaluationResult):
    train_time: float  # in seconds
    prediction_time: float  # in seconds
    model_path: str
    norm_mean: np.ndarray
    norm_std: np.ndarray
    ip_threshold: float
    confidence_threshold: float
    labels: np.ndarray
    predictions: np.ndarray
    probabilities: np.ndarray

    @property
    def num_io(self) -> int:
        assert (
            len(self.labels) == len(self.predictions) == len(self.probabilities)
        ), f"Length of labels ({len(self.labels)}), predictions ({len(self.predictions)}), and probabilities ({len(self.probabilities)}) must be the same"

        return len(self.labels)

    @property
    def num_reject_io(self) -> int:
        return np.sum(self.labels == 1)

    @property
    def num_accept_io(self) -> int:
        return np.sum(self.labels == 0)

    @property
    def prediction_result(self) -> PredictionResult:
        return PredictionResult(
            labels=self.labels,
            predictions=self.predictions,
            probabilities=self.probabilities,
        )

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
            "confidence_threshold": self.confidence_threshold,
            "num_io": self.num_io,
            "num_reject_io": self.num_reject_io,
            "num_accept_io": self.num_accept_io,
        }

    def __str__(self):
        return f"{super().__str__()}, Train Time: {self.train_time}, Prediction Time: {self.prediction_time}, Model Path: {self.model_path}, IP Threshold: {self.ip_threshold},  Confidence Threshold: {self.confidence_threshold}"

    def to_indented_file(self, file: IndentedFile):
        super().to_indented_file(file)
        file.writeln("Train Time: %s", self.train_time)
        file.writeln("Prediction Time: %s", self.prediction_time)
        file.writeln("Model Path: %s", self.model_path)
        file.writeln("IP Threshold: %s", self.ip_threshold)
        file.writeln("Confidence Threshold: %s", self.confidence_threshold)
        with file.section("IO"):
            file.writeln("Total: %s", self.num_io)
            file.writeln("Reject: %s", self.num_reject_io)
            file.writeln("Accept: %s", self.num_accept_io)


__all__ = ["FlashnetTrainResult"]
