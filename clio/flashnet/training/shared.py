from dataclasses import dataclass, field

import numpy as np

from clio.flashnet.eval import FlashnetEvaluationResult


@dataclass(kw_only=True)
class FlashnetTrainResult(FlashnetEvaluationResult):
    train_time: float  # in seconds
    prediction_time: float  # in seconds
    model_path: str
    norm_mean: np.ndarray
    norm_std: np.ndarray
    ip_threshold: float
    num_io: int
    confidence_threshold: float
    low_confidence_indices: list[int] = field(default_factory=list)
    high_confidence_indices: list[int] = field(default_factory=list)

    @property
    def num_low_confidence(self):
        return len(self.low_confidence_indices)

    @property
    def num_high_confidence(self):
        return len(self.high_confidence_indices)

    @property
    def ratio_low_confidence(self) -> float:
        return self.num_low_confidence / self.num_io if self.num_io > 0 else 0.0

    @property
    def ratio_high_confidence(self) -> float:
        ratio = self.num_high_confidence / self.num_io if self.num_io > 0 else 0.0
        assert ratio + self.ratio_low_confidence == 1.0
        return ratio

    def eval_dict(self) -> dict:
        assert self.num_io == len(self.low_confidence_indices) + len(
            self.high_confidence_indices
        ), f"Num IO ({self.num_io}) != Num of Low Confidence ({len(self.low_confidence_indices)}) + Num of High Confidence ({len(self.high_confidence_indices)})"
        return super().as_dict()

    def as_dict(self):
        assert self.num_io == len(self.low_confidence_indices) + len(
            self.high_confidence_indices
        ), f"Num IO ({self.num_io}) != Num of Low Confidence ({len(self.low_confidence_indices)}) + Num of High Confidence ({len(self.high_confidence_indices)})"
        return {
            **super().as_dict(),
            "train_time": self.train_time,
            "prediction_time": self.prediction_time,
            "model_path": self.model_path,
            "norm_mean": self.norm_mean,
            "norm_std": self.norm_std,
            "ip_threshold": self.ip_threshold,
            "num_io": self.num_io,
            "confidence_threshold": self.confidence_threshold,
            "low_confidence_indices": self.low_confidence_indices,
            "num_low_confidence": self.num_low_confidence,
            "ratio_low_confidence": self.ratio_low_confidence,
            "high_confidence_indices": self.high_confidence_indices,
            "num_high_confidence": self.num_high_confidence,
        }

    def __str__(self):
        assert self.num_io == len(self.low_confidence_indices) + len(
            self.high_confidence_indices
        ), f"Num IO ({self.num_io}) != Num of Low Confidence ({len(self.low_confidence_indices)}) + Num of High Confidence ({len(self.high_confidence_indices)})"
        return f"{super().__str__()}, Train Time: {self.train_time}, Prediction Time: {self.prediction_time}, Model Path: {self.model_path}, IP Threshold: {self.ip_threshold},  Confidence Threshold: {self.confidence_threshold} Low Confidence: {self.num_low_confidence} ({self.ratio_low_confidence:.2%}), High Confidence: {self.num_high_confidence} ({self.ratio_high_confidence:.2%})"


__all__ = ["FlashnetTrainResult"]
