from dataclasses import dataclass

import numpy.typing as npt
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_auc_score

import keras

from clio.utils.keras import Trainer


@dataclass(kw_only=True)
class FlashnetEvaluationResult:
    stats: list[str]
    accuracy: float
    precision: float
    recall: float
    f1: float
    auc: float
    fpr: float
    fnr: float

    def __str__(self):
        return f"Accuracy: {self.accuracy}, Precision: {self.precision}, Recall: {self.recall}, F1 Score: {self.f1}, ROC-AUC: {self.auc}, FPR: {self.fpr}, FNR: {self.fnr}"

    def as_dict(self):
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "auc": self.auc,
            "fpr": self.fpr,
            "fnr": self.fnr,
        }


def flashnet_evaluate(y_test: npt.ArrayLike, y_pred: npt.ArrayLike) -> FlashnetEvaluationResult:
    y_test_class, y_pred_class = y_test, y_pred
    stats = []
    cm = confusion_matrix(y_test_class, y_pred_class)
    # Calculate ROC-AUC and FPR/FNR
    cm_values = cm.ravel()
    TN, FP, FN, TP = cm_values[0], cm_values[1], cm_values[2], cm_values[3]
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1_sc = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    FPR, FNR = round(FP / (FP + TN + 0.1), 3), round(FN / (TP + FN + 0.1), 3)

    try:
        ROC_AUC = round(roc_auc_score(y_test, y_pred), 3)
    except ValueError:
        ROC_AUC = 0  # if all value are classified into one class, which is BAD dataset

    stats.append("FPR = " + str(FPR) + "  (" + str(round(FPR * 100, 1)) + "%)")
    stats.append("FNR = " + str(FNR) + "  (" + str(round(FNR * 100, 1)) + "%)")
    stats.append("ROC-AUC = " + str(ROC_AUC) + "  (" + str(round(ROC_AUC * 100, 1)) + "%)")

    return FlashnetEvaluationResult(
        stats=stats,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1_sc,
        auc=ROC_AUC,
        fpr=FPR,
        fnr=FNR,
    )


def flashnet_predict(model: Trainer, data: pd.DataFrame, batch_size: int | None = None, tqdm: bool = False) -> tuple[list[int], list[int]]:
    pred_arr = []
    true_arr = []

    df = data.copy(deep=True).drop(columns=["latency", "reject", "ts_record", "original_ts_record"], axis=1, errors="ignore")

    callbacks: list[keras.callbacks.Callback] = []
    if tqdm:
        from tqdm.keras import TqdmCallback

        callbacks.append(TqdmCallback(verbose=2))

    pred_arr = (model.predict(df, verbose=0, batch_size=batch_size, callbacks=callbacks) > 0.5).flatten().tolist()
    true_arr = data["reject"].tolist()

    return pred_arr, true_arr


__all__ = ["flashnet_predict", "flashnet_evaluate", "FlashnetEvaluationResult"]
