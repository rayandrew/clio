from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from sklearn.metrics import confusion_matrix, roc_auc_score

from clio.utils.indented_file import IndentedFile
from clio.utils.logging import log_get

log = log_get(__name__)


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

    def to_indented_file(self, file: IndentedFile):
        file.writeln("Accuracy: %s", self.accuracy)
        file.writeln("Precision: %s", self.precision)
        file.writeln("Recall: %s", self.recall)
        file.writeln("F1 Score: %s", self.f1)
        file.writeln("ROC-AUC: %s", self.auc)
        file.writeln("FPR: %s", self.fpr)
        file.writeln("FNR: %s", self.fnr)


def flashnet_evaluate(labels: npt.ArrayLike, predictions: npt.ArrayLike) -> FlashnetEvaluationResult:
    stats = []
    cm = confusion_matrix(labels, predictions, labels=[0, 1])
    # log.info("Confusion Matrix", tab=1)
    # log.info(cm, tab=2)

    # Calculate ROC-AUC and FPR/FNR
    cm_values = cm.ravel()
    TN, FP, FN, TP = cm_values[0], cm_values[1], cm_values[2], cm_values[3]
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    f1_sc = 2 * (precision * recall) / (precision + recall) if precision + recall > 0 else 0

    FPR, FNR = round(FP / (FP + TN + 0.1), 3), round(FN / (TP + FN + 0.1), 3)

    try:
        ROC_AUC = round(roc_auc_score(labels, predictions), 3)
    except ValueError:
        ROC_AUC = 0  # if all value are classified into one class, which is BAD dataset

    stats.append("FPR = %s (%s %%)" % (FPR, round(FPR * 100, 1)))
    stats.append("FNR = %s (%s %%)" % (FNR, round(FNR * 100, 1)))
    stats.append("ROC-AUC = %s (%s %%)" % (ROC_AUC, round(ROC_AUC * 100, 1)))

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


def flashnet_uncertain_prediction(y_probs: npt.ArrayLike, threshold: float = 0.5, **kwargs):
    threshold = np.array([threshold] * len(y_probs))
    return np.isclose(y_probs, threshold, **kwargs).astype(int)


__all__ = [
    "flashnet_predict",
    "flashnet_evaluate",
    "FlashnetEvaluationResult",
]
