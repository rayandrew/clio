from dataclasses import dataclass, field

import numpy as np

from serde import serde

from clio.utils.characteristic import Statistic
from clio.utils.general import ratio_to_percentage, ratio_to_percentage_str
from clio.utils.indented_file import IndentedFile
from clio.utils.logging import log_get
from clio.utils.metric import binary_uncertainty

log = log_get(__name__)


@serde
@dataclass(kw_only=True, frozen=True)
class UncertaintyResult:
    uncertainty: np.ndarray
    sorted_indices: np.ndarray
    statistic: Statistic = field(default_factory=Statistic)

    def as_dict(self):
        return {f"uncertainty_{k}": v for k, v in self.statistic.to_dict().items()}


def get_uncertainty_result(labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray) -> UncertaintyResult:
    probabilities = np.array(probabilities)
    probabilities = probabilities.copy()

    # Calculate the uncertainty
    uncertainty = binary_uncertainty(probabilities)

    # find the indices of the most uncertain samples
    uncertain_indices = np.argsort(uncertainty)[::-1]

    # print the most uncertain samples
    log.info("Most uncertain samples:", tab=1)
    if len(uncertain_indices) > 0:
        for i in range(min(len(uncertain_indices), 5)):
            index = uncertain_indices[i]
            log.info(
                "Sample %s, Label: %s, Prediction: %s, Probability: %s, Uncertainty: %s",
                index,
                labels[index],
                predictions[index],
                ratio_to_percentage_str(probabilities[index]),
                uncertainty[index],
                tab=2,
            )

    return UncertaintyResult(
        uncertainty=uncertainty,
        statistic=Statistic.generate(uncertainty),
        sorted_indices=uncertain_indices,
    )


__all__ = [
    "UncertaintyResult",
    "get_uncertainty_result",
]
