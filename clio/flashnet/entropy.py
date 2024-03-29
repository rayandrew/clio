from dataclasses import dataclass, field

import numpy as np

from serde import serde

from clio.utils.characteristic import Statistic
from clio.utils.general import ratio_to_percentage, ratio_to_percentage_str
from clio.utils.indented_file import IndentedFile
from clio.utils.logging import log_get
from clio.utils.metric import binary_entropy

log = log_get(__name__)


@serde
@dataclass(kw_only=True, frozen=True)
class EntropyResult:
    entropy: np.ndarray
    sorted_indices: np.ndarray
    statistic: Statistic = field(default_factory=Statistic)

    def as_dict(self):
        return {f"entropy_{k}": v for k, v in self.statistic.to_dict().items()}


def get_entropy_result(labels: np.ndarray, predictions: np.ndarray, probabilities: np.ndarray) -> EntropyResult:
    probabilities = np.array(probabilities)

    # Calculate the uncertainty
    entropy = binary_entropy(probabilities)

    # find the indices of the most uncertain samples
    entropy_indices = np.argsort(entropy)[::-1]

    # print the most uncertain samples
    log.info("Entropy samples:", tab=1)
    for i in range(5):
        index = entropy_indices[i]
        log.info(
            "Sample %s, Label: %s, Prediction: %s, Probability: %s, Entropy: %s",
            index,
            labels[index],
            predictions[index],
            ratio_to_percentage_str(probabilities[index]),
            entropy[index],
            tab=2,
        )

    return EntropyResult(
        entropy=entropy,
        statistic=Statistic.generate(entropy),
        sorted_indices=entropy_indices,
    )


__all__ = [
    "EntropyResult",
    "get_entropy_result",
]
