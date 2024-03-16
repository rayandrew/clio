from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import torch

import clio.flashnet.ip_finder as ip_finder
from clio.flashnet.constants import FEATURE_COLUMNS
from clio.flashnet.eval import flashnet_evaluate
from clio.flashnet.training.shared import FlashnetTrainResult
from clio.flashnet.training.simple import FlashnetDataset, FlashnetModel, flashnet_predict, prepare_data

from clio.utils.logging import log_get
from clio.utils.timer import default_timer as timer


@dataclass(kw_only=True)
class FlashnetStudentTrainResult(FlashnetTrainResult):
    low_confidence_idx: list[int] = field(default_factory=list)

    def eval_dict(self) -> dict:
        return super().as_dict()

    def as_dict(self):
        return {
            **super().as_dict(),
        }

    def __str__(self):
        return f"{super().__str__()}, N Low Confidence Data: {len(self.low_confidence_idx)}"
