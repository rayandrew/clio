from pathlib import Path

import keras
from keras.src.trainers.trainer import Trainer

## add typing to these utilitiy functions


def load_model(path: str | Path) -> Trainer:
    return keras.models.load_model(path)


def save_model(model: Trainer, path: str | Path) -> None:
    model.save(path)


__all__ = ["load_model", "save_model", "Trainer"]
