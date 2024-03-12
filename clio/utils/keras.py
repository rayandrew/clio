from pathlib import Path

import keras

# from keras.src.trainers.trainer import Trainer

## add typing to these utilitiy functions


def load_model(path: str | Path):
    return keras.models.load_model(path)


def save_model(model, path: str | Path) -> None:
    model.save(path)


def tf_load_model(path: str | Path):
    import tensorflow as tf

    return tf.keras.models.load_model(path)


def tf_save_model(model, path: str | Path) -> None:
    import tensorflow as tf

    model.save(path)


__all__ = ["load_model", "save_model", "Trainer"]
