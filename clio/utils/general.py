import random

import numpy as np

import torch


def general_set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)


def tf_set_seed(seed: int) -> None:
    import tensorflow as tf

    general_set_seed(seed)
    tf.random.set_seed(seed)


def torch_set_seed(seed: int) -> None:

    general_set_seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def enable_dropout_module(model: torch.nn.Module) -> None:
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()


def enable_dropout_torchscript_module(model: torch.jit.ScriptModule) -> None:
    for m in model.modules():
        if m.original_name.startswith("Dropout"):
            m.train()


def enable_dropout(model: torch.nn.Module | torch.jit.ScriptModule) -> None:
    if isinstance(model, torch.jit.ScriptModule):
        enable_dropout_torchscript_module(model)
    else:
        enable_dropout_module(model)


def parse_time(time: int | str) -> int:
    """
    Parse time from string to int and convert to minutes if necessary.

    :param time: The time to parse.
    :return: The parsed duration in minutes.
    """
    if isinstance(time, int):
        return time
    if isinstance(time, str):
        if time.isdigit():
            return int(time)
        if time.endswith("m"):
            return int(time[:-1])
        if time.endswith("min"):
            return int(time[:-3])
        if time.endswith("h"):
            return int(time[:-1]) * 60
        if time.endswith("hr"):
            return int(time[:-2]) * 60
        if time.endswith("hour"):
            return int(time[:-4]) * 60
        if time.endswith("hours"):
            return int(time[:-5]) * 60
        if time.endswith("d"):
            return int(time[:-1]) * 60 * 24
        if time.endswith("day"):
            return int(time[:-3]) * 60 * 24
        if time.endswith("days"):
            return int(time[:-4]) * 60 * 24

    raise ValueError(f"Invalid time: {time}")


def ratio_to_percentage(ratio: float, rounding: int = -1) -> float:
    return round(ratio * 100, rounding) if rounding >= 0 else ratio * 100


def percentage_to_ratio(percentage: float, rounding: int = -1) -> float:
    return round(percentage / 100, rounding) if rounding >= 0 else percentage / 100


def ratio_to_percentage_str(ratio: float, rounding: int = -1) -> str:
    return f"{ratio_to_percentage(ratio, rounding=rounding):.2f}%"


__all__ = [
    "general_set_seed",
    "tf_set_seed",
    "torch_set_seed",
    "parse_time",
    "ratio_to_percentage",
    "percentage_to_ratio",
    "ratio_to_percentage_str",
    "enable_dropout",
]
