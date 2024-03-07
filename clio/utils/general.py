def general_set_seed(seed: int) -> None:
    import random

    import numpy as np

    np.random.seed(seed)
    random.seed(seed)


def tf_set_seed(seed: int) -> None:
    import tensorflow as tf

    general_set_seed(seed)
    tf.random.set_seed(seed)


def torch_set_seed(seed: int) -> None:
    import torch

    general_set_seed(seed)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


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


__all__ = [
    "general_set_seed",
    "tf_set_seed",
    "torch_set_seed",
]
