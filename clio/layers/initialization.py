import torch.nn as nn


def weight_init(m: nn.Module) -> None:
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain("relu"))
        nn.init.zeros_(m.bias)


def init_weights(model: nn.Module) -> None:
    model.apply(weight_init)


__all__ = ["init_weights", "weight_init"]
