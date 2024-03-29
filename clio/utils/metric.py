from typing import TypeVar

import numpy as np
import numpy.typing as npt

import torch

T = TypeVar("T", np.ndarray, torch.Tensor, npt.ArrayLike, list, covariant=True)


def entropy_torch(p: torch.Tensor) -> torch.Tensor:
    return -torch.sum(p * torch.log(torch.clamp(p, 1e-8)))


def entropy_np(p: np.ndarray | list) -> torch.Tensor:
    p = np.array(p)
    return -np.sum(p * np.log(p.clip(1e-8)))


def entropy(p: T) -> T:
    if isinstance(p, torch.Tensor):
        return entropy_torch(p)
    elif isinstance(p, (np.ndarray, list)):
        return entropy_np(p)
    else:
        raise ValueError("p should be either numpy array or torch tensor")


def binary_entropy_torch(p: torch.Tensor) -> torch.Tensor:
    eps = torch.finfo(p.dtype).eps
    p = torch.clamp(p, eps, 1 - eps)
    return -p * torch.log2(p) - (1 - p) * torch.log2(1 - p)


def binary_entropy_np(p: np.ndarray | list) -> np.ndarray:
    p = np.array(p)
    eps = np.finfo(p.dtype).eps
    p = np.clip(p, eps, 1 - eps)
    return -p * np.log2(p) - (1 - p) * np.log2(1 - p)


def binary_entropy(p: T) -> T:
    if isinstance(p, torch.Tensor):
        return binary_entropy_torch(p)
    elif isinstance(p, (np.ndarray, list)):
        return binary_entropy_np(p)
    else:
        raise ValueError("p should be either numpy array or torch tensor")


def binary_uncertainty_torch(p: torch.Tensor) -> torch.Tensor:
    return 1 - torch.abs(p - (1 - p))


def binary_uncertainty_np(p: np.ndarray) -> np.ndarray:
    p = np.array(p)
    return 1 - np.abs(p - (1 - p))


def binary_uncertainty(p: T) -> T:
    if isinstance(p, torch.Tensor):
        return binary_uncertainty_torch(p)
    elif isinstance(p, (np.ndarray, list)):
        return binary_uncertainty_np(p)
    else:
        raise ValueError("p should be either numpy array or torch tensor")


__all__ = ["entropy", "binary_entropy", "binary_uncertainty"]
