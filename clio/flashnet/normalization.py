from pathlib import Path

import numpy as np

from clio.utils.logging import log_get

log = log_get(__name__)


def get_cached_norm(path: str | Path) -> tuple[np.ndarray | None, np.ndarray | None]:
    # check if norm_mean and norm_std exists in train_data
    norm_mean = None
    norm_std = None
    norm_mean_path = path / "norm_mean.npy"
    norm_std_path = path / "norm_std.npy"
    if norm_mean_path.exists() and norm_std_path.exists():
        log.info("Loading precomputed norm_mean and norm_std", tab=2)
        norm_mean = np.load(norm_mean_path, allow_pickle=True)
        norm_std = np.load(norm_std_path, allow_pickle=True)
        # check if norm_mean and norm_std is valid
        if norm_mean.size <= 1 or norm_std.size <= 1:
            log.error("Invalid norm_mean and norm_std")
            norm_mean = None
            norm_std = None

    return norm_mean, norm_std


def save_norm(path: str | Path, norm_mean: np.ndarray, norm_std: np.ndarray) -> None:
    norm_mean_path = path / "norm_mean.npy"
    norm_std_path = path / "norm_std.npy"
    np.save(norm_mean_path, norm_mean)
    np.save(norm_std_path, norm_std)


def parse_norm(norm_str: str) -> np.ndarray:
    return np.fromstring(norm_str, sep="\n")


def norm_to_str(norm: np.ndarray) -> str:
    return "\n".join(norm.astype(str))


__all__ = ["get_cached_norm", "save_norm", "parse_norm", "norm_to_str"]
