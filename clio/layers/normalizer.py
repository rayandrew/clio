import numpy as np

import torch
import torch.nn as nn

from clio.utils.logging import log_get

log = log_get(__name__)


# MIT LICENSE
# https://github.com/szymonmaszke/torchlayers/blob/master/torchlayers/preprocessing.py
class Normalize(nn.Module):
    """Normalize batch of tensor images with mean and standard deviation.

    Given mean values: `(M1,...,Mn)` and std values: `(S1,..,Sn)` for `n` channels
    (or other broadcastable to `n` values),
    this transform will normalize each channel of tensors in batch via formula:
    `output[channel] = (input[channel] - mean[channel]) / std[channel]`

    Parameters
    ----------
    mean : Tuple | List | torch.tensor
        Sequence of means for each channel
    std : Tuple | List | torch.tensor
        Sequence of means for each channel
    inplace : bool, optional
        Whether to run this operation in-place. Default: `False`

    """

    @classmethod
    def _transform_to_tensor(cls, tensor, name: str):
        if not torch.is_tensor(tensor):
            if isinstance(tensor, (tuple, list)):
                return torch.tensor(tensor)
            else:
                raise ValueError("{} is not an instance of either list, tuple or torch.tensor.".format(name))
        return tensor

    @classmethod
    def _check_shape(cls, tensor, name):
        if len(tensor.shape) > 1:
            raise ValueError("{} should be 0 or 1 dimensional tensor. Got {} dimensional tensor.".format(name, len(tensor.shape)))

    def __init__(self, mean: torch.Tensor, std: torch.Tensor, input_size: int, inplace: bool = False):
        tensor_mean = Normalize._transform_to_tensor(mean, "mean")
        tensor_std = Normalize._transform_to_tensor(std, "std")
        self._input_size = input_size
        Normalize._check_shape(tensor_mean, "mean")
        Normalize._check_shape(tensor_std, "std")

        # if torch.any(tensor_std == 0):
        #     raise ValueError("One or more std values are zero which would lead to division by zero.")

        super().__init__()

        self.register_buffer("mean", tensor_mean)
        self.register_buffer("std", tensor_std)
        self.inplace: bool = inplace

    def forward(self, inputs):
        # inputs_length = len(inputs.shape) - 2
        # mean = self.mean.view(1, -1, *([1] * inputs_length))  # type: ignore
        # std = self.std.view(1, -1, *([1] * inputs_length))  # type: ignore

        # mean = self.mean.view(1, -1
        mean = self.mean
        std = self.std

        if self.inplace:
            inputs.sub_(mean).div_(std)
            return inputs
        # redundancy to prevent zero division
        std = torch.clamp(std, min=1e-7)
        return (inputs - mean) / std


class NormalizerMixin:
    def __init__(self, input_size: int) -> None:
        self.input_size = input_size
        self.normalizer: nn.Module | None = None
        self.norm_mean: np.ndarray | None = None
        self.norm_std: np.ndarray | None = None

    def set(self, mean: np.ndarray, std: np.ndarray) -> None:
        self.normalizer = Normalize(torch.tensor(mean), torch.tensor(std), input_size=self.input_size)
        self.norm_mean = mean
        self.norm_std = std

    def adapt(self, x: torch.Tensor) -> None:
        """Adapts the model to the new input shape.

        :param x: input sample
        """
        if self.normalizer is None:
            norm_std, norm_mean = torch.std_mean(x, dim=0)
            self.norm_std = torch.clamp(norm_std, min=1e-7)  # prevent zero division
            self.normalizer = Normalize(norm_mean, norm_std, input_size=self.input_size)
            self.norm_mean = norm_mean.numpy()
            self.norm_std = norm_std.numpy()

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.normalizer is None:
            log.warning("Normalizer not initialized, returning input")
            return x

        return self.normalizer(x)

    def normalizer_info(self) -> str:
        return f"mean: {self.norm_mean}, std: {self.norm_std}"
