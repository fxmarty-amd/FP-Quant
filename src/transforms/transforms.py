import math
from abc import abstractmethod

from typing import Optional

import torch
import torch.nn as nn
from fast_hadamard_transform import hadamard_transform

from ..utils.common_utils import filter_kwarg_dict


class BaseTransform(nn.Module):

    def __init__(self, *args, **kwargs):
        super().__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor, inv_t: bool = False, dim: int = -1):
        pass


class IdentityTransform(BaseTransform):

    def __init__(self, *args, **kwargs):
        super().__init__()

    def forward(self, x: torch.Tensor, inv_t: bool = False, dim: int = -1):
        return x


class HadamardTransform(BaseTransform):

    def __init__(self, group_size: Optional[int]):
        super().__init__()

        if group_size is not None:
            self.group_size = group_size
            self.scale = 1 / math.sqrt(self.group_size)
        else:
            self.group_size = None

    def forward(self, x: torch.Tensor, inv_t: bool = False, dim: int = -1):
        if dim != -1:
            assert dim == 0
            assert x.ndim == 2

            x = x.T.contiguous()

        x_shape = x.shape
        # print("x_shape here", x_shape.shape)
        if self.group_size is not None:
            # Hadamard transform is it own inverse
            x = hadamard_transform(x.view(-1, self.group_size), scale=self.scale).view(x_shape)
        else:
            group_size = x_shape[-1]
            scale = 1 / math.sqrt(group_size)
            x = hadamard_transform(x.view(-1, group_size), scale=scale).view(x_shape)

        if dim != -1:
            x = x.T.contiguous()

        return x


TRANSFORMS = {
    "identity": IdentityTransform,
    "hadamard": HadamardTransform,
}

def build_transform(transform_class: str, **transform_kwargs) -> BaseTransform:
    transform = TRANSFORMS[transform_class]
    return transform(**filter_kwarg_dict(transform.__init__, transform_kwargs))

def get_transform_matrix(
    transform_class: str, 
    size: int, 
    device: torch.device = None, 
    dtype: torch.dtype = None
) -> torch.Tensor:
    if transform_class == "hadamard":
        return hadamard_transform(torch.eye(size, device=device, dtype=dtype), scale=1 / math.sqrt(size))
    elif transform_class == "identity":
        return torch.eye(size, device=device, dtype=dtype)
    else:
        raise NotImplementedError(f"get_transform_matrix is implemented only for Hadamard and Identity transforms")
