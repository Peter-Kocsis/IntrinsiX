from typing import Iterable, Mapping, Callable, Union, MutableMapping

import torch

from batch import Batch

from torchvision.transforms import RandomHorizontalFlip
import torchvision.transforms.functional as F

# =========================================== Transforms ===========================================
class DynamicTransform(torch.nn.Module):
    pass


class Compose(DynamicTransform):
    def __init__(self, transforms: Mapping[str, Callable], *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.transforms = transforms

    def __call__(self, img, *args, **kwargs):
        for t in self.transforms.values():
            if t is None:
                continue

            if isinstance(t, DynamicTransform):
                img = t(img, *args, **kwargs)
            else:
                img = t(img)
        return img


class BatchTransform(torch.nn.Module):
    def __init__(self, transform: Union[Mapping[str, Callable], Callable], reset_params=False, *args, **kwargs):
        super().__init__()
        self.transform = transform
        self.reset_params = reset_params

    def __getitem__(self, index) -> Callable:
        if isinstance(self.transform, Mapping):
            return self.transform.get(index, self.transform.get("_default", None))
        else:
            return self.transform

    def forward(self, x_dict: Batch) -> Mapping[str, torch.Tensor]:
        """
        Transforms the elements of a dictionary according to the transform table.
        :param x_dict: The input dictionary
        :return: The transformed dictionary
        """
        x_out = x_dict
        if self.reset_params:
            self.reset_parameters()
        for key in x_dict.keys(depth=-1):
            transform = self[key]
            if transform is not None:
                val = x_dict[key]

                if isinstance(transform, MutableMapping):
                    for out_key, t in transform.items():
                        x_out[out_key] = self.eval_transform(t, val, x_dict)
                else:
                    x_out[key] = self.eval_transform(transform, val, x_dict)

        return x_out

    def eval_transform(self, transform, val, batch):
        if isinstance(transform, DynamicTransform):
            return transform(val, batch)
        else:
            return transform(val)

    def inverse(self, x_trans_dict: Batch) -> Mapping[str, torch.Tensor]:
        """
        Inverse transforms the elements of a dictionary according to the transform table.
        :param x_dict: The transformed dictionary
        :return: The inverse transformed dictionary
        """
        x_out = x_trans_dict
        for key in x_trans_dict.keys(recursive=True):
            val = x_trans_dict[key]
            if self[key] is not None and hasattr(self[key], "inverse"):
                x_out[key] = self[key].inverse(val)
            else:
                x_out[key] = val

        return x_out

    def reset_parameters(self):
        reset_transform_params(self.transform)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(transform={self.transform})"


class MultiplyTransform(torch.nn.Module):
    def __init__(self, factor=1.):
        super().__init__()
        self.factor = torch.tensor(factor)

    def forward(self, x):
        return x * self.factor.to(x.device)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(factor={self.factor})"

class NormalRandomHorizontalFlip(RandomHorizontalFlip):
    def forward(self, img):
        """
        Flips the normals accordingly
        """
        if torch.rand(1) < self.p:
            img = F.hflip(img)
            img[..., 0, :, :] = -img[..., 0, :, :]
            return img
        return img
    

class Clamp(torch.nn.Module):
    def __init__(self, min=None, max=None):
        super().__init__()
        self.min = min
        self.max = max

    def forward(self, x) -> torch.Tensor:
        """
        Transforms the range of tensor.
        :param x: The input tensor
        :return: The transformed tensor
        """
        return torch.clamp(x, min=self.min, max=self.max)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(min={self.min}, max={self.max}, linear={self.linear})"


def reset_transform_params(transform):
    if isinstance(transform, MutableMapping):
        reset_transform_params(list(transform.values()))
    elif isinstance(transform, Iterable):
        for t in transform:
            reset_transform_params(t)
    elif isinstance(transform, Compose):
        reset_transform_params(transform.transforms)
    elif hasattr(transform, "reset_parameters"):
        transform.reset_parameters()