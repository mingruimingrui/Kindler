"""
Misc helpers for various highly specific tasks
"""
import torch
from torch._six import string_classes, container_abcs


def to_device(obj, device):
    """ Deep transfer an object over to a device """
    if isinstance(obj, torch.Tensor):
        return torch.to(device)
    elif isinstance(obj, string_classes):
        return obj
    elif isinstance(obj, container_abcs.Mapping):
        return {k: to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, container_abcs.Sequence):
        return [to_device(e, device) for e in obj]
    else:
        return obj
