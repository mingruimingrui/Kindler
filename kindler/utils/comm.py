# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
This file contains primitives for multi-gpu communication.
This is useful when doing distributed training.
"""

import os
import pickle
import tempfile
import time

import torch


def get_world_size():
    if not torch.distributed.is_initialized():
        return 1
    return torch.distributed.get_world_size()


def get_rank():
    if not torch.distributed.is_initialized():
        return 0
    return torch.distributed.get_rank()


def is_main_process():
    if not torch.distributed.is_initialized():
        return True
    return torch.distributed.get_rank() == 0
