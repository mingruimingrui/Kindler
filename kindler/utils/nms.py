# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
from .. import _C

# NMS takes Tensors and applies non max supression
nms = _C.nms
