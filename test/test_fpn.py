import os
import sys

# Append Kindler root directory to sys
test_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.abspath(os.path.join(test_dir, os.pardir))
sys.path.append(root_dir)

import pytest

import numpy as np
import torch

from kindler.backbone import Backbone
from kindler.fpn import FPN


@pytest.mark.parametrize(
    'type, last_conv, feature_size, min_level, max_level, input_size',[
        ('resnet50', 5, 128, 2, 6,  64),
        ('resnet18', 4, 256, 2, 7, 128),
        ('resnet50', 4,  64, 4, 5,  32),
    ]
)
def test_fpn(type, last_conv, feature_size, min_level, max_level, input_size):
    """
    Ensure that FPN is capable of building and producing results of the
    correct shapes
    """
    # Generate dummy inputs
    dummy_input = np.random.randn(1, 3, input_size, input_size).astype('float32')
    dummy_input = torch.Tensor(dummy_input)

    # Create backbone model
    backbone = Backbone(TYPE=type, LAST_CONV=last_conv)

    # Get backbone channel sizes
    raw_features = backbone(dummy_input)
    backbone_channel_sizes = [raw_features[i].shape[1] for i in range(2, last_conv + 1)]

    # Create fpn model
    fpn = FPN(
        BACKBONE_CHANNEL_SIZES=backbone_channel_sizes,
        FEATURE_SIZE=feature_size,
        MIN_LEVEL=min_level,
        MAX_LEVEL=max_level
    )

    assert fpn.config.MAX_INPUT_LEVEL == last_conv, \
        'FPN not calculating max input level correctly'

    # Ensure that features generated are of right shapes
    features = fpn(raw_features)

    assert max(features.keys()) == max_level, \
        'FPN not generating feature levels correctly'
    assert min(features.keys()) == min_level, \
        'FPN not generating feature levels correctly'

    for i in range(min_level, max_level + 1):
        feature_shape = features[i].shape

        assert feature_shape[1] == feature_size, \
            'FPN not generating features of correct size'

        assert feature_shape[2] == feature_shape[3], \
            'FPN generating features of uneven shapes'

        assert feature_shape[2] == input_size / 2 ** i, \
            'FPN generating features of incorrect shapes'

    del backbone, fpn
    del dummy_input, raw_features, backbone_channel_sizes, features
