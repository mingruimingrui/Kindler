"""
Backbone configs
"""
import logging
from ..utils.config_system import ConfigSystem

logger = logging.getLogger(__name__)

_C = ConfigSystem()
config = _C

valid_backbone_types = {
    'resnet18',
    'resnet34',
    'resnet50',
    'resnet101',
    'resnet152'
}

backbone_type_to_channel_sizes = {
    'resnet18': [64, 128,  256,  512],
    'resnet34': [64, 128,  256,  512],
    'resnet50': [256, 512, 1024, 2048],
    'resnet101': [256, 512, 1024, 2048],
    'resnet152': [256, 512, 1024, 2048]
}


# --------------------------------------------------------------------------- #
# Backbone options
# --------------------------------------------------------------------------- #

# The type of backbone to use
# For valid backbone types refer to above
_C.TYPE = 'resnet50'

# Use pretrained imagenet weights in backbone?
_C.PRETRAINED = True

# Last conv layer to output (based on number of pooling)
# eg. 5 will mean the last layer of conv for resnet
_C.LAST_CONV = 5

# Layers below this will be frozen
_C.FREEZE_AT = 0

# Freeze the batch norm layers?
_C.FREEZE_BN = True

# Use group normalization?
_C.USE_GN = False

# The number of groups to use for group normalization
_C.GN_NUM_GROUPS = 32

# --------------------------------------------------------------------------- #
# End of options
# --------------------------------------------------------------------------- #
_C.immutable(True)


def validate_config(config):
    """
    Check validity of configs
    """
    assert config.TYPE in valid_backbone_types, \
        '{} is invalid backbone type'.format(config.TYPE)

    assert config.LAST_CONV in {2, 3, 4, 5}, \
        'Currently only [2, 3, 4, 5] are accepted values for config.BACKBONE.LAST_CONV'

    assert config.FREEZE_AT in {0, 1, 2, 3, 4, 5}, \
        'FREEZE_AT must be a one of [0, 1, 2, 3, 4, 5]'

    if config.FREEZE_BN and config.USE_GN:
        logger.warn('Normalization layers will not be frozen if using group normalization')
