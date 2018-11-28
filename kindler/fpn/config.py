"""
FPN configs
"""
import logging
from ..utils.config_system import ConfigSystem

logger = logging.getLogger(__name__)

_C = ConfigSystem()
config = _C


# --------------------------------------------------------------------------- #
# FPN options
# --------------------------------------------------------------------------- #

# The channel sizes of the input features of levels 2 to an implied
# MAX_INPUT_LEVEL
#
# If len(BACKBONE_CHANNEL_SIZES) == 4, it will be assumed that the
# MAX_INPUT_LEVEL is 5 (since the channel sizes will represent the level
# [2, 3, 4, 5] feature levels)
#
# Similarly if len(BACKBONE_CHANNEL_SIZES) == 3, the MAX_INPUT_LEVEL will be
# assumed to be 4
#
# The default values are assuming resnet50 at feature levels 2-5
_C.BACKBONE_CHANNEL_SIZES = [256, 512, 1024, 2048]

# The minimum input feature level
# Treat this as a dummy config, only accepted value is 2
_C.MIN_INPUT_LEVEL = 2

# Treat this as a dummy config, will be automatically filled based on your
# values for BACKBONE_CHANNEL_SIZES
_C.MAX_INPUT_LEVEL = None

# The finest and lowest feature level to produce
_C.MIN_LEVEL = 2

# The coarest and highest feature level to produce
_C.MAX_LEVEL = 6

# The size of features produced by the FPN layer
_C.FEATURE_SIZE = 256

# --------------------------------------------------------------------------- #
# End of options
# --------------------------------------------------------------------------- #
_C.immutable(True)


def validate_config(config):
    """
    Check validity of configs
    """
    assert len(config.BACKBONE_CHANNEL_SIZES) >= 1, \
        'There cannot be no values for BACKBONE_CHANNEL_SIZES'

    assert config.MIN_LEVEL >= 2, \
        'Minimum feature levels lower than 2 is currently not supported'

    assert config.MIN_LEVEL <= 5, \
        'Minimum feature levels greater than 5 is currently not supported'

    assert config.MIN_INPUT_LEVEL == 2, \
        'MIN_INPUT_LEVEL must be 2'

    implied_max_input_level = config.MIN_INPUT_LEVEL + len(config.BACKBONE_CHANNEL_SIZES) - 1
    if not config.MAX_INPUT_LEVEL == implied_max_input_level:
        config.update({'MAX_INPUT_LEVEL': implied_max_input_level})
        logger.info('MAX_INPUT_LEVEL automatically set to {}'.format(implied_max_input_level))
