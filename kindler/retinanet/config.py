"""
Retinanet configs
"""
import logging
from ..utils.config_system import ConfigSystem
from ..backbone.config import config as backbone_config
from ..backbone.config import validate_config as validate_backbone_config
from ..backbone.config import backbone_type_to_channel_sizes
from ..fpn.config import config as fpn_config
from ..fpn.config import validate_config as validate_fpn_config

logger = logging.getLogger(__name__)

_C = ConfigSystem()
config = _C


# ---------------------------------------------------------------------------- #
# Backbone options
# ---------------------------------------------------------------------------- #
_C.BACKBONE = backbone_config.clone()
_C.BACKBONE.immutable(False)

# Refer to kindler.backbone.config for full list of configs

# ---------------------------------------------------------------------------- #
# FPN options
# ---------------------------------------------------------------------------- #
_C.FPN = fpn_config.clone()
_C.BACKBONE.immutable(False)

# Refer to kindler.fpn.config for full list of configs
# You can safely ignore BACKBONE_CHANNEL_SIZES, MIN_INPUT_LEVEL and
# MAX_INPUT_LEVEL as these are values that will be inferred from your backbone
# However do note that the FPN configs will determine the feature levels at
# which retinanet will be predicting on

# ---------------------------------------------------------------------------- #
# Classifier and regressor options
# ---------------------------------------------------------------------------- #
_C.CLASSIFIER = ConfigSystem()
_C.REGRESSOR = ConfigSystem()
_C.COMBINED = ConfigSystem()
# In the original retinanet, the classifier and regression heads are separated.
# You have the option of using a unified classification and regression model
# to reduce memory and computation requirements

# Inner features size in the classification head
_C.CLASSIFIER.FEATURE_SIZE = 256

# Number of layers of convolution layers in the classification head
_C.CLASSIFIER.NUM_LAYERS = 4

# Inner features size in the regression head
_C.REGRESSOR.FEATURE_SIZE = 256

# Number of layers of convolution layers in the regression head
_C.REGRESSOR.NUM_LAYERS = 4

# Use a unified classification and regression model?
_C.COMBINED.USE = False

# Inner features size in the combined head
_C.COMBINED.FEATURE_SIZE = 256

# Number of layers of convolution layers in the combined head
_C.COMBINED.NUM_LAYERS = 4

# ---------------------------------------------------------------------------- #
# Anchor options
# ---------------------------------------------------------------------------- #
_C.ANCHOR = ConfigSystem()

# The ratios of anchors at each moving window
_C.ANCHOR.RATIOS = [0.5, 1., 2.]

# The number of scales per feature level
_C.ANCHOR.SCALES_PER_OCTAVE = 3

# The multiplier for the anchor sizes at each level
# The anchor size at level n will be "SIZE_MULT * (2 ** n)"
_C.ANCHOR.SIZE_MULT = 4.0

# The multiplier for the anchor strides at each level
# The anchor stride at level n will be "STRIDE_MULT * (2 ** n)"
_C.ANCHOR.STRIDE_MULT = 1.0

# ---------------------------------------------------------------------------- #
# Target options
# ---------------------------------------------------------------------------- #
_C.TARGET = ConfigSystem()

# Then number of object classes to predict
# This is a required variable
_C.TARGET.NUM_CLASSES = -1

# Should there be a background class?
# By setting background predictor to be True will also affect loss computation
_C.TARGET.BACKGROUND_PREDICTOR = False

# Should each class have it's own regression model?
_C.TARGET.CLASS_SPECIFIC_BBOX = False

# Overlap threshold for classification to be positive
_C.TARGET.POSITIVE_OVERLAP = 0.5

# Overlap threshold for classification to be negative
_C.TARGET.NEGATIVE_OVERLAP = 0.4

# ---------------------------------------------------------------------------- #
# Initialization options
# ---------------------------------------------------------------------------- #
_C.INITIALIZATION = ConfigSystem()

# Initial classification output for objects
# Initial background (if background predictor is true) output will be 1 - PRIOR_PROB
_C.INITIALIZATION.PRIOR_PROB = 0.01

# ---------------------------------------------------------------------------- #
# Loss options
# ---------------------------------------------------------------------------- #
_C.LOSS = ConfigSystem()

# Use focal loss? If not use BCE loss for classification
_C.LOSS.USE_FOCAL = True

# ALpha for focal loss
_C.LOSS.FOCAL_ALPHA = 0.25

# Gamma for focal loss
_C.LOSS.FOCAL_GAMMA = 2.0

# Weight of bounding box regression
_C.LOSS.REG_WEIGHT = 1.0

# Smooth L1 loss beta for bounding box regression
_C.LOSS.REG_BETA = 0.11

# ---------------------------------------------------------------------------- #
# Output options
# ---------------------------------------------------------------------------- #
_C.OUTPUT = ConfigSystem()

# Apply nms on output? If not just returns raw output
_C.OUTPUT.APPLY_NMS = True

# Threshold to filter detections for NMS
_C.OUTPUT.NMS_THRESH = 0.3

# Threshold to determine if an area is background
# Will only be used if BACKGROUND_PREDICTOR is used
_C.OUTPUT.BACKGROUND_THRESH = 0.7

# Maximum number of detections to produce
_C.OUTPUT.NUM_DETECTIONS = 300

# ---------------------------------------------------------------------------- #
# End of options
# ---------------------------------------------------------------------------- #
_C.immutable(True)

def validate_config(config):
    """
    Check validity of configs
    """
    # Validate backbone
    validate_backbone_config(config.BACKBONE)

    # Determine backbone_channel_sizes and validate fpn
    backbone_channel_sizes = backbone_type_to_channel_sizes[config.BACKBONE.TYPE]
    backbone_channel_sizes = backbone_channel_sizes[:config.BACKBONE.LAST_CONV - 1]

    config.FPN.update({'BACKBONE_CHANNEL_SIZES': backbone_channel_sizes})
    validate_fpn_config(config.FPN)

    assert config.TARGET.NUM_CLASSES >= 1, \
        'Num classes is a required variable, it cannot be {}'.format(config.TARGET.NUM_CLASSES)

    assert 0.0 <= config.TARGET.POSITIVE_OVERLAP <= 1.0, \
        'Value of POSITIVE_OVERLAP is invalid'

    assert 0.0 <= config.TARGET.NEGATIVE_OVERLAP <= 1.0, \
        'Value of NEGATIVE_OVERLAP is invalid'

    assert 0.0 <= config.INITIALIZATION.PRIOR_PROB <= 1.0, \
        'Value of PRIOR_PROB is invalid'

    assert 0.0 <= config.OUTPUT.NMS_THRESH <= 1.0, \
        'Value of NMS_THRESH is invalid'

    assert 0.0 <= config.OUTPUT.BACKGROUND_THRESH <= 1.0, \
        'Value of BACKGROUND_THRESH is invalid'