import torch

from .config import config, validate_config
from ..backbone import Backbone
from ..fpn import FPN
from ._retinanet import (
    ClassificationHead,
    RegressionHead,
    CombinedHead,
    ComputeAnchors
)


class RetinaNet(torch.nn.Module):
    def __init__(self, config_file=None, **kwargs):
        super(RetinaNet, self).__init__()
        self.config = config.make_config(config_file, validate_config, **kwargs)
        self._make_modules()

    def forward(self, image_batch, annotations_batch=None):
        raise NotImplementedError()

    def _make_modules(self):
        self.backbone = Backbone(**self.config.BACKBONE)
        self.fpn = FPN(**self.config.FPN)

        num_anchors = len(self.config.ANCHOR.RATIOS) * self.config.ANCHOR.SCALES_PER_OCTAVE

        if self.config.COMBINED.USE:
            self.combined_head = CombinedHead(
                input_feature_size=self.config.FPN.FEATURE_SIZE,
                feature_size=self.config.COMBINED.FEATURE_SIZE,
                num_layers=self.config.COMBINED.NUM_LAYERS,
                num_anchors=num_anchors,
                num_classes=self.config.TARGET.NUM_CLASSES,
                use_class_specific_bbox=self.config.TARGET.CLASS_SPECIFIC_BBOX,
                use_background_predictor=self.config.TARGET.BACKGROUND_PREDICTOR,
                prior_prob=self.config.INITIALIZATION.PRIOR_PROB
            )
        else:
            self.classification_head = ClassificationHead(
                input_feature_size=self.config.FPN.FEATURE_SIZE,
                feature_size=self.config.CLASSIFIER.FEATURE_SIZE,
                num_layers=self.config.CLASSIFIER.NUM_LAYERS,
                num_anchors=num_anchors,
                num_classes=self.config.TARGET.NUM_CLASSES,
                use_background_predictor=self.config.TARGET.BACKGROUND_PREDICTOR,
                prior_prob=self.config.INITIALIZATION.PRIOR_PROB
            )
            self.regression_head = RegressionHead(
                input_feature_size=self.config.FPN.FEATURE_SIZE,
                feature_size=self.config.REGRESSOR.FEATURE_SIZE,
                num_layers=self.config.REGRESSOR.NUM_LAYERS,
                num_anchors=num_anchors,
                num_classes=self.config.TARGET.NUM_CLASSES,
                use_class_specific_bbox=self.config.TARGET.CLASS_SPECIFIC_BBOX,
            )
