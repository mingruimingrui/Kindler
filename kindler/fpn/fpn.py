import torch

from .config import config, validate_config


class FPN(torch.nn.Module):
    """
    FPN layer as defined in
    """
    def __init__(self, config_file=None, **kwargs):
        super(FPN, self).__init__()
        self.config = config.make_config(config_file, validate_config, **kwargs)
        self._make_modules()

    def forward(self, input_features):
        # Produce features in this order
        # MAX_INPUT_LEVEL
        # MAX_INPUT_LEVEL - 1
        # ...
        # MIN_LEVEL + 1
        # MIN_LEVEL
        # MAX_INPUT_LEVEL + 1
        # ...
        # MAX_LEVEL - 1
        # MAX_LEVEL
        reduced_features = {}
        output_features = {}

        for level in range(self.config.MAX_INPUT_LEVEL, self.config.MIN_LEVEL - 1, -1):
            reduced_features[level] = self.conv_reduce[str(level)](input_features[level])

            if level == self.config.MAX_INPUT_LEVEL:
                # PX = conv(CX)
                # output_feature  = conv(reduced_feature)
                output_features[level] = self.conv[str(level)](reduced_features[level])

            else:
                # PX = conv("PX+1" + CX)
                # Perform these actions inline to reduce memory cost
                output_features[level] = self.conv[str(level)](
                    torch.nn.functional.interpolate(
                        reduced_features[level + 1],
                        size=reduced_features[level].shape[-2:],
                        mode='bilinear',
                        align_corners=False
                    ) + reduced_features[level]
                )

        for level in range(self.config.MAX_INPUT_LEVEL + 1, self.config.MAX_LEVEL + 1):
            output_features[level] = self.conv[str(level)](self.relu(output_features[level - 1]))

        return output_features

    def _make_modules(self):
        output_feature_levels = range(self.config.MIN_LEVEL, self.config.MAX_LEVEL + 1)

        # conv_reduce are conv layers that reduces channel sizes all input
        # features to FEATURE_SIZE
        conv_reduce = {}

        # conv are conv layers that produces the output features of FPN
        conv = {}

        for level in output_feature_levels:
            if level > self.config.MAX_INPUT_LEVEL:
                # To produce features of levels greater than MAX_INPUT_LEVEL,
                # take the outputs of the previous layer and apply conv with
                # stride = 2 to it
                conv[str(level)] = torch.nn.Conv2d(
                    self.config.FEATURE_SIZE,
                    self.config.FEATURE_SIZE,
                    kernel_size=3,
                    stride=2,
                    padding=1
                )

            else:
                # To produces features of levels smaller or equal to MAX_INPUT_LEVEL,
                # First reduce the channel size of input_features, get the
                # outputs from a higher level and apply conv with stride = 1
                conv_reduce[str(level)] = torch.nn.Conv2d(
                    self.config.BACKBONE_CHANNEL_SIZES[level - self.config.MIN_INPUT_LEVEL],
                    self.config.FEATURE_SIZE,
                    kernel_size=1,
                    stride=1,
                    padding=0
                )

                conv[str(level)] = torch.nn.Conv2d(
                    self.config.FEATURE_SIZE,
                    self.config.FEATURE_SIZE,
                    kernel_size=3,
                    stride=1,
                    padding=1
                )

        self.conv_reduce = torch.nn.ModuleDict(conv_reduce)
        self.conv = torch.nn.ModuleDict(conv)
        self.relu = torch.nn.ReLU(inplace=False)
