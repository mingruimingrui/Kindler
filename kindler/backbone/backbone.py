import re

import torch
import torchvision

from .config import config, validate_config


class Backbone(torch.nn.Module):
    def __init__(self, config_file=None, **kwargs):
        """
        Creates a backbone model
        """
        super(Backbone, self).__init__()
        self.config = config.make_config(config_file, validate_config, **kwargs)

        if 'resnet' in self.config.TYPE:
            self._make_resnet_modules()

        if self.config.FREEZE_AT > 0:
            self._freeze_backbone()

        if self.config.FREEZE_BN:
            self._freeze_batchnorm()

        if self.config.USE_GN:
            self._use_groupnorm()

    def forward(self, x):
        features = {}
        for i, layer in enumerate(self.layers, 1):
            x = layer(x)
            if i >= 2:
                features[i] = x
        return features

    def _make_resnet_modules(self):
        resnet_model_fn = getattr(torchvision.models, self.config.TYPE)
        resnet_model = resnet_model_fn(pretrained=self.config.PRETRAINED)
        layers = []

        # C1 layer
        layers.append(torch.nn.Sequential(
            resnet_model.conv1,
            resnet_model.bn1,
            resnet_model.relu
        ))

        # C2 layer
        if self.config.LAST_CONV >= 2:
            layers.append(torch.nn.Sequential(
                resnet_model.maxpool,
                resnet_model.layer1
            ))

        # C3 layer
        if self.config.LAST_CONV >= 3:
            layers.append(resnet_model.layer2)

        # C4 layer
        if self.config.LAST_CONV >= 4:
            layers.append(resnet_model.layer3)

        # C5 layer
        if self.config.LAST_CONV >= 5:
            layers.append(resnet_model.layer4)

        self.layers = torch.nn.ModuleList(layers)
        del resnet_model

    def _freeze_backbone(self):
        for i, layer in enumerate(self.layers, 1):
            if self.config.FREEZE_AT < i:
                continue
            for param in layer.parameters():
                param.requires_grad = False

    def _freeze_batchnorm(self):
        for layer in self.modules():
            if not isinstance(layer, torch.nn.BatchNorm2d):
                continue
            for param in layer.parameters():
                param.requires_grad = False

    def _use_groupnorm(self):
        """
        Swap all batch norm layers for group norm layers
        Group norm layers will not be frozen
        """
        errmsg = 'Channel cannot be divided by group number at {} ' + \
            'Got channel_size of {} and num_group of {}'

        for name, layer in self.named_modules():
            if not isinstance(layer, torch.nn.BatchNorm2d):
                continue

            channel_size = len(layer.weight)
            assert channel_size % self.config.GN_NUM_GROUPS == 0, \
                errmsg.format(name, channel_size, self.config.GN_NUM_GROUPS)

            path = [(int(n) if n.isdigit() else n) for n in name.split('.')]
            layer_addr = self
            for p in path:
                if isinstance(p, int):
                    layer_addr = layer_addr[p]
                elif isinstance(p, str):
                    layer_addr = getattr(layer_addr, p)

            layer_addr = torch.nn.GroupNorm(self.config.GN_NUM_GROUPS, channel_size)
