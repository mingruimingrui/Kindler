"""
A helper class for helping to create config.py files
"""
import json
import yaml
import logging
from copy import deepcopy
from .collections import AttrDict

logger = logging.getLogger(__name__)


class ConfigSystem(AttrDict):
    """
    A dictionary like data structure for storing and loading model configs
    """
    def clone(self):
        return deepcopy(self)

    def update(self, new_config):
        """
        Similar to update in normal dict like data structures but only updates
        the keys which are already present
        """
        orig_mutability = self.is_immutable()
        self.immutable(False)
        for key, value in new_config.items():
            if not hasattr(self, key):
                logger.warn('"{}" is not a valid key, skipping'.format(key))
                continue
            if isinstance(value, dict):
                self[key].update(value)
            else:
                self[key] = value
        self.immutable(orig_mutability)

    def merge_from_file(self, file_name):
        """
        Retrieves configs form a file and updates current configs with those
        from the file

        Currently accepts json and yaml files

        Args:
            file_name: The file containing model configs. Must be of either
                json or yaml format
        """
        if file_name.endswith('.json'):
            with open(file_name, 'r') as f:
                new_config = json.load(f)
        elif file_name.endswith('.yaml'):
            with open(file_name, 'r') as f:
                new_config = yaml.load(f)
        else:
            errmsg = '{} is not an accepted file format. Must be either a .json or .yaml file'
            raise ValueError(errmsg.format(file_name))

        self.update(new_config)

    def make_config(self, config_file=None, validate_config=None, **kwargs):
        """
        Helper function to help models clone and make config
        """
        config = self.clone()

        # Retrieve configs from file
        if config_file is not None:
            config.merge_from_file(config_file)

        # Overwrite with direct options
        config.update(kwargs)

        # Validate configs
        if validate_config is not None:
            validate_config(config)

        return config
