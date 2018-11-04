import os
import sys

# Append Kindler root directory to sys
test_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.abspath(os.path.join(test_dir, os.pardir))
sys.path.append(root_dir)

import yaml
import logging
import argparse

import torch

from kindler.utils.logger import setup_logging

logger = logging.getLogger(__name__)


def makedirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def parse_args(args):
    parser = argparse.ArgumentParser('Trainer for detection model')

    parser.add_argument('-y', '--yaml-file', type=str,
        help='Path to yaml file containing script configs')

    parser.add_argument('--model-config-file', type=str,
        help='Path to model config file')
    parser.add_argument('--ann-files', type=str, nargs='+',
        help='Path to annotation files multiple files can be accepted')
    parser.add_argument('--root-img-dirs', type=str, nargs='+',
        help='Path to image directories, should have same entries as ann files')

    parser.add_argument('--batch-size', type=int, default=1,
        help='Size of batches during training')
    parser.add_argument('--max-iter', type=int, default=1440000,
        help='Maximum number of iterations to perform during training')
    parser.add_argument('--lr', type=float, default=0.001,
        help='Learning rate to use during training')

    parser.add_argument('--log-dir', type=str, default='./',
        help='Directory to store log files')
    parser.add_argument('--checkpoint-dir', type=str, default='./',
        help='Directory to store checkpoint files')

    parser.add_argument('--local_rank', type=int, default=0)

    return parser.parse_args(args)


def config_args(args):
    """
    Does a number of things
    - Ensure that args are valid
    - Create directory and files
    - Set up logging
    """
    if args.yaml_file is not None:
        with open(args.yaml_file, 'r') as f:
            yaml_configs = yaml.load(f)
        for key, value in yaml_configs.items():
            if not hasattr(args, key):
                continue
            setattr(args, key, value)

    if isinstance(args.ann_files, str):
        args.ann_files = [args.ann_files]
    if isinstance(args.root_img_dirs, str):
        args.root_img_dirs = [args.root_img_dirs]

    assert len(args.ann_files) == len(args.root_img_dirs)

    assert args.batch_size > 0
    assert args.max_iter > 0
    assert args.lr > 0

    makedirs(args.log_dir)
    makedirs(args.checkpoint_dir)
    setup_logging(os.path.join(args.log_dir, 'train.log'))

    return args


def main(args):
    model = 


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    args = config_args(args)
    main(args)
