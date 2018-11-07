import os
import sys

# Append Kindler root directory to sys
this_dir = os.path.dirname(os.path.realpath(__file__))
root_dir = os.path.abspath(os.path.join(this_dir, os.pardir))
sys.path.append(root_dir)

import json
import yaml
import logging
import argparse

import torch

# Tester
from kindler.engine import do_coco_test

# Env and logging
from kindler.utils.logger import setup_logging
from kindler.utils.collect_env import collect_env_info

logger = logging.getLogger(__name__)


def makedirs(path):
    if not os.path.isdir(path):
        os.makedirs(path)


def parse_args(args):
    parser = argparse.ArgumentParser('Tester for detection model on the coco dataset')

    parser.add_argument('-y', '--yaml-file', type=str,
        help='Path to yaml file containing script configs')

    parser.add_argument('--type', type=str, default='bbox',
        help='Type of evaluation to perform')

    parser.add_argument('--model-pth-file', type=str,
        help='Path to model compiled pth file')
    parser.add_argument('--model-tar-file', type=str,
        help='Path to model training tar file')
    parser.add_argument('--ann-file', type=str,
        help='Path to annotation file. Should be only a single file')
    parser.add_argument('--root-img-dir', type=str,
        help='Path to image directory. Should be only a single file')

    parser.add_argument('--max-vis', type=int, default=100,
        help='Maximum number of images to visualize')

    parser.add_argument('--log-dir', type=str, default='./',
        help='Directory to store log files')
    parser.add_argument('--vis-dir', type=str, default='./images',
        help='Directory to store result visualizations')

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
            assert hasattr(args, key), \
                '{} is an invalid option'.format(key)
            setattr(args, key, value)

    assert args.type in {'segm', 'bbox', 'keypoints'}
    assert args.model_pth_file is not None or args.model_tar_file is not None, \
        'Either one of model pth file or model tar file has to be provided'

    makedirs(args.log_dir)
    makedirs(args.vis_dir)

    return args


def load_model(args):
    if args.model_pth_file is not None:
        if args.model_tar_file is not None:
            logger.warn('model_pth_file is already provided, model_tar_file will be ignored')
        model = torch.load(args.model_pth_file)
    else:
        checkpoint = torch.load(args.model_tar_file)

        # Load retinanet
        from kindler.retinanet import RetinaNet
        model = RetinaNet(**checkpoint['config'])

        model.load_state_dict(checkpoint['model'])

    return model.eval().cuda()


def main(args):
    torch.cuda.set_device(0)

    setup_logging(os.path.join(args.log_dir, 'test.log'))

    # Load model
    model = load_model(args)
    model_config = model.config

    logger.info('Collecting env info')
    logger.info('\n' + collect_env_info() + '\n')

    logger.info('Testing config: {}\n'.format(
        json.dumps(vars(args), indent=2)
    ))

    logger.info('Model config: {}\n'.format(
        json.dumps(dict(model_config), indent=2)
    ))

    do_coco_test(
        model=model,
        ann_file=args.ann_file,
        root_img_dir=args.root_img_dir,
        max_vis=args.max_vis,
        vis_dir=args.vis_dir,
        test_type=args.type,
    )


if __name__ == '__main__':
    args = parse_args(sys.argv[1:])
    args = config_args(args)
    main(args)
