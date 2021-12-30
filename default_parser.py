# Coding by SunWoo(tjsntjsn20@gmail.com)

import argparse


def setting_default_parser():
    parser = argparse.ArgumentParser(description='U^2NET')

    # parser.add_argument('--config_file', type=str, help='config file path for training and inference based on yaml.')

    parser.add_argument('--image_dir', type=str, required=True, help='image data directory.')
    parser.add_argument('--mask_dir', type=str, required=True, help='gt masks for input-images.')
    parser.add_argument('--output_dir', type=str, required=True, help='for saving results.')

    return parser.parse_args()
