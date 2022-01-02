# Coding by SunWoo(tjsntjsn20@gmail.com)

import os
import hydra

from u2net.data.types import DictConfigs
from u2net.utils.defaults import load_yaml, merge_configs, setup_configs, save_configs
from u2net.utils.logger import set_logger
from u2net.engine import build_trainer, build_tester


def setup(config: DictConfigs) -> DictConfigs:
    assert config["CONFIG_FILE_PATH"] is not None, \
        'Must enter the CONFIG_FILE_PATH.\n' \
        'ex. python main.py CONFIG_FILE_PATH=$yaml_file_path'

    personal_config = load_yaml(config["CONFIG_FILE_PATH"])
    config = merge_configs(personal_config, config)
    config = setup_configs(config)

    return config


@hydra.main(config_path='u2net/config', config_name='defaults')
def main(default_config: DictConfigs):
    configs = setup(config=default_config)

    # Make output directory.
    os.makedirs(configs["OUTPUT_DIR"], exist_ok=True)
    save_configs(configs, save_dir=configs["OUTPUT_DIR"])

    # Build logger.
    logger = set_logger(level=configs["SOLVER"]["LOG"]["LEVEL"], save_dir=configs["OUTPUT_DIR"])

    # Start train or infer.
    if configs["MODEL"]["TRAIN"]:
        trainer = build_trainer(configs, logger)
        trainer.train()
    else:
        testor = build_tester(configs, logger)
        testor.test()


if __name__ == '__main__':
    main()
