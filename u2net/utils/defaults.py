# Coding by SunWoo(tjsntjsn20@gmail.com)

import yaml
import torch
from omegaconf import DictConfig, OmegaConf


def load_yaml(config_path: str) -> dict:
    with open(config_path, encoding='utf-8') as file:
        configs = yaml.load(file, Loader=yaml.FullLoader)
    return configs


def merge_configs(personal_config: dict, default_config: DictConfig) -> DictConfig:
    configs = OmegaConf.merge(default_config, personal_config)
    return configs


def setup_configs(configs: DictConfig) -> DictConfig:
    configs["MODEL"]["DEVICE"] = 'cuda' if torch.cuda.is_available() else 'cpu'
    return configs
