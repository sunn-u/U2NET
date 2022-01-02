# Coding by SunWoo(tjsntjsn20@gmail.com)

import yaml
import torch
from omegaconf import OmegaConf

from ..data.types import DictConfigs


def load_yaml(config_path: str) -> dict:
    with open(config_path, encoding='utf-8') as file:
        configs = yaml.load(file, Loader=yaml.FullLoader)
    return configs


def merge_configs(personal_config: dict, default_config: DictConfigs) -> DictConfigs:
    configs = OmegaConf.merge(default_config, personal_config)
    return configs


def setup_configs(configs: DictConfigs) -> DictConfigs:
    configs["MODEL"]["DEVICE"] = 'cuda' if torch.cuda.is_available() else 'cpu'
    return configs


def save_configs(configs: DictConfigs, save_dir: str):
    with open(f'{save_dir}/configs.yaml', 'w') as f:
        OmegaConf.save(config=configs, f=f.name)
