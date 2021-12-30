import torch

from sunn_models.models.rsu import U2NET
from sunn_models.models.base import weight_init_xavier


def build_model(cfg):
    model = U2NET(cfg.MODEL.RSU_DICT)

    # todo : 이게 맞는 흐름인지 체크?
    if cfg.MODEL.RESUME:
        model.load_state_dict(torch.load(cfg.MODEL.MODEL_DIR))
    else:
        model.apply(weight_init_xavier)

    return model
