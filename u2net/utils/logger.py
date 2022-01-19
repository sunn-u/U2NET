# Coding by SunWoo(tjsntjsn20@gmail.com)

import os
import logging
from logging.handlers import TimedRotatingFileHandler

from ..data.types import Logging


def set_logger(level: int, save_dir: str, logger_name=None) -> Logging:
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    formatter = logging.Formatter(
        '[%(asctime)s] [%(filename)s > %(funcName)s > %(lineno)d] : %(message)s',
        datefmt='%Y%m%d-%H-%M'
    )
    SH = logging.StreamHandler()
    SH.setFormatter(formatter)
    logger.addHandler(SH)

    log_dir = os.path.join(save_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    logger_name = 'log' if logger_name is None else logger_name
    FH = TimedRotatingFileHandler(
        filename=os.path.join(log_dir, f"{logger_name}.log"),
        when='midnight',
        encoding='utf-8'
    )
    FH.setFormatter(formatter)
    FH.suffix = '%Y%m%d'
    logger.addHandler(FH)

    return logger
