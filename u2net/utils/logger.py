# Coding by SunWoo(tjsntjsn20@gmail.com)

import os
import logging
from logging.handlers import TimedRotatingFileHandler

from ..data.types import Logging


def set_logger(level: int, save_dir: str) -> Logging:
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    formatter = logging.Formatter(
        '[%(asctime)s] [%(filename)s > %(funcName)s > %(lineno)d] : %(message)s',
        datefmt='%Y%m%d-%H-%M'
    )
    log_dir = os.path.join(save_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)

    SH = logging.StreamHandler()
    SH.setFormatter(formatter)
    logger.addHandler(SH)

    FH = TimedRotatingFileHandler(
        filename=os.path.join(log_dir, 'log.log'),
        when='midnight',
        encoding='utf-8'
    )
    FH.setFormatter(formatter)
    FH.suffix = '%Y%m%d'
    logger.addHandler(FH)

    return logger
