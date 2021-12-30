# Coding by SunWoo(tjsntjsn20@gmail.com)

import default_parser
from config.config import get_config
from engine.trainer import Trainer


def setup():
    cfg = get_config()

    return cfg


def main():
    cfg = setup()

    # if cfg.eval:
    #     pass
    Trainer(cfg).train()


if __name__ == '__main__':
    main()
