
from sunn_models import default_parser
from sunn_models.config.config import get_config
from sunn_models.engine.trainer import Trainer


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
