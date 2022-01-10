# Coding by SunWoo(tjsntjsn20@gmail.com)

import os
import hydra

from u2net.data.types import DictConfigs
from u2net.engine import build_trainer, build_tester


@hydra.main(config_path='configs', config_name='defaults')
def main(configs: DictConfigs) -> None:
    # Make output directory.
    os.makedirs(configs.user.training.output_dir, exist_ok=True)

    # Start train or infer.
    if configs.user.model.train:
        trainer = build_trainer(configs)
        trainer.train()
    else:
        testor = build_tester(configs)
        testor.test()


if __name__ == '__main__':
    main()
