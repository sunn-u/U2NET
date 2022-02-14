# Coding by SunWoo(tjsntjsn20@gmail.com)

import time
import datetime
import logging
from fvcore.common.timer import Timer

from u2net.engine.train_loop import HookBase


__all__ = [
    "EpochTimer",
    "PeriodicCheckpointer",
    "EvalHook",
    "CommonWriter"
]

class EpochTimer(HookBase):
    '''
        - Track the time spent for each epoch.
        - Print a summary in the end of training.

        This hook uses the time between the call to its :meth:'before_step'
        and :meth:'aftet_step' methods.
        Under the convention that :meth:'before_step' of all hooks shoud only
        take negligible amount of time, the :class:'EpochTimer' hook should be
        places at the beginning of the list of hooks to obtain accurate timing.
    '''

    def __init__(self):
        self._start_time = time.perf_counter()
        self._step_timer = Timer()

    def before_train(self):
        self._start_time = time.perf_counter()

    def before_step(self):
        self._step_timer.reset()

    def after_step(self):
        step_secs = self._step_timer.seconds()
        self.trainer.storage.put_scalars(time=step_secs)

    def after_train(self):
        total_time = time.perf_counter() - self._start_time
        hook_time = sum([info['time'] for info in self.trainer.storage.history.values()])/self.trainer.max_epoch

        logger = logging.getLogger(__name__)
        logger.info(
            f"Total training time: {str(datetime.timedelta(seconds=int(total_time)))} "
            f"({str(datetime.timedelta(seconds=int(hook_time)))} on hooks)"
        )


class PeriodicCheckpointer(HookBase):
    '''
        - Load the model file or last-checkpoint.
        - Save the model weights in output directory.
    '''

    def __init__(self, checkpointer, checkpoint_period: int):
        self.checkpointer = checkpointer
        self.checkpoint_period = checkpoint_period

    def before_train(self):
        self.checkpointer.load(model=self.trainer._trainer.model)

    def after_step(self):
        if (self.trainer.epoch + 1) % self.checkpoint_period == 0:
            self.checkpointer.save(
                model=self.trainer._trainer.model,
                epoch=self.trainer.epoch,
                score=self.trainer.storage.history[self.trainer.epoch]["scores"]
            )


class EvalHook(HookBase):
    '''
        - Run an evaluation function periodically.
    '''

    def __init__(self, eval_function, eval_period: int):
        self.eval = eval_function
        self.eval_period = eval_period

    def after_step(self):
        if (self.trainer.epoch + 1) % self.eval_period == 0:
            eval_score = self.eval(model=self.trainer._trainer.model)
            self.trainer.storage.put_scalars(scores=eval_score)

    def after_train(self):
        eval_score = self.eval(model=self.trainer._trainer.model)
        self.trainer.storage.put_scalars(scores=eval_score)


class CommonWriter(HookBase):
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def before_train(self):
        self.logger.info("Start Training!")

    def after_step(self):
        storage = self.trainer.storage.history[self.trainer.epoch]

        logs = f"[{self.trainer.epoch}/{self.trainer.max_epoch}]: "
        for key, value in storage.items():
            logs += f"{key}: {value} "
        self.logger.info(logs)

    def after_train(self):
        self.logger.info("Training is over. Well done!")
