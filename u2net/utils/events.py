# Coding by SunWoo(tjsntjsn20@gmail.com)

import datetime
import logging
from collections import OrderedDict, defaultdict


_CURRENT_STORAGE_STACK = []

class EventStorage(object):
    '''
        This class for storing and logging records.
    '''

    def __init__(self):
        self.history = defaultdict(dict)
        self.epoch = 0

    def put_scalars(self, **kwargs):
        for key, value in kwargs.items():
            self.history[self.epoch][key] = value

    def __enter__(self):
        _CURRENT_STORAGE_STACK.append(self.history[self.epoch])

    def __exit__(self):
        if len(_CURRENT_STORAGE_STACK) != 1:
            _CURRENT_STORAGE_STACK.pop(-2)


class CommonMetricWriter(object):
    def __init__(self, max_epochs: int):
        self.max_epochs = max_epochs
        self.logger = logging.getLogger(__name__)

    def write(self):
        events = _CURRENT_STORAGE_STACK[-1]
        losses = events["losses"]
        time = str(datetime.timedelta(seconds=int(events["time"])))

        self.logger.info(
            f"[{events.keys()}/{self.max_epochs}] losses: {losses} - time : {time}'"
        )
