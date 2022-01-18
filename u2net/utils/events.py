# Coding by SunWoo(tjsntjsn20@gmail.com)

import logging
from collections import OrderedDict, defaultdict


_CURRENT_STORAGE_STACK = []


def get_event_storage():
    return _CURRENT_STORAGE_STACK[-1]


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
        _CURRENT_STORAGE_STACK.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        assert _CURRENT_STORAGE_STACK[-1] == self
        _CURRENT_STORAGE_STACK.pop()
