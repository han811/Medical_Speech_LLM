from typing import *

PRE_PROCESSORS = {}


def make(id: str):
    cls = PRE_PROCESSORS[id]
    return cls


def register(id: str):
    def _register(cls):
        PRE_PROCESSORS[id] = cls
        return cls

    return _register
