from typing import *

POST_PROCESSORS = {}


def make(id: str):
    cls = POST_PROCESSORS[id]
    return cls


def register(id: str):
    def _register(cls):
        POST_PROCESSORS[id] = cls
        return cls

    return _register
