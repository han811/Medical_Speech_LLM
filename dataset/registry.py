from typing import *

DATASET_FACTORY = {}


def make(id: str):
    cls = DATASET_FACTORY[id]
    return cls


def register(id: str):
    def _register(cls):
        DATASET_FACTORY[id] = cls
        return cls

    return _register
