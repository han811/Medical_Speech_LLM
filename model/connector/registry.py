from typing import *

CONNECTOR_MODELS = {}


def make(id: str):
    cls = CONNECTOR_MODELS[id]
    return cls


def register(id: str):
    def _register(cls):
        CONNECTOR_MODELS[id] = cls
        return cls

    return _register
