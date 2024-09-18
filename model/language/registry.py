from typing import *

LANGUAGE_MODELS = {}


def make(id: str):
    cls = LANGUAGE_MODELS[id]
    return cls


def register(id: str):
    def _register(cls):
        LANGUAGE_MODELS[id] = cls
        return cls

    return _register
