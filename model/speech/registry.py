from typing import *

SPEECH_MODELS = {}


def make(id: str):
    cls = SPEECH_MODELS[id]
    return cls


def register(id: str):
    def _register(cls):
        SPEECH_MODELS[id] = cls
        return cls

    return _register
