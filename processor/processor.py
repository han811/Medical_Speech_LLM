from . import preprocessor, postprocessor


class Processor:
    def __init__(self, processor_id: str):
        self.processor_id = processor_id
        self.preprocessor = preprocessor.make(self.processor_id)
        self.postprocessor = postprocessor.make(self.processor_id)

    def preprocess(self, waveform):
        return self.preprocessor(waveform)

    def postprocess(self):
        raise NotImplementedError
