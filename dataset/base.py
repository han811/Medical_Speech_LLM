from abc import abstractmethod

from .voice_llm_dataset import VoiceLLMDataset, VoiceLLMCollator


class DatasetFactory:
    def __init__(self, test_size: float = 0.2, shuffle: bool = True):
        self.test_size = test_size
        self.shuffle = shuffle

    def get_collator_fn(self, speech_preprocessor, llm_preprocessor, device="cpu"):
        return VoiceLLMCollator(speech_preprocessor, llm_preprocessor, device)

    def get_dataset(self, is_train: bool = True):
        data_list = self.get_data_list(is_train=is_train)
        return VoiceLLMDataset(data_list=data_list)

    @abstractmethod
    def get_data_list(self, is_train: bool = True):
        raise NotImplementedError
