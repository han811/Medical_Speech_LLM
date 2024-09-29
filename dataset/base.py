from abc import abstractmethod

from torch.utils.data import Dataset


class BaseDatasetFactory:
    def __init__(self):
        pass

    @abstractmethod
    def get_train_dataset(self):
        raise NotImplementedError

    @abstractmethod
    def get_val_dataset(self):
        raise NotImplementedError


class BaseDataset(Dataset):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def __len__(self):
        raise NotImplementedError

    @abstractmethod
    def __getitem__(self, index):
        raise NotImplementedError
