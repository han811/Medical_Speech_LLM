import random

import json
import gin

from ..base import DatasetFactory
from ..registry import register


@gin.configurable
@register("snu_data")
class SNUDatasetFactory(DatasetFactory):
    def __init__(self, jsonl_path: str, test_size: float = 0.2, shuffle: bool = True):
        super().__init__(test_size, shuffle)

        self.jsonl_path = jsonl_path

        self.total_data_list = read_jsonl(self.jsonl_path)
        if self.shuffle:
            random.shuffle(self.total_data_list)

    def get_dataset(self, is_train: bool = True):
        if is_train:
            return self.total_data_list[
                : int(len(self.total_data_list) * self.test_size)
            ]
        else:
            return self.total_data_list[
                int(len(self.total_data_list) * self.test_size) :
            ]


def read_jsonl(jsonl_file: str):
    def parse_dict(json_dict):
        return_dict = dict()
        return_dict["audio_path"] = json_dict["messages"][0]["audio"]
        return_dict["question"] = json_dict["messages"][0]["content"]
        return_dict["answer"] = json_dict["messages"][1]["content"]
        return return_dict

    data_list = list()
    with open(jsonl_file) as f:
        for line in f.read().splitlines():
            data_list.append(parse_dict(json.loads(line)))

    return data_list
