# encoding: utf-8
"""
@author: bqw
@time: 2021/7/20 21:53
@file: utils.py
@desc: 
"""
import torch

from config import data_args
from dataclasses import dataclass
from transformers import BertTokenizer
from typing import List, Union, Optional, Dict
from torch.utils.data import Dataset, DataLoader
from transformers.tokenization_utils_base import PaddingStrategy, PreTrainedTokenizerBase

tokenizer = BertTokenizer.from_pretrained(data_args.model_name_or_path)


class PairDataset(Dataset):
    def __init__(self, examples: List[str]):
        total = len(examples)
        sentences_pair = examples + examples
        sent_features = tokenizer(sentences_pair,
                                  max_length=data_args.max_seq_length,
                                  truncation=True,
                                  padding=False)
        features = {}
        for key in sent_features:
            features[key] = [[sent_features[key][i], sent_features[key][i + total]] for i in range(total)]
        self.input_ids = features["input_ids"]
        self.attention_mask = features["attention_mask"]
        self.token_type_ids = features["token_type_ids"]

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, item):
        return {
            "input_ids": self.input_ids[item],
            "attention_mask": self.attention_mask[item],
            "token_type_ids": self.token_type_ids[item]
        }


@dataclass
class DataCollator:
    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None

    def __call__(self, features: List[Dict[str, List[int]]]) -> Dict[str, torch.Tensor]:
        special_keys = ['input_ids', 'attention_mask', 'token_type_ids']
        batch_size = len(features)
        if batch_size == 0:
            return
        # flat_features: [sen1, sen1, sen2, sen2, ...]
        flat_features = []
        for feature in features:
            for i in range(2):
                flat_features.append({k: feature[k][i] for k in feature.keys() if k in special_keys})
        # padding
        batch = self.tokenizer.pad(
            flat_features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        # batch_size, 2, seq_len
        batch = {k: batch[k].view(batch_size, 2, -1) for k in batch if k in special_keys}
        return batch


if __name__ == "__main__":
    with open(data_args.train_file, encoding="utf8") as file:
        texts = [line.strip() for line in file.readlines()]
    dataset = PairDataset(texts)
    print(dataset[0])
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=DataCollator(tokenizer))
    batch = next(iter(dataloader))
    print(batch.keys())
    print(batch["input_ids"].shape)