# encoding: utf-8
"""
@author: bqw
@time: 2021/7/20 21:51
@file: config.py
@desc: 
"""
from dataclasses import dataclass, field


@dataclass
class DataArguments:
    train_file: str = field(default="./data/simcse/wiki1m_for_simcse.txt",
                            metadata={"help": "The path of train file"})
    model_name_or_path: str = field(default="E:/pretrained/bert-base-uncased",
                                    metadata={"help": "The name or path of pre-trained language model"})
    max_seq_length: int = field(default=32,
                                metadata={"help": "The maximum total input sequence length after tokenization."})


@dataclass
class OurTrainingArguments:
    output_dir: str = field(default="./checkpoints")
    epochs: int = field(default=1)
    train_batch_size: int = field(default=64)
    learning_rate: float = field(default=3e-5)
    load_best_model_at_end: bool = field(default=True)
    overwrite_output_dir: bool = field(default=True)
    do_train: bool = field(default=True)
    do_eval: bool = field(default=False)
    logging_step: int = field(default=100)


data_args = DataArguments()
