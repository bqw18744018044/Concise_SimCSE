# encoding: utf-8
"""
@author: bqw
@time: 2021/7/20 23:27
@file: train.py
@desc: 
"""
import transformers

from typing import List
from utils import PairDataset, DataCollator, tokenizer
from model import BertForCL
from config import data_args, OurTrainingArguments
from transformers import TrainingArguments, Trainer

transformers.utils.logging.set_verbosity_info()
transformers.utils.logging.enable_default_handler()
transformers.utils.logging.enable_explicit_format()


def run(examples: List[str], our_training_args: OurTrainingArguments):
    train_dataset = PairDataset(examples)
    collate_fn = DataCollator(tokenizer)
    training_args = TrainingArguments(
        output_dir=our_training_args.output_dir,
        num_train_epochs=our_training_args.epochs,
        per_device_train_batch_size=our_training_args.train_batch_size,
        learning_rate=our_training_args.learning_rate,
        evaluation_strategy=our_training_args.evaluation_strategy,
        eval_steps=our_training_args.eval_steps,
        load_best_model_at_end=our_training_args.load_best_model_at_end,
        overwrite_output_dir=our_training_args.overwrite_output_dir,
        do_train=our_training_args.do_train,
        do_eval=our_training_args.do_eval
    )
    model = BertForCL.from_pretrained(data_args.model_name_or_path)
    model.resize_token_embeddings(len(tokenizer))
    trainer = Trainer(model=model,
                      train_dataset=train_dataset,
                      args=training_args,
                      tokenizer=tokenizer,
                      data_collator=collate_fn)
    trainer.train()
    trainer.save_model("models/test")


if __name__ == "__main__":
    with open(data_args.train_file, encoding="utf8") as file:
        texts = [line.strip() for line in file.readlines()]
    our_training_args = OurTrainingArguments()
    run(texts, our_training_args)