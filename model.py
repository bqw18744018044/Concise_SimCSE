# encoding: utf-8
"""
@author: bqw
@time: 2021/7/20 23:03
@file: model.py
@desc: 
"""
import torch
import torch.nn as nn

from config import data_args
from utils import PairDataset, DataCollator, tokenizer
from abc import ABC
from torch.utils.data import DataLoader
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertModel
from transformers.modeling_outputs import SequenceClassifierOutput, BaseModelOutputWithPoolingAndCrossAttentions


class MLPLayer(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__()
        self.dense = nn.Linear(input_size, output_size)
        self.activation = nn.Tanh()

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = self.activation(x)
        return x


class Similarity(nn.Module):
    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp


class BertForCL(BertPreTrainedModel, ABC):
    def __init__(self, config):
        super().__init__(config)
        self.bert = BertModel(config)
        self.mlp = MLPLayer(config.hidden_size, config.hidden_size)
        self.sim = Similarity(temp=0.05)

    def forward(self,
                input_ids=None,
                attention_mask=None,
                token_type_ids=None,
                position_ids=None,
                head_mask=None,
                inputs_embeds=None,
                labels=None,
                output_attentions=None,
                output_hidden_states=None,
                return_dict=None,
                sent_emb=False):
        if sent_emb:
            return self.sentemb_forward(input_ids=input_ids,
                                        attention_mask=attention_mask,
                                        token_type_ids=token_type_ids,
                                        position_ids=position_ids,
                                        head_mask=head_mask,
                                        inputs_embeds=inputs_embeds,
                                        labels=labels,
                                        output_attentions=output_attentions,
                                        output_hidden_states=output_hidden_states,
                                        return_dict=return_dict)
        else:
            return self.cl_forward(input_ids=input_ids,
                                   attention_mask=attention_mask,
                                   token_type_ids=token_type_ids,
                                   position_ids=position_ids,
                                   head_mask=head_mask,
                                   inputs_embeds=inputs_embeds,
                                   labels=labels,
                                   output_attentions=output_attentions,
                                   output_hidden_states=output_hidden_states,
                                   return_dict=return_dict)

    def sentemb_forward(self,
                        input_ids=None,
                        attention_mask=None,
                        token_type_ids=None,
                        position_ids=None,
                        head_mask=None,
                        inputs_embeds=None,
                        labels=None,
                        output_attentions=None,
                        output_hidden_states=None,
                        return_dict=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask, return_dict=True)
        cls_output = outputs.last_hidden_state[:, 0]
        cls_output = self.mlp(cls_output)
        if not return_dict:
            return (outputs[0], cls_output) + outputs[2:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            pooler_output=cls_output,
            last_hidden_state=outputs.last_hidden_state,
            hidden_states=outputs.hidden_states,
        )

    def cl_forward(self,
                   input_ids=None,
                   attention_mask=None,
                   token_type_ids=None,
                   position_ids=None,
                   head_mask=None,
                   inputs_embeds=None,
                   labels=None,
                   output_attentions=None,
                   output_hidden_states=None,
                   return_dict=None):
        # input_ids: batch_size, num_sent, len
        batch_size = input_ids.size(0)
        num_sent = input_ids.size(1)  # 2
        # input_ids: batch_size * num_sent, len
        input_ids = input_ids.view((-1, input_ids.size(-1)))
        attention_mask = attention_mask.view((-1, attention_mask.size(-1)))
        outputs = self.bert(input_ids, attention_mask=attention_mask, return_dict=True)
        cls_output = outputs.last_hidden_state[:, 0]
        cls_output = cls_output.view((batch_size, num_sent, cls_output.size(-1)))
        # batch_size, num_sent, 768
        cls_output = self.mlp(cls_output)
        z1, z2 = cls_output[:, 0], cls_output[:, 1]
        cos_sim = self.sim(z1.unsqueeze(1), z2.unsqueeze(0))
        # [0,1,...,batch_size-1]
        labels = torch.arange(cos_sim.size(0)).long().to(self.device)
        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(cos_sim, labels)

        if not return_dict:
            output = (cos_sim,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=cos_sim,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


if __name__ == "__main__":
    with open(data_args.train_file, encoding="utf8") as file:
        texts = [line.strip() for line in file.readlines()]
    dataset = PairDataset(texts)
    print(dataset[0])
    dataloader = DataLoader(dataset, batch_size=4, collate_fn=DataCollator(tokenizer))
    batch = next(iter(dataloader))
    model = BertForCL.from_pretrained(data_args.model_name_or_path)
    cl_out = model(**batch)
    print("aaa")