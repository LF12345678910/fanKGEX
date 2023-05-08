#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/30 15:35
# @Author  : fan


import random
import json
import numpy as np
import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel, AdamW
from .config import config,gpuConfig
device = gpuConfig.device


# model four s 谐音命名，model for s。s是主体
# 这个模型用来找主体，由主体出发，指向客体
class Model4s(nn.Module):
    def __init__(self, hidden_size=768):  # hidden_size隐藏记忆单元个数,是指隐藏层的维度大小
        super(Model4s, self).__init__()
        self.bert = BertModel.from_pretrained(config.PATH_BERT)  # 模型的加载路径
        for param in self.bert.parameters():  # 对bert层的参数做初始化
            param.requires_grad = True   # 是否进行Finetune
        self.dropout = nn.Dropout(p=0.2)
        self.linear = nn.Linear(in_features=hidden_size, out_features=2, bias=True)

        self.sigmoid = nn.Sigmoid()

    # 将上一层的输出作为下一层的输入，并计算下一层的输出，一直到运算到输出层为止，即模型层按顺序输出
    def forward(self, input_ids, input_mask, segment_ids, hidden_size=768):
        hidden_states = self.bert(input_ids,
                                  attention_mask=input_mask,
                                  token_type_ids=segment_ids)[0]  # (batch_size, sequence_length, hidden_size)
        # pow(x,y)：表示x的y次幂。
        output = self.sigmoid(
                 self.linear(
                 self.dropout(hidden_states)
                 )).pow(2)  # output是Tensor格式，二维数组，这里是把数组里的每个元素取平方

        # print("output",output)
        # print("outputpow",output.pow(2))
        # exit()
        return output, hidden_states


class Model4po(nn.Module):
    def __init__(self, num_p=config.num_p, hidden_size=768):
        super(Model4po, self).__init__()
        self.dropout = nn.Dropout(p=0.4)  # 在训练期间，使用伯努利分布的样本，以概率p随机归零输入张量的一些元素。
        self.linear = nn.Linear(in_features=hidden_size, out_features=num_p * 2, bias=True)
        self.sigmoid = nn.Sigmoid()

    def forward(self, hidden_states, batch_subject_ids, input_mask):
        all_s = torch.zeros((hidden_states.shape[0], hidden_states.shape[1], hidden_states.shape[2]),
                            dtype=torch.float32)

        for b in range(hidden_states.shape[0]):
            s_start = batch_subject_ids[b][0]
            s_end = batch_subject_ids[b][1]
            s = hidden_states[b][s_start] + hidden_states[b][s_end]
            cue_len = torch.sum(input_mask[b])
            all_s[b, :cue_len, :] = s
        hidden_states += all_s.to(device)

        output = self.sigmoid(self.linear(self.dropout(hidden_states))).pow(4)

        return output  # (batch_size, max_seq_len, num_p*2)


# 损失函数
def loss_fn(pred, target):
    # 二元交叉熵损失函数，多用于多分类
    loss_fct = nn.BCELoss(reduction='none').to(device)
    # loss_fct = nn.BCEWithLogitsLoss(reduction='none').to(device)
    # 交叉熵损失函数
    # loss_fct = nn.CrossEntropyLoss(reduction='none').to(device)
    # 均方误差（MSE）损失函数，用于回归
    # loss_fct = nn.MSELoss(reduction='none').to(device)
    return loss_fct(pred, target)
