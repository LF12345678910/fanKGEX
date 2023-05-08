#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/13 17:48
# @Author  : fan
from transformers import BertTokenizer, BertModel, AdamW
import torch


class config:
    weight_decay = 0.01
    batch_size = 1
    learning_rate = 1e-5
    # learning_rate = 2e-4
    EPOCH = 5
    max_seq_len = 128  # 一句话最长256
    num_p = 16   # 关系relation的个数,关系数量在外边定义了！！！

    PATH_SCHEMA = '../../fanKGEX/predicate.json'
    PATH_TRAIN = '../../fanKGEX/data/json/weiyaoOut.json'
    # 预训练模型
    PATH_BERT = '../../fanKGEX/model/re/medical_re'
    PATH_MODEL = '../../fanKGEX/model/re/medical_re/model_re.pkl'
    # 模型保存位置,要加上新模型的名字！！！
    PATH_SAVE = '../../fanKGEX/model/save/new_model.pkl'

    # 分词器，处理不同的语言，要加载不同语言的词向量表（语料表）这里的这个是bert提供的
    tokenizer = BertTokenizer.from_pretrained('../../fanKGEX/model/re/medical_re/' + 'vocab.txt')

    id2predicate = {}
    predicate2id = {}


class gpuConfig:
    cudaid = 0
    device = torch.device(f"cuda:{cudaid}" if torch.cuda.is_available() else "cpu")   # f就是能用来用大括号连接字符串的