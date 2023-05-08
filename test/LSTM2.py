#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/20 10:57
# @Author  : fan


import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer
from fanKGEX.model_re.config import config, gpuConfig

# 加载 Bert 模型和 tokenizer
bert_model = BertModel.from_pretrained(config.PATH_BERT)
tokenizer = config.tokenizer

# 定义 BERT + LSTM 模型
class MyModel(nn.Module):
    def __init__(self, bert_model, hidden_size=512, num_layers=1, num_classes=2):
        super(MyModel, self).__init__()
        self.bert = bert_model
        self.lstm = nn.LSTM(768, hidden_size, num_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size*2, num_classes)

    def forward(self, x, attention_mask):
        # 对输入进行 BERT 编码
        with torch.no_grad():
            bert_output = self.bert(x, attention_mask=attention_mask)[0]
        # 将 BERT 输出传递给 LSTM，并进行分类
        out, _ = self.lstm(bert_output)
        out = self.fc(out[:, -1, :])
        return out

# 定义模型的输入和标签
inputs = ["This is a good movie", "This is a bad movie"]
labels = torch.LongTensor([1, 0])

# 对输入进行 BERT 编码，使用 tokenizer 将输入转换成 token_ids 和 attention_mask
encoded_input = tokenizer(inputs, padding=True, truncation=True, return_tensors="pt")
input_ids, attention_mask = encoded_input["input_ids"], encoded_input["attention_mask"]

# 将输入传递给模型，计算损失并进行反向传播
model = MyModel(bert_model)
outputs = model(input_ids, attention_mask)
loss_func = nn.CrossEntropyLoss()
loss = loss_func(outputs, labels)
print(loss)
loss.backward()