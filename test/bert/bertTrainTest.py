#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/28 14:44
# @Author  : fan
"""
该示例中，首先定义了训练和测试数据集，并初始化了BERT模型和分词器。
然后，通过分词器将文本转换为模型所需的输入格式，并构建数据集和数据加载器。
接下来，定义了优化器和损失函数，并进行模型训练。在训练过程中，通过每个batch的输入数据计算损失，并进行反向传播和权重更新。
最后，进行模型测试，并输出测试结果。
"""

import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, RandomSampler
from torch.utils.data import TensorDataset

# 定义数据集
sentences = ['This is a positive sentence.', 'This is a negative sentence.']
labels = [1, 0]

# 初始化BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)

# 将文本转换为模型所需的输入格式
inputs = tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
labels = torch.tensor(labels)

# 构建数据集并进行数据加载
dataset = TensorDataset(inputs['input_ids'], inputs['attention_mask'], labels)
sampler = RandomSampler(dataset)
dataloader = DataLoader(dataset, sampler=sampler, batch_size=2)

# 定义优化器和损失函数
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
loss_function = torch.nn.CrossEntropyLoss()

# 进行模型训练
model.train()
for epoch in range(3):
    for batch in dataloader:
        batch_input_ids, batch_attention_mask, batch_labels = batch
        optimizer.zero_grad()
        outputs = model(batch_input_ids, attention_mask=batch_attention_mask, labels=batch_labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# 进行模型测试
model.eval()
with torch.no_grad():
    test_input = tokenizer('This is a test sentence.', padding=True, truncation=True, return_tensors='pt')
    test_output = model(test_input['input_ids'], attention_mask=test_input['attention_mask'])
    predicted_label = torch.argmax(test_output.logits[0]).item()

print('The predicted label is:', predicted_label)
