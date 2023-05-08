#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/20 10:55
# @Author  : fan


import torch
from torch import nn

# 定义 LSTM 模型
class MyLSTM(nn.Module):
    def __init__(self, input_size=768, hidden_size=512, num_layers=1, num_classes=2):
        super(MyLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :]) # 选择最后一个时间步的输出
        return out

# 加载 BERT 输出
bert_output = torch.randn(2, 10, 768, requires_grad=True)

# 创建新的 Tensor 对象代替原始的 Tensor 对象
inputs = bert_output.clone().detach()

# 将 Tensor 对象传递给 LSTM 模型
lstm_model = MyLSTM()
outputs = lstm_model(inputs)

# 计算梯度
labels = torch.randn(2, 2)
loss_func = nn.CrossEntropyLoss()
loss = loss_func(outputs, torch.max(labels, 1)[1])
loss.backward()
