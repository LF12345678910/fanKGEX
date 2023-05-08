#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/20 10:07
# @Author  : fan


import torch
import torch.nn as nn
from fanKGEX.model_re.model_ex import Model4s

class MyLSTM(nn.Module):
    def __init__(self):
        super(MyLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=768, hidden_size=384, num_layers=1, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(768, 10)

    def forward(self, inputs):
        # 输入数据
        x, _ = self.lstm(inputs)
        # x = x[:, -1, :]  # 取最后一个时间步的状态
        x = self.fc(x)
        return x

# 加载数据
# model4s = Model4s()
bert_output = torch.randn(1, 512, 768).cuda()  # 假设 BERT 模型的输出为 (1, 10, 768)
# bert_output = model4s()
inputs = bert_output.clone().detach()

# 实例化模型并将模型移动到 GPU 上
model = MyLSTM().cuda()

# 进行前向传播
output = model(inputs)

# 计算损失
loss = torch.nn.functional.mse_loss(output, torch.randn(1, 10).cuda())

print(loss)
# 反向传播并更新模型参数
loss.backward()