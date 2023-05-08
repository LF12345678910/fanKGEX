#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/28 14:16
# @Author  : fan


import torch
import torch.nn as nn

# 定义RNN模型
class RNN(nn.Module):
    # 定义RNN模型类。该模型包含输入层到隐藏层的全连接层、输入层到输出层的全连接层和LogSoftmax激活函数
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.i2h = nn.Linear(input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(input_size + hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    # 定义模型的前向传播函数，包括将输入和隐藏状态拼接起来的操作、计算新的隐藏状态、计算输出并进行LogSoftmax激活
    def forward(self, input, hidden):
        combined = torch.cat((input, hidden), 1)
        hidden = self.i2h(combined)
        output = self.i2o(combined)
        output = self.softmax(output)
        return output, hidden

# 定义超参数,包括输入特征维度、隐藏状态维度、输出类别数和学习率
input_size = 10
hidden_size = 20
output_size = 5
learning_rate = 0.01

# 初始化RNN模型、损失函数和优化器
rnn = RNN(input_size, hidden_size, output_size)
criterion = nn.NLLLoss()
optimizer = torch.optim.SGD(rnn.parameters(), lr=learning_rate)

# 生成输入数据和目标数据,包括5个随机生成的输入数据和对应的随机目标类别
inputs = [torch.randn(1, input_size) for _ in range(5)]
targets = [torch.randint(output_size, size=(1,)) for _ in range(5)]

# 开始训练模型,在每个epoch中，先初始化隐藏状态为0，然后遍历所有输入数据，依次计算输出、损失和梯度，并更新模型参数。最后输出本轮的训练损失
for i in range(10):
    loss = 0
    hidden = torch.zeros(1, hidden_size)
    for j in range(len(inputs)):
        output, hidden = rnn(inputs[j], hidden)
        loss += criterion(output, targets[j])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    print('Epoch %d, Loss: %.4f' % (i+1, loss.item()))
