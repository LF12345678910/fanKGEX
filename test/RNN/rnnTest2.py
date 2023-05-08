#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/28 14:15
# @Author  : fan


import torch
import torch.nn as nn

# 定义RNN模型
class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)  # 定义RNN层
        self.fc = nn.Linear(hidden_size, output_size)  # 定义全连接层

    def forward(self, x, h0):
        out, hn = self.rnn(x, h0)  # 进行RNN层的前向传播
        out = self.fc(out[:, -1, :])  # 取RNN输出序列的最后一个时间步，通过全连接层得到输出
        return out

# 定义超参数
input_size = 10
hidden_size = 20
output_size = 5
num_layers = 1
batch_size = 16
seq_len = 10
learning_rate = 0.01
num_epochs = 20

# 生成数据
x = torch.randn(batch_size, seq_len, input_size)  # 生成输入数据x
y = torch.randint(0, output_size, (batch_size,))  # 生成标签数据y
print("x",x,x.shape)
print("=================================================================")
print("y",y,y.shape)
exit()

# 初始化模型和优化器
model = RNN(input_size, hidden_size, output_size)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.CrossEntropyLoss()

# 训练模型
for epoch in range(num_epochs):
    # 清空梯度
    optimizer.zero_grad()

    # 前向传播
    h0 = torch.zeros(num_layers, batch_size, hidden_size)
    outputs = model(x, h0)

    # 计算损失
    loss = criterion(outputs, y)

    # 反向传播和优化
    loss.backward()
    optimizer.step()

    # 打印训练结果
    if (epoch+1) % 5 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
