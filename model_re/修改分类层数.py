#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/13 17:25
# @Author  : fan


import torch
import torch.nn as nn
import torch.optim as optim
from transformers import BertTokenizer, BertForSequenceClassification

# 加载预训练模型和分词器
pretrained_model_path = r'D:\A_TheStoragePool\MyDirection\KnowledgeGraph\MyCode\KgExtract\fanKGEX\model\re\medical_re\model_re.pkl'
tokenizer = BertTokenizer.from_pretrained(pretrained_model_path)
pretrained_model = BertForSequenceClassification.from_pretrained(
    pretrained_model_path, num_labels=23)

# 冻结预训练模型的参数
for param in pretrained_model.parameters():
    param.requires_grad = False

# print(pretrained_model.classifier)
# exit()
# 修改分类器的输出层
pretrained_model.classifier = nn.Linear(pretrained_model.config.hidden_size, 4)

# 定义训练参数
lr = 1e-5
batch_size = 32
num_epochs = 10

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(pretrained_model.classifier.parameters(), lr=lr)

# 加载训练和测试数据
train_data = ...  # 自己准备
test_data = ...   # 自己准备

# 训练模型
for epoch in range(num_epochs):
    total_loss = 0
    for i in range(0, len(train_data), batch_size):
        batch_data = train_data[i:i+batch_size]
        inputs = tokenizer(batch_data['text'], padding=True, truncation=True, return_tensors='pt')
        labels = torch.tensor(batch_data['labels'])

        # 前向传播
        outputs = pretrained_model(inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=labels)
        loss = outputs.loss

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / (len(train_data) // batch_size)
    print(f'Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}')

# 在测试集上评估模型
correct = 0
total = 0
with torch.no_grad():
    for i in range(0, len(test_data), batch_size):
        batch_data = test_data[i:i+batch_size]
        inputs = tokenizer(batch_data['text'], padding=True, truncation=True, return_tensors='pt')
        labels = torch.tensor(batch_data['labels'])

        # 前向传播
        outputs = pretrained_model(inputs['input_ids'], attention_mask=inputs['attention_mask'], labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=1)

        # 统计正确率
        total += labels.size(0)
        correct += (predictions == labels).sum().item()

accuracy = correct / total
print(f'Test Accuracy: {accuracy:.4f}')