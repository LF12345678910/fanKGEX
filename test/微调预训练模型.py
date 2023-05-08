#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/13 16:28
# @Author  : fan
"""
以下是一个使用PyTorch实现微调预训练模型的示例代码，以适应只有4种关系的知识抽取任务：
在上述代码中，我们首先加载了预训练的BERT模型和分词器，并将其用于微调任务。
然后，我们准备了训练和评估数据集，并对其进行了编码和处理。接着，我们对模型进行了微调，并使用AdamW优化器进行训练。
在每个epoch结束时，我们输出了平均损失。最后，我们对微调后的模型进行了评估，并输出了准确率。
需要注意的是，在微调时，我们将模型的输出层大小从23更改为4，以适应只有4种关系的知识抽取任务。
"""

import torch
from transformers import BertTokenizer, BertForSequenceClassification

# 加载预训练模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=4)

# 准备数据集
train_texts = ["text1", "text2", "text3", ...]
train_labels = [0, 1, 2, ...]  # 4种关系对应的标签

# 对数据进行编码和处理
train_encodings = tokenizer(train_texts, truncation=True, padding=True)
train_labels = torch.tensor(train_labels)

# 对模型进行微调
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)
train_dataset = torch.utils.data.TensorDataset(train_encodings['input_ids'], train_encodings['attention_mask'], train_labels)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model.to(device)

for epoch in range(5):
    running_loss = 0.0
    for batch in train_loader:
        inputs = {'input_ids': batch[0].to(device),
                  'attention_mask': batch[1].to(device),
                  'labels': batch[2].to(device)}
        optimizer.zero_grad()
        outputs = model(**inputs)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        running_loss += loss.item()

    print('Epoch {} loss: {:.4f}'.format(epoch+1, running_loss / len(train_loader)))

# 对微调后的模型进行评估
eval_texts = ["text4", "text5", "text6", ...]
eval_labels = [0, 1, 2, ...]

eval_encodings = tokenizer(eval_texts, truncation=True, padding=True)
eval_labels = torch.tensor(eval_labels)

eval_dataset = torch.utils.data.TensorDataset(eval_encodings['input_ids'], eval_encodings['attention_mask'], eval_labels)
eval_loader = torch.utils.data.DataLoader(eval_dataset, batch_size=16, shuffle=False)

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for batch in eval_loader:
        inputs = {'input_ids': batch[0].to(device),
                  'attention_mask': batch[1].to(device),
                  'labels': batch[2].to(device)}
        outputs = model(**inputs)
        _, predicted = torch.max(outputs.logits, 1)
        total += inputs['labels'].size(0)
        correct += (predicted == inputs['labels']).sum().item()

    print('Accuracy on evaluation set: {:.2%}'.format(correct / total))