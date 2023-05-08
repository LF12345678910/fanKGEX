#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/28 14:40
# @Author  : fan


import torch
from transformers import BertTokenizer, BertModel

# 初始化BERT模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# 定义待抽取关系的实体对和文本
entity_1 = 'Tom Brady'
entity_2 = 'New England Patriots'
text = 'Tom Brady is a quarterback for the New England Patriots.'

# 将文本分词，并添加特殊标记
inputs = tokenizer(text, return_tensors='pt')
# 使用BERT模型进行预测
outputs = model(**inputs)

# 获取实体对在文本中对应的位置
entity_1_start, entity_1_end = text.find(entity_1), text.find(entity_1) + len(entity_1)
entity_2_start, entity_2_end = text.find(entity_2), text.find(entity_2) + len(entity_2)

# 获取实体对所对应的词向量
entity_1_embedding = outputs.last_hidden_state[0][entity_1_start:entity_1_end, :]
entity_2_embedding = outputs.last_hidden_state[0][entity_2_start:entity_2_end, :]

# 计算实体对之间的关系向量
relation_embedding = torch.mean(outputs.last_hidden_state[0][entity_1_end:entity_2_start, :], dim=0)

# 将实体对向量和关系向量拼接在一起，并进行线性变换和softmax操作
relation_input = torch.cat([entity_1_embedding, relation_embedding, entity_2_embedding], dim=0)
relation_output = torch.nn.functional.softmax(model.relation_classifier(relation_input))

# 输出预测结果
predicted_relation = torch.argmax(relation_output).item()
print('The predicted relation is:', predicted_relation)

'''
该示例中，首先初始化了BERT模型和分词器，然后定义了待抽取关系的实体对和文本。
接下来，通过分词器将文本转换成了词汇表中的编号，并使用BERT模型进行预测，得到了文本中每个词的词向量表示。
然后，通过实体对在文本中对应的位置，获取了实体对所对应的词向量，并计算了实体对之间的关系向量。
最后，将实体对向量和关系向量拼接在一起，并进行线性变换和softmax操作，得到了关系预测结果。
'''