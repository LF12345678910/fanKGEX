#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/13 16:10
# @Author  : fan

"""
以下是使用BERT预训练模型进行知识抽取任务的代码示例：
首先加载了BERT预训练模型和分词器。
然后，我们定义了一个输入文本，并使用分词器对其进行分词和编码。
接着，我们将编码后的输入传递给模型进行预测，并使用argmax函数确定最有可能的标签。最后，我们将预测结果解码为标签，并输出结果。
请注意，这只是一个简单的示例代码，实际应用中需要根据具体任务对模型进行微调和优化。
"""


import torch
from transformers import BertTokenizer, BertForTokenClassification

# 加载BERT预训练模型和分词器
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertForTokenClassification.from_pretrained('bert-base-uncased', num_labels=2)

# 输入文本
text = "Barack Obama was born in Hawaii."

# 对文本进行分词和编码
tokens = tokenizer.encode(text, add_special_tokens=True)
input_ids = torch.tensor([tokens])

# 对输入进行预测
outputs = model(input_ids)
predictions = torch.argmax(outputs.logits, dim=-1)

# 解码预测结果
predicted_tags = [model.config.id2label[tag_id] for tag_id in predictions[0].tolist()]

# 输出结果
print(predicted_tags)
