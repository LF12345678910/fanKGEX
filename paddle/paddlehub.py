# encoding: utf-8
# @author: fan
# @file: paddlehub.py
# @time: 2023/3/7 13:40

import paddle

# PaddleClas
models = paddle.hub.list('PaddlePaddle/PaddleClas:develop', source='github', force_reload=True,)
print(models)

docs = paddle.hub.help('PaddlePaddle/PaddleClas:develop', 'alexnet', source='github', force_reload=False,)
print(docs)

model = paddle.hub.load('PaddlePaddle/PaddleClas:develop', 'alexnet', source='github', force_reload=False, pretrained=True)
data = paddle.rand((1, 3, 224, 224))
out = model(data)
print(out.shape) # [1, 1000]


# PaddleNLP
docs = paddle.hub.help('PaddlePaddle/PaddleNLP:develop', model='bert',)
print(docs)

model, tokenizer = paddle.hub.load('PaddlePaddle/PaddleNLP:develop', model='bert', model_name_or_path='bert-base-cased')
