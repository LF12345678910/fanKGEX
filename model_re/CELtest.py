#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/13 18:10
# @Author  : fan


import torch

tags_obj_pred = torch.tensor([[24.],
                              [31.],
                              [13.],
                              [13.],
                              [24.],
                              [31.],
                              [31.],
                              [ 2.],
                              [31.],
                              [31.],
                              [31.],
                              [ 8.],
                              [26.],
                              [ 6.],
                              [30.],
                              [19.],
                              [14.],
                              [ 6.],
                              [31.],
                              [ 8.],
                              [26.],
                              [31.],
                              [17.],
                              [31.],
                              [ 8.],
                              [30.],
                              [31.],
                              [19.],
                              [ 8.],
                              [31.],
                              [19.],
                              [31.],
                              [19.],
                              [19.],
                              [19.],
                              [31.],
                              [19.],
                              [ 8.],
                              [31.],
                              [19.],
                              [11.],
                              [24.],
                              [19.],
                              [19.],
                              [31.],
                              [19.],
                              [19.],
                              [19.],
                              [19.],
                              [19.],
                              [24.],
                              [31.],
                              [17.],
                              [31.],
                              [24.]], device='cuda:0')

tags_obj_true = torch.tensor([23, 9, 14, 2, 6, 17, 9, 25, 20, 18, 1, 19, 19, 19, 19, 19, 7, 18,
                              1, 19, 19, 19, 21, 11, 25, 20, 21, 22, 20, 21, 22, 20, 21, 22, 20, 21,
                              22, 20, 21, 22, 20, 21, 22, 20, 21, 22, 20, 19, 21, 6, 5, 29, 4, 24,
                              10], device='cuda:0')

import torch.nn as nn

loss_fn = nn.CrossEntropyLoss()
loss = loss_fn(tags_obj_pred, tags_obj_true)
print(loss.item())