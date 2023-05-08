#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/30 10:15
# @Author  : fan
import torch


class gpuConfig:
    cudaid = 0
    device = torch.device(f"cuda:{cudaid}" if torch.cuda.is_available() else "cpu")  # f就是能用来用大括号连接字符串的
    