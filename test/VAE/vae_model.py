#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/12 16:29
# @Author  : fan


import torch
from torch import nn


class VAE(nn.Module):

    def __init__(self):
        super(VAE, self).__init__()

        # [b, 784] =>[b,20]
        # u: [b, 10]
        # sigma: [b, 10]
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 20),
            nn.ReLU()
        )

        # [b,10] => [b, 784]
        # sigmoid函数把结果压缩到0~1
        self.decoder = nn.Sequential(
            nn.Linear(10, 64),
            nn.ReLU(),
            nn.Linear(64, 256),
            nn.ReLU(),
            nn.Linear(256, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        """
        :param x:
        :return:
        """
        batchsz = x.size(0)
        # flatten
        x = x.view(batchsz, 784)
        # encoder
        # [b, 20], including mean and sigma
        h_ = self.encoder(x)
        # chunk 在第二维上拆分成两部分
        # [b, 20] => [b,10] and [b, 10]
        mu, sigma = h_.chunk(2, dim=1)
        # reparametrize tirchk, epison~N(0, 1)
        # torch.randn_like(sigma)表示正态分布
        h = mu + sigma * torch.randn_like(sigma)

        # decoder
        x_hat = self.decoder(h)
        # reshape
        x_hat = x_hat.view(batchsz, 1, 28, 28)

        # KL
        # 1e-8是防止σ^2接近于零时该项负无穷大
        # (batchsz*28*28)是让kld变小
        kld = 0.5 * torch.sum(
            torch.pow(mu, 2) +
            torch.pow(sigma, 2) -
            torch.log(1e-8 + torch.pow(sigma, 2)) - 1
        ) / (batchsz * 28 * 28)

        return x, kld