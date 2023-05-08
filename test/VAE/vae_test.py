#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/3/12 16:24
# @Author  : fan

import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision import transforms, datasets

from vae_model import VAE

import visdom


def main():
    mnist_train = datasets.MNIST('mnist', True, transform=transforms.Compose([
        transforms.ToTensor()
    ]), download=True)
    mnist_train = DataLoader(mnist_train, batch_size=32, shuffle=True)

    mnist_test = datasets.MNIST('mnist', False, transform=transforms.Compose([
        transforms.ToTensor()
    ]), download=True)
    mnist_test = DataLoader(mnist_test, batch_size=32, shuffle=True)

    # 无监督学习，不能使用label
    x, _ = iter(mnist_train).next()
    print('x:', x.shape)

    device = torch.device('cuda')
    # model = AE().to(device)
    model = VAE().to(device)
    criteon = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    print(model)

    viz = visdom.Visdom()

    for epoch in range(1000):

        for batchidx, (x, _) in enumerate(mnist_train):
            # [b, 1, 28, 28]
            x = x.to(device)

            x_hat, kld = model(x)
            loss = criteon(x_hat, x)

            if kld is not None:
                elbo = - loss - 1.0 * kld
                loss = - elbo

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    print(epoch, 'loss', loss.item(), kld.item())

    x, _ = iter(mnist_test).next()
    x = x.to(device)
    with torch.no_grad():

        x_hat = model(x)
    # nrow表示一行的图片
    viz.images(x, nrow=8, win='x', optis=dic(title='x'))
    viz.images(x_hat, nrow=8, win='x_hat', optis=dic(title='x_hat'))


if __name__ == '__main__':
    main()