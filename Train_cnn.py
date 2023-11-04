import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from Net import Generator
from Net import Discriminator

import numpy as np
import math
import matplotlib.pyplot as plt
import torchvision.transforms as transforms

from vis_base import plot_loss

def train_gan(config, opt):
    img_shape = (opt.channels, opt.img_size, opt.img_size)
    cuda = True if torch.cuda.is_available() else False

    d_losses = []
    g_losses = []

    # 这里，我们将从config中提取超参数
    lr = config["lr"]
    b1 = config["b1"]
    b2 = config["b2"]
    # 可以继续为其他需要的超参数做同样的操作

    # Loss function
    adversarial_loss = torch.nn.BCELoss()

    # Initialize generator and discriminator
    generator = Generator(opt)
    discriminator = Discriminator()

    if cuda:
        generator.cuda()
        discriminator.cuda()
        adversarial_loss.cuda()

    # Configure data loader

    dataset = datasets.ImageFolder(
        "E:\\anime\\out1",
        transform=transforms.Compose(
            [transforms.Resize((opt.img_size, opt.img_size)), transforms.ToTensor(),
             transforms.Normalize([0.5] * 3, [0.5] * 3)]
        )
    )

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=opt.batch_size,
        shuffle=True,
        drop_last=True,
    )

    # Optimizers
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=lr, betas=(b1, b2))
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=lr / 5, betas=(b1, b2))

    scheduler_G = torch.optim.lr_scheduler.ExponentialLR(optimizer_G, gamma=opt.lr_decay)
    scheduler_D = torch.optim.lr_scheduler.ExponentialLR(optimizer_D, gamma=opt.lr_decay)
    # 分别定义学习率衰减的策略

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

    # 使用config中的参数替换原有的opt参数


    for epoch in range(opt.n_epochs):
        for i, (imgs, _) in enumerate(dataloader):

            # Adversarial ground truths

            #valid = Variable(Tensor(opt.batch_size, opt.img_size).fill_(1.0), requires_grad=False)
            valid = torch.ones(opt.batch_size, opt.img_size).fill_(1.0).to('cuda', dtype=torch.float32)
            fake = Variable(Tensor(opt.batch_size, opt.img_size).fill_(0.0),
                            requires_grad=False)  # 这里小改了一些，不知道为什么，反转就是要改

            # Configure input
            real_imgs = Variable(imgs.type(Tensor))

            # -----------------
            #  Train Generator

            # Sample noise as generator input
            z = Variable(Tensor(np.random.normal(0, 1, (imgs.shape[0], opt.latent_dim))))

            # Generate a batch of images
            # -----------------

            optimizer_G.zero_grad()
            gen_imgs = generator(z)

            # Loss measures generator's ability to fool the discriminator
            g_loss = adversarial_loss(discriminator(gen_imgs), valid)

            g_loss.backward()
            optimizer_G.step()

            # ---------------------
            #  Train Discriminator
            # ---------------------

            optimizer_D.zero_grad()
            real_loss = adversarial_loss(discriminator(real_imgs), valid)
            fake_loss = adversarial_loss(discriminator(gen_imgs.detach()), fake)
            d_loss = real_loss / 2 + fake_loss #想了很久，觉得把real_loss 给小一点，因为生成器输出图像也很烂（逻辑理解

            # 判别真伪
            if (epoch <= 2):
                d_loss.backward()
                optimizer_D.step()
            elif (i % 2 == 0):
                d_loss.backward()
                optimizer_D.step()

            batches_done = epoch * len(dataloader) + i
            if (epoch % 1 == 0):
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
                )
                plot_loss(batches_done, d_loss.item(), g_loss.item())

            """if batches_done % opt.sample_interval == 0:
                save_image(gen_imgs.data[:4], "cnn_128/%d.png" % batches_done, nrow=2, normalize=True)
            if batches_done % 3200 == 0:
                torch.save(generator.state_dict(), 'cnn_128.pth')"""
        scheduler_D.step()
        scheduler_G.step()

    # 返回某种评价指标，例如G的损失，来指示这种配置的效果。Hyperband会尝试最小化这个值。
    return d_loss.item()    #因为我discriminator比较差，所以我优先优化这个部分
    #return list(d_loss.values())[-1]