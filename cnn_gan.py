import argparse
import os
import numpy as np
import math
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
from torchvision.utils import save_image
import visdom

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch

os.makedirs("images", exist_ok=True)
vis = visdom.Visdom(server='http://localhost', port=8097)

def plot_loss(iteration, loss_D, loss_G):
    vis.line(
        X=[iteration],
        Y=[loss_D],
        win='Loss D',
        update='append' if iteration > 0 else None,
        opts=dict(title='Discriminator Loss', xlabel='Iteration', ylabel='Loss'),
        )
    vis.line(
        X=[iteration],
        Y=[loss_G],
        win='Loss G',
        update='append' if iteration > 0 else None,
        opts=dict(title='Generator Loss', xlabel='Iteration', ylabel='Loss'),
        )


parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.00015, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.4, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.65, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=500, help="interval between image samples")
parser.add_argument("--lr_decay", type=float, default=0.95, help="exponential learning rate decay")#学习率衰减策略
parser.add_argument('--model', type=str, default='cnn.pth', help='path to pretrained model')#加载预训练模型
opt = parser.parse_args()
print(opt)

img_shape = (opt.channels, opt.img_size, opt.img_size)
cuda = True if torch.cuda.is_available() else False

d_losses = []
g_losses = []

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.init_size = opt.img_size // 4
        self.l1 = nn.Sequential(nn.Linear(opt.latent_dim, 128 * self.init_size ** 2))

        self.conv_blocks = nn.Sequential(
            nn.BatchNorm2d(128),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 128, 3, stride=1, padding=1),
            nn.BatchNorm2d(128, 0.9),
            nn.LeakyReLU(0.3, inplace=True),
            nn.Upsample(scale_factor=2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1),
            nn.BatchNorm2d(64, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 32, 3, stride=1, padding=1),
            nn.BatchNorm2d(32, 0.8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 3, 3, stride=1, padding=1),
            nn.Tanh(),
        )

    def forward(self, z):
        out = self.l1(z)
        out = out.view(out.shape[0], 128, self.init_size, self.init_size)
        img = self.conv_blocks(out)
        return img


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 4, stride=2, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(256),
            nn.Flatten(),
            nn.Linear(65536, 128),
            nn.Sigmoid()
        )

    def forward(self, img):
        validity = self.model(img)
        return validity


# Loss function
adversarial_loss = torch.nn.BCELoss()

# 初始化 generator and discriminator
generator = Generator()
discriminator = Discriminator()

if cuda:
    generator.cuda()
    discriminator.cuda()
    adversarial_loss.cuda()

# Configure data loader

dataset = datasets.ImageFolder(
    "E:\\anime\\out1",
    transform=transforms.Compose(
        [transforms.Resize((opt.img_size, opt.img_size)), transforms.ToTensor(), transforms.Normalize([0.5] * 3, [0.5] * 3)]
    )
)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=opt.batch_size,
    shuffle=True,
    drop_last=True,
)


# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr/5, betas=(opt.b1, opt.b2))

scheduler_G = torch.optim.lr_scheduler.ExponentialLR(optimizer_G, gamma=opt.lr_decay)
scheduler_D = torch.optim.lr_scheduler.ExponentialLR(optimizer_D, gamma=opt.lr_decay)
#分别定义学习率衰减的策略

Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

# ----------
#  Training
# ----------

for epoch in range(opt.n_epochs):
    for i, (imgs, _) in enumerate(dataloader):

        # Adversarial ground truths

        valid = Variable(Tensor(opt.batch_size, opt.img_size).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(opt.batch_size, opt.img_size).fill_(0.0), requires_grad=False)#这里小改了一些，不知道为什么，反转就是要改

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
        d_loss = real_loss + fake_loss / 2

        # Measure discriminator's ability to classify real from generated samples
        if(epoch <= 2):
            d_loss.backward()
            optimizer_D.step()
        elif(i%2==0):
            d_loss.backward()
            optimizer_D.step()

        batches_done = epoch * len(dataloader) + i
        if(epoch%1==0):
            print(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                % (epoch, opt.n_epochs, i, len(dataloader), d_loss.item(), g_loss.item())
            )
            plot_loss(batches_done, d_loss.item(), g_loss.item())

        if batches_done % opt.sample_interval == 0:
            save_image(gen_imgs.data[:4], "cnn_128/%d.png" % batches_done, nrow=2, normalize=True)
        if batches_done % 3200 == 0:
            torch.save(generator.state_dict(), 'cnn_128.pth')
    scheduler_D.step()
    if(epoch%2==0 and 0.0002**(epoch/2) >= 1e-8):
        scheduler_G.step()