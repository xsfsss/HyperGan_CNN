
import torch

import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()

        # 第一个卷积层，卷积核大小为 3x3，步幅根据参数 `stride` 决定，填充为 1
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)

        # 批标准化层，用于加速收敛和稳定性
        self.bn1 = nn.BatchNorm2d(out_channels)

        # 非线性激活函数，这里使用 ReLU
        self.relu = nn.ReLU(inplace=True)

        # 第二个卷积层，卷积核大小为 3x3，步幅为 1，填充为 1
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)

        # 批标准化层
        self.bn2 = nn.BatchNorm2d(out_channels)

        # 步幅
        self.stride = stride

    def forward(self, x):
        # 保存输入作为残差连接的一部分
        residual = x

        # 第一个卷积层 -> 批标准化 -> 非线性激活
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        # 第二个卷积层 -> 批标准化
        out = self.conv2(out)
        out = self.bn2(out)

        # 处理残差连接，如果步幅不为 1 或输入通道数与输出通道数不一致，则需要适配残差
        if self.stride != 1 or x.size(1) != out.size(1):
            # 通过平均池化改变输入的大小
            residual = nn.functional.avg_pool2d(residual, 2)
            # 使用零填充来适配通道数
            residual = torch.cat([residual] + [residual * 0], dim=1)

        # 将残差与输出相加
        out += residual
        # 应用非线性激活函数
        out = self.relu(out)

        return out


class Generator(nn.Module):
    def __init__(self, opt):
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
            nn.Dropout(0.4),  # 添加 Dropout 层，可以根据需要调整丢弃概率
            ResidualBlock(64, 128, stride=2),
            #nn.LeakyReLU(0.2, inplace=True),残差模块中加了
            nn.BatchNorm2d(128),
            nn.Dropout(0.4),  # 添加 Dropout 层，可以根据需要调整丢弃概率
            ResidualBlock(128, 256, stride=2),
            #nn.LeakyReLU(0.2, inplace=True),
            nn.BatchNorm2d(256),
            nn.Dropout(0.3),  # 添加 Dropout 层，可以根据需要调整丢弃概率
            nn.Flatten(),
            nn.Linear(65536, 128),
            nn.Sigmoid()
        )

    def forward(self, img):
        validity = self.model(img)
        return validity

