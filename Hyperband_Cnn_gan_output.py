import argparse
import os
import logging
logging.getLogger().setLevel(logging.CRITICAL + 1)

from Train_cnn_output import train_gan
import torch





os.makedirs("images", exist_ok=True)

parser = argparse.ArgumentParser()
parser.add_argument("--n_epochs", type=int, default=1000, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=128, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.000318407269634977, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.7353787983238228, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.5404371677379848, help="adam: decay of first order momentum of gradient")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--latent_dim", type=int, default=100, help="dimensionality of the latent space")
parser.add_argument("--img_size", type=int, default=128, help="size of each image dimension")
parser.add_argument("--channels", type=int, default=3, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=500, help="interval between image samples")
parser.add_argument("--lr_decay", type=float, default=0.95, help="exponential learning rate decay")#学习率衰减策略
parser.add_argument('--model', type=str, default='cnn.pth', help='path to pretrained model')#加载预训练模型
opt = parser.parse_args()
print(opt)


def main():
    # 使用函数：
    res = train_gan(opt)

if __name__ == "__main__":
    main()





