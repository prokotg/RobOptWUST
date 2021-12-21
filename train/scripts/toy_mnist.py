import argparse

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from data.dataset import OnePixelMNIST, BGColouredMNIST
from models.mnist import MNISTPlainModel, MNISTPlainModelGrad, ColoredMNISTPlainModel, ColoredMNISTPlainModelGrad

parser = argparse.ArgumentParser(description='Run One-pixel MNIST training with or without gradient regularization')
parser.add_argument('-m', '--mode', type=str, choices=['vanilla', 'onepixel', 'bgcolor'], default='vanilla')
parser.add_argument('-g', '--gradreg', action='store_true', help='turn on gradient regularization')
parser.add_argument('-p', '--permute', action='store_true',
                    help='turn on permutation of bias pixels in validation dataset ')

args = parser.parse_args()

dataset = MNIST(root='../../data', train=True, download=True, transform=transforms.ToTensor())
mnist_train, mnist_val = random_split(dataset, [55000, 5000])


if args.mode == 'onepixel':
    mnist_train = OnePixelMNIST(train=True, dataset=mnist_train, permute=False)
    mnist_val = OnePixelMNIST(train=False, dataset=mnist_val, permute=args.permute)
elif args.mode == 'bgcolor':
    mnist_train = BGColouredMNIST(train=True, dataset=mnist_train, permute=False)
    mnist_val = BGColouredMNIST(train=False, dataset=mnist_val, permute=args.permute)

train_loader = DataLoader(mnist_train, batch_size=32)
val_loader = DataLoader(mnist_val, batch_size=512)

# model

if args.mode == 'vanilla' or (args.mode == 'onepixel' and not args.gradreg):
    model = MNISTPlainModel()
elif args.mode == 'onepixel':
    model = MNISTPlainModelGrad()
elif args.mode == 'bgcolor' and not args.gradreg:
    model = ColoredMNISTPlainModel()
elif args.mode == 'bgcolor':
    model = ColoredMNISTPlainModelGrad()

# training

model_checkpoint = ModelCheckpoint(save_top_k=-1)

trainer = pl.Trainer(max_epochs=50, callbacks=[model_checkpoint])
trainer.fit(model, train_loader, val_loader, )
