import argparse

import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms
from torchvision.datasets import MNIST

from data.dataset import OnePixelMNIST
from models.toy import MNISTToyModel, MNISTToyModelGrad

parser = argparse.ArgumentParser(description='Run One-pixel MNIST training with or without gradient regularization')
parser.add_argument('-m', '--mode', type=str, choices=['vanilla', 'bias'], default='vanilla')
parser.add_argument('-g', '--gradreg', action='store_true', help='turn on Jacobian regularization')
parser.add_argument('-p', '--permute', action='store_true',
                    help='turn on permutation of bias pixels in validation dataset ')

args = parser.parse_args()

dataset = MNIST(root='../../data', train=True, download=True, transform=transforms.ToTensor())
mnist_train, mnist_val = random_split(dataset, [55000, 5000])

if args.mode == 'bias':
    mnist_train = OnePixelMNIST(train=True, dataset=mnist_train, permute=False)
    mnist_val = OnePixelMNIST(train=False, dataset=mnist_val, permute=args.permute)

train_loader = DataLoader(mnist_train, batch_size=32)
val_loader = DataLoader(mnist_val, batch_size=512)

# model

if args.mode == 'vanilla' or (not args.gradreg):
    model = MNISTToyModel()
else:
    model = MNISTToyModelGrad()

# training
trainer = pl.Trainer(max_epochs=50)
trainer.fit(model, train_loader, val_loader)
