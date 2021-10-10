import argparse
import timm
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import transforms

from data.dataset import ImageNet9L

parser = argparse.ArgumentParser(description='Run One-pixel MNIST training with or without gradient regularization')
parser.add_argument('-n', '--network', type=str)
parser.add_argument('-p', '--save-path', type=str, default='trained_model.pkl')
parser.add_argument('-d', '--dataset', type=str, default='original')

args = parser.parse_args()

dataset = ImageNet9L(args.dataset)
imagenet_train, imagenet_val = random_split(dataset, [55000, 5000])

train_loader = DataLoader(imagenet_train, batch_size=32)
val_loader = DataLoader(imagenet_val, batch_size=512)
model = timm.create_model(args.network, pretrained=True)

# training
trainer = pl.Trainer(max_epochs=50)
trainer.fit(model, train_loader, val_loader)
