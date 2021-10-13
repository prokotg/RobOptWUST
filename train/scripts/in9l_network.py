import argparse
import timm
import pickle
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from pytorch_lightning import loggers as pl_loggers
from models.timm import TIMMModel

import data.imagenet as ImageNet9

parser = argparse.ArgumentParser(description='Make any network from TIMM learn on IN9L!')
parser.add_argument('-n', '--network', type=str)
parser.add_argument('-p', '--save-path', type=str, default='trained_model.pkl')
parser.add_argument('-d', '--dataset-path', type=str, default='data/only_fg/')
parser.add_argument('-l', '--log-dir', type=str, default='logs/')
parser.add_argument('-w', '--workers', default=4)
parser.add_argument('-g', '--gpus', default=1)
parser.add_argument('-e', '--epochs', default=50)

args = parser.parse_args()

tensorboard_logger = pl_loggers.TensorBoardLogger(args.log_dir, name=f'in9l_{args.network}')

imagenet_dataset = ImageNet9.ImageNet9(args.dataset_path)

train_loader, val_loader = imagenet_dataset.make_loaders(batch_size=64, workers=args.workers)
model = TIMMModel(timm.create_model(args.network, pretrained=False, num_classes=9))

# training
trainer = pl.Trainer(max_epochs=args.epochs, logger=tensorboard_logger, gpus=args.gpus)
trainer.fit(model, train_loader, val_loader)

model_file = open(args.save_path, 'wb')
pickle.dump(model, model_file)
model_file.close()
