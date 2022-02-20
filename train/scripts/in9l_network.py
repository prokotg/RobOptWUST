import argparse
import timm
import pickle
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from models.timm import TIMMModel
from data.augmentations import UpdateChancesBasedOnAccuracyCallback, UpdateChancesBasedOnPerSampleEntropyCallback, UpdateChancesBasedOnEntropyTakeTopCallback
import data.imagenet as ImageNet9

parser = argparse.ArgumentParser(description='Make any network from TIMM learn on IN9L!')
parser.add_argument('-n', '--network', type=str)
parser.add_argument('-p', '--save-path', type=str, default='trained_model.pkl')
parser.add_argument('-d', '--dataset-path', type=str, default='data/original/')
parser.add_argument('-l', '--log-dir', type=str, default='logs/')
parser.add_argument('-w', '--workers', type=int, default=4)
parser.add_argument('-g', '--gpus', type=int, default=1)
parser.add_argument('-e', '--epochs', type=int, default=200)
parser.add_argument('-t', '--use-background-transform', type=bool, default=False)
parser.add_argument('-b', '--use-background-blur', type=bool, default=False)
parser.add_argument('--use-auto-accuracy-background-transform', type=bool, default=False)
parser.add_argument('--use-auto-entropy-background-transform', type=bool, default=False)
parser.add_argument('--use-auto-top-entropy-background-transform', type=bool, default=False)
parser.add_argument('--calculate-entropy-per-sample', type=bool, default=True)
parser.add_argument('--reverse-transform-chances', type=bool, default=True)
parser.add_argument('--recalculate-entropy-after-learning', type=bool, default=False)
parser.add_argument('--background-transform-chance', type=float, default=0.0)
parser.add_argument('--augmentation-dataset-size', type=float, default=0.2)
parser.add_argument('--backgrounds-path', type=str, default='data/only_bg_t/train')
parser.add_argument('--foregrounds-path', type=str, default='data/only_fg/train')

args = parser.parse_args()

tensorboard_logger = pl_loggers.TensorBoardLogger(args.log_dir, name=f'in9l_{args.network}')

if args.use_background_transform and args.use_background_blur:
    print('Do not use both of the augmentation in the same time!')
if args.use_background_transform:
    imagenet_dataset = ImageNet9.ImageNetBackgroundChangeAugmented(args.dataset_path, args.backgrounds_path, args.foregrounds_path, args.background_transform_chance)
elif args.use_background_blur:
    imagenet_dataset = ImageNet9.ImageNetBackgroundBlurAugmented(args.dataset_path, args.backgrounds_path, args.foregrounds_path, args.background_transform_chance if args.background_transform_chance != 0.0 else None)
else:
    imagenet_dataset = ImageNet9.ImageNet9(args.dataset_path)


train_loader, val_loader = imagenet_dataset.make_loaders(batch_size=32, workers=args.workers, add_path=True, add_target=True, index_instead_of_class=args.calculate_entropy_per_sample)
model = TIMMModel(timm.create_model(args.network, pretrained=False, num_classes=9), save_entropy=not args.recalculate_entropy_after_learning)

callbacks = []
if args.use_auto_accuracy_background_transform:
    callbacks.append(UpdateChancesBasedOnAccuracyCallback(model, imagenet_dataset.augmentation, args.augmentation_dataset_size, use_cuda=args.gpus > 0, reverse_chances=args.reverse_transform_chances))
elif args.use_auto_entropy_background_transform:
    callbacks.append(UpdateChancesBasedOnPerSampleEntropyCallback(train_loader, model, imagenet_dataset.augmentation, recalculate_on_end=args.recalculate_entropy_after_learning, use_cuda=args.gpus > 0, reverse_chances=args.reverse_transform_chances))
elif args.use_auto_top_entropy_background_transform:
    callbacks.append(UpdateChancesBasedOnEntropyTakeTopCallback(train_loader, model, imagenet_dataset.augmentation, args.augmentation_dataset_size, recalculate_on_end=args.recalculate_entropy_after_learning, use_cuda=args.gpus > 0, reverse_chances=args.reverse_transform_chances))

# training
trainer = pl.Trainer(max_epochs=args.epochs, logger=tensorboard_logger, gpus=args.gpus, callbacks=callbacks)
trainer.fit(model, train_loader, val_loader)

print(args.save_path)
model_file = open(args.save_path, 'wb')
pickle.dump(model, model_file)
model_file.close()
