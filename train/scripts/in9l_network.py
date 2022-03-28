import argparse
import timm
import pickle as pkl
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torchvision import transforms, models
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from models.timm import TIMMModel
from models.flow import FlowModel, ConditionalNICE
from torch import nn
from data.augmentations import UpdateChancesBasedOnAccuracyCallback
import data.imagenet as ImageNet9

if __name__ == "__main__":

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
    parser.add_argument('--use-auto-background-transform', type=bool, default=False)
    parser.add_argument('--use-swap-background-minibatch-loader', type=bool, default=False)
    parser.add_argument('--background-transform-chance', type=float, default=0.0)
    parser.add_argument('--augmentation-checking-dataset-size', type=float, default=0.2)
    parser.add_argument('--backgrounds-path', type=str, default='data/only_bg_t/train')
    parser.add_argument('--foregrounds-path', type=str, default='data/only_fg/train')
    parser.add_argument('--base-model-path', type=str, default='networks/network.pkl')

    args = parser.parse_args()

    tensorboard_logger = pl_loggers.TensorBoardLogger(args.log_dir, name=f'in9l_{args.network}')

    if args.use_background_transform and args.use_background_blur:
        print('Do not use both of the augmentation in the same time!')
    if args.use_background_transform:
        imagenet_dataset = ImageNet9.ImageNetBackgroundChangeAugmented(args.dataset_path, args.backgrounds_path, args.foregrounds_path, args.background_transform_chance)
    elif args.use_background_blur:
        imagenet_dataset = ImageNet9.ImageNetBackgroundBlurAugmented(args.dataset_path, args.backgrounds_path, args.foregrounds_path, args.background_transform_chance)
    else:
        imagenet_dataset = ImageNet9.ImageNet9(args.dataset_path, divide_transforms=args.use_swap_background_minibatch_loader)
    
    train_loader, val_loader = imagenet_dataset.make_loaders(batch_size=8, workers=args.workers, add_path=True, use_swap_background_minibatch_loader=args.use_swap_background_minibatch_loader, additional_paths=[args.backgrounds_path, args.foregrounds_path] if args.use_swap_background_minibatch_loader else None)

    if False:
        model = TIMMModel(timm.create_model(args.network, pretrained=False, num_classes=9))
    else:
        if False:
            model_file = open(args.base_model_path, 'rb')
            base_model = pkl.load(model_file)
            last_layer_size = len(base_model[-1].bias)
            model_file.close()
        else:
            base_model = models.resnet50(pretrained=True)
            last_layer_size = base_model.fc.weight.shape[1]
            base_model.fc = nn.Identity()
        
        for p in base_model.parameters():
            p.requires_grad = False
        embedding_size = 100
        
        model = FlowModel(base_model,
        nn.Sequential(
            nn.Linear(last_layer_size, 512),
            nn.ReLU(),
            nn.Linear(512, embedding_size),
            nn.Sigmoid(),
        ), nn.Sequential(
            nn.Linear(embedding_size, 128),
            nn.Sigmoid(),
            nn.Linear(128, 9)
        ), 9, flow=ConditionalNICE(embedding_size, hidden_sizes=[256, 256, 256], num_layers=4, conditional_count=embedding_size), embedding_size=embedding_size, z_count=2)

    callbacks = []
    if args.use_auto_background_transform:
        callbacks.append(UpdateChancesBasedOnAccuracyCallback(model, imagenet_dataset.augmentation, args.augmentation_checking_dataset_size, args.gpus > 0))
    # training
    trainer = pl.Trainer(max_epochs=args.epochs, logger=tensorboard_logger, gpus=args.gpus, callbacks=callbacks)
    trainer.fit(model, train_loader, val_loader)

    print(args.save_path)
    model_file = open(args.save_path, 'wb')
    pkl.dump(model, model_file)
    model_file.close()
