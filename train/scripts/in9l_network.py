import argparse
import timm
import pickle as pkl
import pytorch_lightning as pl
from torchvision import models
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import Callback
from models.timm import TIMMModel
from models.flow_model import FlowModel, FreezeNetworkCallback
from models.flows import ConditionalNICE, ConditionalMAF
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
    parser.add_argument('-t', '--use-background-transform', type=bool, default=False, action='store_true')
    parser.add_argument('-b', '--use-background-blur', type=bool, default=False, action='store_true')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--use-auto-background-transform', type=bool, default=False, action='store_true')
    parser.add_argument('--use-swap-background-minibatch-loader', type=bool, default=False, action='store_true')
    parser.add_argument('--use-flow-model', type=bool, default=False, action='store_true')
    parser.add_argument('--use-loaded-model', type=bool, default=False, action='store_true')
    parser.add_argument('--select-gpu', type=int, default=None)
    parser.add_argument('--limit-backgrounds-per-instance', type=int, default=None)
    parser.add_argument('--use-staged-flow-learning', type=bool, default=True)
    parser.add_argument('--flow-learning-stage-start-epoch', type=int, default=5)
    parser.add_argument('--flow-embedding-size', type=int, default=128)
    parser.add_argument('--background-transform-chance', type=float, default=0.0)
    parser.add_argument('--augmentation-checking-dataset-size', type=float, default=0.2)
    parser.add_argument('--backgrounds-path', type=str, default='data/only_bg_t/train')
    parser.add_argument('--foregrounds-path', type=str, default='data/only_fg/train')
    parser.add_argument('--base-model-path', type=str, default='networks/network.pkl')

    args = parser.parse_args()

    tensorboard_logger = pl_loggers.TensorBoardLogger(args.log_dir, name=f'in9l_{args.network}')

    assert (1 if args.use_background_transform else 0) + (1 if args.use_background_blur else 0) <= 1, 'Do not use two of the augmentation in the same time!'
    assert args.use_background_transform if args.use_background_replacement_metadata else True, 'Background replacement metadata must be used along with background transforms!'
    
    if args.use_background_transform:
        imagenet_dataset = ImageNet9.ImageNetBackgroundChangeAugmented(args.dataset_path, args.backgrounds_path, args.foregrounds_path, args.background_transform_chance)
    elif args.use_background_blur:
        imagenet_dataset = ImageNet9.ImageNetBackgroundBlurAugmented(args.dataset_path, args.backgrounds_path, args.foregrounds_path, args.background_transform_chance)
    else:
        imagenet_dataset = ImageNet9.ImageNet9(args.dataset_path, divide_transforms=args.use_swap_background_minibatch_loader)
    
    train_loader, val_loader = imagenet_dataset.make_loaders(batch_size=32, workers=args.workers, add_path=True, use_swap_background_minibatch_loader=args.use_swap_background_minibatch_loader, additional_paths=[args.backgrounds_path, args.foregrounds_path] if args.use_swap_background_minibatch_loader else None, assigned_backgrounds_per_instance=args.limit_backgrounds_per_instance, random_seed=42)

    if not args.use_flow_model:
        model = TIMMModel(timm.create_model(args.network, pretrained=False, num_classes=9))
    else:
        # Hiperparametry -
        #   - z_count - czyli jak dużo ma dogenerować obrazków z flowa
        #   - SwapBackgroundFolder ma argument changed_backgrounds_count nigdzie nie wypuszczony, mówiący ile ma być obrazków z podmienionymi tłami w minibatchu
        # Zmiana flowa polega na zmianie obiektu flow. Przykład CMAFowy:
        # flow = ConditionalMAF(embedding_size, hidden_features=64, num_layers=3, conditional_count=embedding_size, num_blocks_per_layer=4)
        if args.use_loaded_model:
            model_file = open(args.base_model_path, 'rb')
            base_model = pkl.load(model_file)
            last_layer_size = base_model.model.fc.weight.shape[1]
            base_model.model.fc = nn.Identity()
            model_file.close()
        else:
            base_model = models.resnet50(pretrained=True)
            last_layer_size = base_model.fc.weight.shape[1]
            base_model.fc = nn.Identity()
        
        for p in base_model.parameters():
            p.requires_grad = False
        
        flow = ConditionalNICE(args.flow_embedding_size, hidden_sizes=[256, 256, 256], num_layers=4, conditional_count=args.flow_embedding_size)
        model = FlowModel(base_model,
            nn.Sequential(
                nn.Linear(last_layer_size, 512),
                nn.ReLU(),
                nn.Linear(512, args.flow_embedding_size),
                nn.ReLU(),
            ), nn.Sequential(
                nn.Linear(args.flow_embedding_size, 128),
                nn.Sigmoid(),
                nn.Linear(128, 9),
                nn.Sigmoid()
            ), 9,
            flow=flow,
            embedding_size=args.flow_embedding_size, z_count=16)

    callbacks = []

    class AutopickleModel(Callback):
        def __init__(self, model):
            self.model = model

        def on_validation_epoch_end(self, trainer, module):
            model_file = open(args.save_path, 'wb')
            if hasattr(self.model, 'base_model'):
                # The base model should not need it anymore- and it's a bit messing up pickling
                self.model.base_model.train_dataloader = None
                self.model.base_model.val_dataloader = None
                self.model.base_model.trainer = None
            pkl.dump(self.model, model_file)
            model_file.close()

    callbacks.append(AutopickleModel(model))
    if args.use_auto_background_transform:
        callbacks.append(UpdateChancesBasedOnAccuracyCallback(model, imagenet_dataset.augmentation, args.augmentation_checking_dataset_size, args.gpus > 0))
    if args.use_flow_model and args.use_staged_flow_learning:
        callbacks.append(FreezeNetworkCallback(model, train_loader, [(1, True, True, False, False), (args.flow_learning_stage_start_epoch, False, True, True, True)]))
    
    trainer = pl.Trainer(max_epochs=args.epochs, logger=tensorboard_logger, gpus=args.gpus if args.select_gpu is None else [args.select_gpu], callbacks=callbacks)
    trainer.fit(model, train_loader, val_loader)

    model_file = open(args.save_path, 'wb')
    pkl.dump(model, model_file)
    model_file.close()
