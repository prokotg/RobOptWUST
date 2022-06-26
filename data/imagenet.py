import torch as ch
import os
from torchvision import transforms
from . import folder
from . import augmentations
from torch.utils.data import DataLoader
from . import shared


def make_loader(workers, batch_size, transforms, data_path, name, shuffle_val=False,
                add_path=False, use_background_replacement=False, additional_paths=None, use_swap_background_minibatch_loader=False,
                assigned_backgrounds_per_instance=None, random_seed=None):
    path = os.path.join(data_path, name)
    if not os.path.exists(path):
        raise ValueError("{1} data must be stored in {0}".format(path, name))

    assert not (use_background_replacement and additional_paths is None), 'When using background replacement dataset, specify with additional_paths where to take the backgrounds from'
    
    if additional_paths is None:
        set_folder = folder.ImageFolder(root=path, transform=transforms, add_path=add_path)
    else:
        if use_background_replacement:
            set_folder = folder.BackgroundReplacementDataset(roots=[path] + additional_paths, transform=transforms, add_path=add_path)
        elif use_swap_background_minibatch_loader:
            set_folder = folder.SwapBackgroundFolder(root=path, backgrounds=additional_paths[0], foregrounds=additional_paths[1], pre_transform=transforms[0], post_transform=transforms[1],
                                                     add_path=add_path, assigned_backgrounds_per_instance=assigned_backgrounds_per_instance, random_seed=random_seed)
        else:
            set_folder = folder.MultiImageFolder(root=[path] + additional_paths, transform=transforms, add_path=add_path)

    loader = DataLoader(set_folder, batch_size=batch_size, shuffle=shuffle_val, num_workers=workers, pin_memory=True)

    return loader


def generate_loaders(workers, batch_size, transform_train, transform_test, data_path, dataset,
                     shuffle_val=False, add_path=False, use_background_replacement=False, additional_paths=None, use_swap_background_minibatch_loader=False,
                     assigned_backgrounds_per_instance=None, random_seed=None):
    '''
    '''
    print(f"==> Preparing dataset {dataset}..")

    train_loader = make_loader(workers, batch_size, transform_train, data_path, 'train', True, add_path=add_path,
                               additional_paths=additional_paths, use_background_replacement=use_background_replacement, use_swap_background_minibatch_loader=use_swap_background_minibatch_loader,
                               assigned_backgrounds_per_instance=assigned_backgrounds_per_instance, random_seed=random_seed)

    test_loader = make_loader(workers, batch_size, transform_test, data_path, 'val', shuffle_val, add_path=add_path)
    return train_loader, test_loader


class DataSet(object):
    '''
    '''

    def __init__(self, ds_name, data_path, **kwargs):
        """
        """
        required_args = ['num_classes', 'mean', 'std', 'transform_test', 'transform_train']
        assert set(kwargs.keys()) == set(required_args), "Missing required args, only saw %s" % kwargs.keys()
        self.ds_name = ds_name
        print(data_path)
        self.data_path = data_path
        self.__dict__.update(kwargs)

    def make_loaders(self, workers, batch_size, shuffle_val=False, add_path=False, use_background_replacement=False,
                     use_swap_background_minibatch_loader=False, additional_paths=None, assigned_backgrounds_per_instance=None, random_seed=None):
        '''
        '''
        return generate_loaders(workers=workers,
                                batch_size=batch_size,
                                transform_train=self.transform_train,
                                transform_test=self.transform_test,
                                data_path=self.data_path,
                                dataset=self.ds_name,
                                shuffle_val=shuffle_val,
                                add_path=add_path,
                                use_background_replacement=use_background_replacement,
                                use_swap_background_minibatch_loader=use_swap_background_minibatch_loader,
                                additional_paths=additional_paths,
                                assigned_backgrounds_per_instance=assigned_backgrounds_per_instance,
                                random_seed=random_seed)

    def get_model(self, arch, pretrained):
        '''
        Args:
            arch (str) : name of architecture
            pretrained (bool): whether to try to load torchvision
                pretrained checkpoint
        Returns:
            A model with the given architecture that works for each
            dataset (e.g. with the right input/output dimensions).
        '''

        raise NotImplementedError


class ImageNet9(DataSet):
    '''
    '''

    def __init__(self, data_path, divide_transforms=False, **kwargs):
        """
        """
        ds_name = 'ImageNet9'
        common_tr = [augmentations.UnwrapTupled(), transforms.Resize((224, 224)), transforms.ToTensor()]
        train_tr = common_tr + [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
                                transforms.ColorJitter(0.4, 0.4, 0.4)]
        ds_kwargs = {
            'num_classes': 9,
            'mean': ch.tensor([0.4717, 0.4499, 0.3837]),
            'std': ch.tensor([0.2600, 0.2516, 0.2575]),
            'transform_train': transforms.Compose(train_tr) if not divide_transforms else (transforms.Compose(train_tr[0:3]), transforms.Compose(train_tr[3:])),
            'transform_test': transforms.Compose(common_tr)
        }
        super(ImageNet9, self).__init__(ds_name,
                                        data_path, **ds_kwargs)


class ImageNet(DataSet):
    '''
    '''

    def __init__(self, data_path, divide_transforms=False, **kwargs):
        """
        """
        ds_name = 'ImageNet'
        common_tr = [augmentations.UnwrapTupled(), transforms.Resize((224, 224)), transforms.ToTensor()]
        train_tr = common_tr + [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
                                transforms.ColorJitter(0.4, 0.4, 0.4)]
        ds_kwargs = {
            'num_classes': 1000,
            'mean': ch.tensor([0.485, 0.456, 0.406]),
            'std': ch.tensor([0.229, 0.224, 0.225]),
            'transform_train': transforms.Compose(train_tr) if not divide_transforms else (transforms.Compose(train_tr[0:3]), transforms.Compose(train_tr[3:])),
            'transform_test': transforms.Compose(common_tr)
        }
        super(ImageNet, self).__init__(ds_name,
                                       data_path, **ds_kwargs)


class DataSetBackgroundAugmented(DataSet):
    def make_loaders(self, workers, batch_size, shuffle_val=False, add_path=False, use_background_replacement=False, use_swap_background_minibatch_loader=False,
                     additional_paths=None, assigned_backgrounds_per_instance=None, random_seed=None):
        if additional_paths is None:
            additional_paths = [self.foregrounds_path]
        return generate_loaders(workers=workers,
                                batch_size=batch_size,
                                transform_train=self.transform_train,
                                transform_test=self.transform_test,
                                data_path=self.data_path,
                                dataset=self.ds_name,
                                shuffle_val=shuffle_val,
                                add_path=add_path,
                                use_background_replacement=use_background_replacement,
                                use_swap_background_minibatch_loader=use_swap_background_minibatch_loader,
                                additional_paths=additional_paths,
                                assigned_backgrounds_per_instance=assigned_backgrounds_per_instance,
                                random_seed=random_seed)


def generate_paths(data_path):
    result = []
    for data_path in [f'{data_path}/train']:
        for inner_dir in sorted(os.listdir(data_path)):
            for image_path in sorted(os.listdir(f'{data_path}/{inner_dir}')):
                result.append(f'{data_path}/{inner_dir}/{image_path}')
    return result


class ImageNetBackgroundChangeAugmented(DataSetBackgroundAugmented):
    '''
    '''

    def __init__(self, data_path, backgrounds_path, foregrounds_path, background_transform_chance, **kwargs):
        """
        """
        self.foregrounds_path = foregrounds_path
        backgrounds = generate_paths(backgrounds_path)
        self.augmentation = augmentations.RandomBackgroundPerClass([background_transform_chance] * 9, backgrounds)

        common_tr = [augmentations.UnwrapTupled(), transforms.Resize((224, 224)), transforms.ToTensor()]
        train_tr = [self.augmentation, transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(0.4, 0.4, 0.4)]
        ds_name = 'ImageNet'
        ds_kwargs = {
            'num_classes': 9,
            'mean': ch.tensor([0.485, 0.456, 0.406]),
            'std': ch.tensor([0.229, 0.224, 0.225]),
            'transform_train': transforms.Compose(train_tr),
            'transform_test': transforms.Compose(common_tr)
        }
        super().__init__(ds_name,
                         data_path, **ds_kwargs)


class ImageNetBackgroundChangeStacked(DataSetBackgroundAugmented):
    '''
    '''

    def __init__(self, data_path, backgrounds_path, foregrounds_path, background_transform_chance, backgrounds_per_example=None, random_seed=None, **kwargs):
        """
        """
        self.foregrounds_path = foregrounds_path
        backgrounds = generate_paths(backgrounds_path)
        if backgrounds_per_example is not None:
            backgrounds = shared.divide_paths(backgrounds, backgrounds, paths_per_instance=backgrounds_per_example, random_seed=random_seed)
        self.augmentation = augmentations.RandomBackgroundPerClass([background_transform_chance] * 9, backgrounds)
        common_tr = [augmentations.UnwrapTupled(), transforms.Resize((224, 224)), transforms.ToTensor()]
        train_tr = [self.augmentation, transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(0.4, 0.4, 0.4)]
        ds_name = 'ImageNet'
        ds_kwargs = {
            'num_classes': 9,
            'mean': ch.tensor([0.485, 0.456, 0.406]),
            'std': ch.tensor([0.229, 0.224, 0.225]),
            'transform_train': transforms.Compose(train_tr),
            'transform_test': transforms.Compose(common_tr)
        }
        super().__init__(ds_name,
                         data_path, **ds_kwargs)


class ImageNetBackgroundBlurAugmented(DataSetBackgroundAugmented):
    '''
    '''

    def __init__(self, data_path, backgrounds_path, foregrounds_path, background_transform_chance, **kwargs):
        """
        """
        self.backgrounds_path = backgrounds_path
        self.default_path = data_path
        self.augmentation = augmentations.RandomBackgroundBlur([background_transform_chance] * 9)

        common_tr = [augmentations.UnwrapTupled(), transforms.Resize((224, 224)), transforms.ToTensor()]
        train_tr = [self.augmentation, transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(),
                    transforms.ColorJitter(0.4, 0.4, 0.4)]
        ds_name = 'ImageNet'
        ds_kwargs = {
            'num_classes': 9,
            'mean': ch.tensor([0.485, 0.456, 0.406]),
            'std': ch.tensor([0.229, 0.224, 0.225]),
            'transform_train': transforms.Compose(train_tr),
            'transform_test': transforms.Compose(common_tr)
        }
        super().__init__(ds_name,
                         foregrounds_path, **ds_kwargs)
