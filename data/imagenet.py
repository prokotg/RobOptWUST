import torch as ch
import os
from torchvision import transforms
from . import folder
from . import augmentations
import time
from torch.utils.data import DataLoader


def make_loader(workers, batch_size, transforms, data_path, name, shuffle_val=False, add_path=False,
                additional_path=None):
    path = os.path.join(data_path, name)
    if not os.path.exists(path):
        raise ValueError("{1} data must be stored in {0}".format(path, name))

    if additional_path is None:
        set_folder = folder.ImageFolder(root=path, transform=transforms, add_path=add_path)
    else:
        set_folder = folder.MultiImageFolder(root=[path, additional_path], transform=transforms, add_path=add_path)
    loader = DataLoader(set_folder, batch_size=batch_size, shuffle=shuffle_val, num_workers=workers, pin_memory=True)

    return loader


def generate_loaders(workers, batch_size, transform_train, transform_test, data_path, dataset,
                     shuffle_val=False, add_path=False, additional_path=None):
    '''
    '''
    print(f"==> Preparing dataset {dataset}..")

    train_loader = make_loader(workers, batch_size, transform_train, data_path, 'train', True, add_path=add_path,
                               additional_path=additional_path)
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

    def make_loaders(self, workers, batch_size, shuffle_val=False, add_path=False, additional_path=None):
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
                                additional_path=additional_path)

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

    def __init__(self, data_path, **kwargs):
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
            'transform_train': transforms.Compose(train_tr),
            'transform_test': transforms.Compose(common_tr)
        }
        super(ImageNet9, self).__init__(ds_name,
                                        data_path, **ds_kwargs)


class ImageNet(DataSet):
    '''
    '''

    def __init__(self, data_path, **kwargs):
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
            'transform_train': transforms.Compose(train_tr),
            'transform_test': transforms.Compose(common_tr)
        }
        super(ImageNet, self).__init__(ds_name,
                                       data_path, **ds_kwargs)


class ImageNetBackgroundChangeAugmented(DataSet):
    '''
    '''

    def __init__(self, data_path, backgrounds_path, foregrounds_path, background_transform_chance, **kwargs):
        """
        """
        self.foregrounds_path = foregrounds_path
        backgrounds = []
        for background_path in [f'{backgrounds_path}/train']:
            for inner_dir in sorted(os.listdir(background_path)):
                for image_path in sorted(os.listdir(f'{background_path}/{inner_dir}')):
                    backgrounds.append(f'{background_path}/{inner_dir}/{image_path}')
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

    def make_loaders(self, workers, batch_size, shuffle_val=False, add_path=False, additional_path=None):
        '''
        '''
        if additional_path is None:
            additional_path = self.foregrounds_path
        transforms = self.transform_test
        return generate_loaders(workers=workers,
                                batch_size=batch_size,
                                transform_train=self.transform_train,
                                transform_test=self.transform_test,
                                data_path=self.data_path,
                                dataset=self.ds_name,
                                shuffle_val=shuffle_val,
                                add_path=add_path,
                                additional_path=additional_path)


class ImageNetBackgroundBlurAugmented(DataSet):
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

    def make_loaders(self, workers, batch_size, shuffle_val=False, add_path=False, additional_path=None):
        '''
        '''
        if additional_path is None:
            additional_path = self.backgrounds_path
        transforms = self.transform_test
        return generate_loaders(workers=workers,
                                batch_size=batch_size,
                                transform_train=self.transform_train,
                                transform_test=self.transform_test,
                                data_path=self.data_path,
                                dataset=self.ds_name,
                                shuffle_val=shuffle_val,
                                add_path=add_path,
                                additional_path=additional_path,
                                valid_path=self.default_path)
