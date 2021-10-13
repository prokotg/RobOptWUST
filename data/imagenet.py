import torch as ch
import os
from torchvision import transforms
from . import folder
from torch.utils.data import DataLoader

def make_loader(workers, batch_size, transforms, data_path, name, shuffle_val=False):
    path = os.path.join(data_path, name)
    if not os.path.exists(path):
        raise ValueError("{1} data must be stored in {0}".format(path, name))

    set_folder = folder.ImageFolder(root=path, transform=transforms)
    loader = DataLoader(set_folder, batch_size=batch_size, shuffle=shuffle_val, num_workers=workers, pin_memory=True)

    return loader

def make_loaders(workers, batch_size, transforms, data_path, dataset, shuffle_val=False):
    '''
    '''
    print(f"==> Preparing dataset {dataset}..")
    
    train_loader = make_loader(workers, batch_size, transforms, data_path, 'train', True)
    test_loader = make_loader(workers, batch_size, transforms, data_path, 'val', False)
    return train_loader, test_loader


class DataSet(object):
    '''
    '''

    def __init__(self, ds_name, data_path, **kwargs):
        """
        """
        required_args = ['num_classes', 'mean', 'std', 'transform_test']
        assert set(kwargs.keys()) == set(required_args), "Missing required args, only saw %s" % kwargs.keys()
        self.ds_name = ds_name
        self.data_path = data_path
        self.__dict__.update(kwargs)

    def make_loaders(self, workers, batch_size, shuffle_val=False):
        '''
        '''
        transforms = self.transform_test
        return make_loaders( workers=workers,
                                batch_size=batch_size,
                                transforms=transforms,
                                data_path=self.data_path,
                                dataset=self.ds_name,
                                shuffle_val=shuffle_val)
    
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
        ds_kwargs = {
            'num_classes': 9,
            'mean': ch.tensor([0.4717, 0.4499, 0.3837]), 
            'std': ch.tensor([0.2600, 0.2516, 0.2575]),
            'transform_test': transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
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
        ds_kwargs = {
            'num_classes': 1000,
            'mean': ch.tensor([0.485, 0.456, 0.406]),
            'std': ch.tensor([0.229, 0.224, 0.225]),
            'transform_test': transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
        }
        super(ImageNet, self).__init__(ds_name,
                data_path, **ds_kwargs)
