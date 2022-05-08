import random
import torch
import torch.utils.data as data
from torch.utils.data import Dataset
from torchvision import transforms
from data.shared import set_background, default_loader, divide_paths
import torchvision.transforms.functional as TF

import os
import os.path
import sys


def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file
        extensions (iterable of strings): extensions to consider (lowercase)

    Returns:
        bool: True if the filename ends with one of given extensions
    """
    filename_lower = filename.lower()
    return any(filename_lower.endswith(ext) for ext in extensions)


def is_image_file(filename):
    """Checks if a file is an allowed image extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    return has_file_allowed_extension(filename, IMG_EXTENSIONS)


def make_dataset(dir, class_to_idx, extensions):
    images = []
    dir = os.path.expanduser(dir)
    for target in sorted(class_to_idx.keys()):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in sorted(fnames):
                if has_file_allowed_extension(fname, extensions):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)
    return images


class DatasetFolder(data.Dataset):
    """A generic data loader where the samples are arranged in this way: ::

        root/class_x/xxx.ext
        root/class_x/xxy.ext
        root/class_x/xxz.ext

        root/class_y/123.ext
        root/class_y/nsdf3.ext
        root/class_y/asd932_.ext

    Args:
        root (string): Root directory path.
        loader (callable): A function to load a sample given its path.
        extensions (list[string]): A list of allowed extensions.
        transform (callable, optional): A function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        target_transform (callable, optional): A function/transform that takes
            in the target and transforms it.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        samples (list): List of (sample path, class_index) tuples
        targets (list): The class_index value for each image in the dataset
    """

    def __init__(self, root, loader, extensions, transform=None,
                 target_transform=None, label_mapping=None, add_path=False):
        classes, class_to_idx = self._find_classes(root)
        if label_mapping is not None:
            classes, class_to_idx = label_mapping(classes, class_to_idx)

        samples = make_dataset(root, class_to_idx, extensions)
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(extensions)))

        self.root = root
        self.loader = loader
        self.extensions = extensions

        self.add_path = add_path
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.targets = [s[1] for s in samples]

        self.transform = transform
        self.target_transform = target_transform

    def _find_classes(self, dir):
        """
        Finds the class folders in a dataset.

        Args:
            dir (string): Root directory path.

        Returns:
            tuple: (classes, class_to_idx) where classes are relative to (dir), and class_to_idx is a dictionary.

        Ensures:
            No class is a subdirectory of another.
        """
        if sys.version_info >= (3, 5):
            # Faster and available in Python 3.5 and above
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in sorted(os.listdir(dir)) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)
           
        if self.add_path:
            return (path, sample), target
        else:
            return sample, target

    def __len__(self):
        return len(self.samples)

    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str


class DatasetMultifolder(DatasetFolder):
    def __init__(self, roots, loader, extensions, transform=None,
                 target_transform=None, label_mapping=None, add_path=False):
        super().__init__(roots[0], loader, extensions, transform, target_transform, label_mapping, add_path)
        self.roots = roots

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        background = self.loader(path.replace(self.roots[0].replace('train', ''), self.roots[1]))
        if self.transform is not None:
            sample = self.transform((sample, background, target))
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        if self.add_path:
            return (path, sample), target
        else:
            return sample, target


IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']


class ImageFolder(DatasetFolder):
    """A generic data loader where the images are arranged in this way: ::

        root/dog/xxx.png
        root/dog/xxy.png
        root/dog/xxz.png

        root/cat/123.png
        root/cat/nsdf3.png
        root/cat/asd932_.png

    Args:
        root (string): Root directory path.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        loader (callable, optional): A function to load an image given its path.

     Attributes:
        classes (list): List of the class names.
        class_to_idx (dict): Dict with items (class_name, class_index).
        imgs (list): List of (image path, class_index) tuples
    """
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, label_mapping=None, add_path=False):
        super(ImageFolder, self).__init__(root, loader, IMG_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform,
                                          label_mapping=label_mapping,
                                          add_path=add_path)
        self.imgs = self.samples


class MultiImageFolder(DatasetMultifolder):
    def __init__(self, root, transform=None, target_transform=None,
                 loader=default_loader, label_mapping=None, add_path=False):
        super().__init__(root, loader, IMG_EXTENSIONS,
                                          transform=transform,
                                          target_transform=target_transform,
                                          label_mapping=label_mapping,
                                          add_path=add_path)
        self.imgs = self.samples


class SwapBackgroundFolder(data.Dataset):
    def __init__(self, root, backgrounds, foregrounds, loader=default_loader, changed_backgrounds_count=8, pre_transform=None,
                 assigned_backgrounds_per_instance=None, random_seed=None, post_transform=None, target_transform=None, label_mapping=None, add_path=False):
        classes, class_to_idx = self._find_classes(root)
        if label_mapping is not None:
            classes, class_to_idx = label_mapping(classes, class_to_idx)

        samples = make_dataset(root, class_to_idx, IMG_EXTENSIONS)
        foregrounds = make_dataset(foregrounds, class_to_idx, IMG_EXTENSIONS)
        backgrounds = make_dataset(backgrounds, class_to_idx, IMG_EXTENSIONS)
        if assigned_backgrounds_per_instance is not None:
            backgrounds = divide_paths(foregrounds, backgrounds, assigned_backgrounds_per_instance, random_seed)
        
        if len(samples) == 0:
            raise(RuntimeError("Found 0 files in subfolders of: " + root + "\n"
                               "Supported extensions are: " + ",".join(IMG_EXTENSIONS)))
        
        self.root = root
        self.loader = default_loader

        self.add_path = add_path
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.samples = samples
        self.backgrounds = backgrounds
        self.foregrounds = foregrounds
        self.targets = [s[1] for s in samples]
        self.changed_backgrounds_count = changed_backgrounds_count
        self.assigned_backgrounds_per_instance = assigned_backgrounds_per_instance
        self.pre_transform = pre_transform
        self.post_transform = post_transform
        self.target_transform = target_transform
        self.state = True

    def _find_classes(self, dir):
        if sys.version_info >= (3, 5):
            classes = [d.name for d in os.scandir(dir) if d.is_dir()]
        else:
            classes = [d for d in sorted(os.listdir(dir)) if os.path.isdir(os.path.join(dir, d))]
        classes.sort()
        class_to_idx = {classes[i]: i for i in range(len(classes))}
        return classes, class_to_idx

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        foreground = self.loader(self.foregrounds[index][0])

        changed_backgrounds = []

        if self.pre_transform is not None:
            sample = self.pre_transform(sample)
        
        if self.target_transform is not None:
            target = self.target_transform(target)
        
        if self.state:
            for i in range(self.changed_backgrounds_count):
                background_set = self.backgrounds
                if self.assigned_backgrounds_per_instance is not None:
                    background_set = self.backgrounds[index]
                selected_background = background_set[int(len(background_set) * random.random())]
                background = self.loader(selected_background[0])
                background = TF.pil_to_tensor(background)
                changed_backgrounds.append(set_background(foreground, background)[0])
            changed_backgrounds = torch.stack(changed_backgrounds)
            minibatch = torch.cat((sample.unsqueeze(0), changed_backgrounds), dim=0)
        else:
            minibatch = sample
        
        if self.add_path:
            return (path, minibatch), target
        else:
            return minibatch, target

    def change_state(self, state):
        self.state = state
    
    def __len__(self):
        return len(self.samples)


class TensorDataset(Dataset):
    """Dataset wrapping tensors.

    Each sample will be retrieved by indexing tensors along the first dimension.

    Arguments:
        *tensors (Tensor): tensors that have the same size of the first dimension.
    """

    def __init__(self, *tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        im, targ = tuple(tensor[index] for tensor in self.tensors)

        if self.transform:
            real_transform = transforms.Compose([
                transforms.ToPILImage(),
                self.transform
            ])
            im = real_transform(im)

        return im, targ

    def __len__(self):
        return self.tensors[0].size(0)
