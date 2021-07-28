import torchvision
import numpy as np
import warnings


class OnePixelMNIST:
    def __init__(self,
                 train: bool = True,
                 permute: bool = True,
                 dataset: torchvision.datasets.VisionDataset = None,
                 **kwargs):
        super().__init__(**kwargs)
        self.train = train
        self.permute = permute
        self.dataset = dataset
        if self.train and self.permute:
            warnings.warn('setting permute on training data has no effect')

    def __getitem__(self, item):
        image, target = self.dataset.__getitem__(item)
        if self.train:
            image[0, target] = 255
        elif self.permute:
            image[0, np.random.randint(0, 9)] = 255
        return image, target

    def __len__(self):
        return len(self.dataset)
