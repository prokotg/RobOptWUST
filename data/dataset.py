import torchvision
import numpy as np
import warnings
import cv2


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
            image[0, 0, target] = 12
        elif self.permute:
            image[0, 0, np.random.randint(0, 9)] = 12
        return image, target

    def __len__(self):
        return len(self.dataset)


class BGColouredMNIST:
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

        self.palette = {
            0: (255, 0, 0), 3: (255, 211, 10), 6: (10, 239, 255), 9: (190, 10, 255),
            1: (255, 135, 0), 4: (161, 211, 10), 7: (20, 125, 245),
            2: (255, 211, 0), 5: (10, 255, 153), 8: (88, 10, 255),
        }

    def __getitem__(self, item):
        gray_img, target = self.dataset.__getitem__(item)
        color_img = cv2.cvtColor(gray_img.numpy()[0], cv2.COLOR_GRAY2RGB)*255

        bg_mask = (gray_img != 0.0)[0]

        if self.train:
            color_img[bg_mask] = self.palette[target]
        elif self.permute:
            color_img[bg_mask] = self.palette[np.random.randint(0, 9)]

        color_img = np.transpose(color_img, (2, 0, 1))
        return color_img, target, bg_mask

    def __len__(self):
        return len(self.dataset)
