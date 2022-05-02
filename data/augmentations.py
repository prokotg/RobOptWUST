import os

import torch
import numpy as np
import torchvision.transforms.functional as TF
from pytorch_lightning.callbacks import Callback
from data import shared


class RandomBackgroundPerClass(object):
    def __init__(self, augment_chances, backgrounds_paths, return_metadata=False):
        self.augment_chances = augment_chances
        self.backgrounds_paths = backgrounds_paths
        self.return_metadata = return_metadata

    def __call__(self, sample):
        if type(sample) is not tuple:
            sample = TF.resize(sample, (224, 224))
            return TF.to_tensor(sample)
        image, backgroundless, target = sample
        augmentation_chance = torch.rand(1)
        mask = np.zeros_like(image)
        background_class = -1
        if augmentation_chance < self.augment_chances[target]:
            image = backgroundless.copy()
            background_id = torch.randint(0, len(self.backgrounds_paths), (1,))
            #  example background path '...\only_bg_t\\only_bg_t\\/train/00_dog/n02085936_2693.JPEG'
            background_class = int(os.path.split(os.path.dirname(self.backgrounds_paths[background_id]))[-1].split('_')[0])
            background = TF.pil_to_tensor(shared.default_loader(self.backgrounds_paths[background_id]))
            image, _ = set_background(image, background)
        else:
            image = TF.resize(image, (224, 224))
            image = TF.to_tensor(image)
        if self.return_metadata:
            return image, mask, background_class
        return image


class RandomBackgroundBlur(object):
    def __init__(self, augment_chances):
        self.augment_chances = augment_chances

    def __call__(self, sample):
        if type(sample) is not tuple:
            sample = TF.resize(sample, (224, 224))
            return TF.to_tensor(sample)
        image, background, target = sample
        augmentation_chance = torch.rand(1)
        background = TF.resize(background, (224, 224))
        if augmentation_chance < self.augment_chances[target]:
            background = TF.gaussian_blur(background, 7)
        background = TF.pil_to_tensor(background)
        image = TF.resize(image, (224, 224))
        image = TF.to_tensor(image)
        indices = (image[0, :, :] < 0.02).logical_and(image[1, :, :] < 0.02).logical_and(image[2, :, :] < 0.02)
        image[:, indices] = (background / 255)[:, indices]
        return image


class UpdateChancesBasedOnAccuracyCallback(Callback):
    def __init__(self, model, augmentation, background_test_count=1., use_cuda=True):
        self.augmentation = augmentation
        self.model = model
        self.background_test_count = background_test_count
        self.use_cuda = use_cuda

    def on_validation_epoch_end(self, trainer, module):
        classes_count = len(self.augmentation.augment_chances)
        if self.background_test_count is None:
            return
        backgrounds = self.augmentation.backgrounds_paths
        image_count = int(self.background_test_count * len(backgrounds))
        background_indices = np.random.choice(np.array(range(len(backgrounds))), size=image_count, replace=False)
        images = []
        for bg_ind in background_indices:
            background = TF.pil_to_tensor(shared.default_loader(backgrounds[bg_ind]))
            if self.use_cuda:
                background = background.cuda()
            background = TF.resize(background.float() / 255, (224, 224))
            images.append(background)

        images = torch.stack(images).chunk(4)
        true_y = np.array_split((background_indices / (len(backgrounds) / classes_count)).astype('int'), 4)
        corrects = [0] * classes_count
        counts = [0] * classes_count
        for im, tr_y in zip(images, true_y):
            ans = self.model(im, )
            for a, y in zip(ans, tr_y):
                counts[y] += 1
                if torch.argmax(a) == y:
                    corrects[y] += 1

        for i in range(classes_count):
            self.augmentation.augment_chances[i] = corrects[i] / counts[i]


class UnwrapTupled(object):
    def __call__(self, sample):
        if type(sample) is not tuple:
            return sample
        image, backgroundless, target = sample
        return image
