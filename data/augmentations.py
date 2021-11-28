import torch
from . import folder
import torchvision.transforms.functional as TF

class RandomBackgroundPerClass(object):
        def __init__(self, augment_chances, backgrounds_paths):
            self.augment_chances = augment_chances
            self.backgrounds_paths = backgrounds_paths
            
        def __call__(self, sample):
            if type(sample) is not tuple:
                return TF.to_tensor(sample)
            image, backgroundless, target = sample
            augmentation_chance, background_numer = torch.rand(2)
            if augmentation_chance < self.augment_chances[target]:
                image = backgroundless.copy()
                indices = image == 0
                background_id = int(background_numer*len(self.backgrounds_paths))
                background = TF.pil_to_tensor(folder.default_loader(self.backgrounds_paths[background_id]))
                image = TF.to_tensor(image)
                image[indices] = background[indices]
            else:
                image = TF.to_tensor(image)
            return image

class UnwrapTupled(object):
        def __call__(self, sample):
            if type(sample) is not tuple:
                return sample
            image, backgroundless, target = sample
            return image