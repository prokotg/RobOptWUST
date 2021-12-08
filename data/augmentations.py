import torch
from . import folder
import numpy as np
import torchvision.transforms.functional as TF
from pytorch_lightning.callbacks import Callback
from torchvision.utils import save_image

class RandomBackgroundPerClass(object):
        def __init__(self, augment_chances, backgrounds_paths):
            self.augment_chances = augment_chances
            self.backgrounds_paths = backgrounds_paths
            
        def __call__(self, sample):
            if type(sample) is not tuple:
                sample = TF.resize(sample, (224, 224))
                return TF.to_tensor(sample)
            image, backgroundless, target = sample
            augmentation_chance, background_numer = torch.rand(2)
            if augmentation_chance < self.augment_chances[target]:
                image = backgroundless.copy()
                background_id = int(background_numer*len(self.backgrounds_paths))
                background = TF.pil_to_tensor(folder.default_loader(self.backgrounds_paths[background_id]))
                background = TF.resize(background, (224, 224))
                image = TF.resize(image, (224, 224))
                image = TF.to_tensor(image)
                indices = image == 0
                image[indices] = (background.float()/255)[indices]
            else:
                image = TF.resize(image, (224, 224))
                image = TF.to_tensor(image)
            return image

class UpdateChancesBasedOnAccuracyCallback(Callback):
        def __init__(self, model, augmentation, background_test_count = 1.):
            self.augmentation = augmentation
            self.model = model
            self.background_test_count = background_test_count
            
        def on_validation_epoch_end(self, trainer, module):
            classes_count = len(self.augmentation.augment_chances)
            if self.background_test_count is None:
                return
            backgrounds = self.augmentation.backgrounds_paths
            image_count = int(self.background_test_count * len(backgrounds))
            background_indices = np.random.choice(np.array(range(len(backgrounds))), size=image_count)
            images = []
            for bg_ind in background_indices:
                background = TF.pil_to_tensor(folder.default_loader(backgrounds[bg_ind]))
                background = TF.resize(background.float()/255, (224, 224)).cuda()
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