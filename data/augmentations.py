import torch
from . import folder
import numpy as np
import torchvision.transforms.functional as TF
from torch.nn import functional as F
from pytorch_lightning.callbacks import Callback
from torchvision.utils import save_image


def join_background_image(background, image):
    indices = (image[0, :, :] < 0.02).logical_and(image[1, :, :] < 0.02).logical_and(image[2, :, :] < 0.02)
    image[:, indices] = (background / 255)[:, indices]
    return image


class RandomBackgroundChange(object):
        def __init__(self, augment_chances, backgrounds_paths):
            self.augment_chances = augment_chances
            self.backgrounds_paths = backgrounds_paths
            
        def __call__(self, sample):
            if type(sample) is not tuple:
                sample = TF.resize(sample, (224, 224))
                return TF.to_tensor(sample)
            image, backgroundless, target = sample
            augmentation_chance = torch.rand(1)

            augment_chance = self.augment_chances if type(self.augment_chances) is float else self.augment_chances[target]
            if augmentation_chance < augment_chance:
                image = backgroundless.copy()
                background_id = torch.randint(0, len(self.backgrounds_paths), (1, ))
                background = TF.pil_to_tensor(folder.default_loader(self.backgrounds_paths[background_id]))
                background = TF.resize(background, (224, 224))
                image = TF.resize(image, (224, 224))
                image = TF.to_tensor(image)
                image = join_background_image(background, image)
            else:
                image = TF.resize(image, (224, 224))
                image = TF.to_tensor(image)
            return image


class RandomBackgroundBlur(object):
        def __init__(self, augment_chances=None, kernel_sizes=[3, 5, 7, 11, 13, 17, 19]):
            if augment_chances is None:
                augment_chances = len(kernel_sizes) / (len(kernel_sizes) + 1)
            self.augment_chances = augment_chances
            self.kernel_sizes = kernel_sizes
            
        def __call__(self, sample):
            if type(sample) is not tuple:
                sample = TF.resize(sample, (224, 224))
                return TF.to_tensor(sample)
            
            image, background, target = sample
            augmentation_roll = torch.rand(1)
            background = TF.resize(background, (224, 224))
            augment_chance = self.augment_chances if type(self.augment_chances) is float else self.augment_chances[target]

            if augmentation_roll < augment_chance:
                kernel_idx = int(augmentation_roll * len(self.kernel_sizes) / augment_chance)
                background = TF.gaussian_blur(background, self.kernel_sizes[kernel_idx])
            background = TF.pil_to_tensor(background)
            image = TF.resize(image, (224, 224))
            image = TF.to_tensor(image)
            image = join_background_image(background, image)
            return image


class UpdateChancesBasedOnAccuracyCallback(Callback):
        def __init__(self, model, augmentation, background_test_count = 1., use_cuda = True, reverse_chances = False):
            self.augmentation = augmentation
            self.model = model
            self.background_test_count = background_test_count
            self.use_cuda = use_cuda
            self.reverse_chances = reverse_chances
            
        def on_validation_epoch_end(self, trainer, module):
            classes_count = len(self.augmentation.augment_chances)
            if self.background_test_count is None:
                return
            backgrounds = self.augmentation.backgrounds_paths
            image_count = int(self.background_test_count * len(backgrounds))
            background_indices = np.random.choice(np.array(range(len(backgrounds))), size=image_count, replace=False)
            images = []
            for bg_ind in background_indices:
                background = TF.pil_to_tensor(folder.default_loader(backgrounds[bg_ind]))
                if self.use_cuda:
                    background = background.cuda()
                background = TF.resize(background.float()/255, (224, 224))
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
                if self.reverse_chances:
                    self.augmentation.augment_chances[i] = 1 - corrects[i] / counts[i]
                else:
                    self.augmentation.augment_chances[i] = corrects[i] / counts[i]


def calculate_entropy(dataloader, model, use_cuda):
    entropy = {}
    with torch.no_grad():
        for X_batch, y_batch in dataloader:
            path, sample, indices = X_batch
            y_pred = model(X_batch if not use_cuda else X_batch.cuda())
            entropy = F.cross_entropy(y_pred if not use_cuda else y_pred.to('cpu'), y_batch)
            for loss, index in zip(entropy, indices):
                if index not in entropy:
                    entropy[index] = []
                    entropy[index].append(loss)
    return entropy


class UpdateChancesBasedOnPerSampleEntropyCallback(Callback):
        def __init__(self, dataloader, model, augmentation, recalculate_on_end = False, use_cuda = True, reverse_chances = False):
            self.augmentation = augmentation
            self.dataloader = dataloader
            self.model = model
            self.use_cuda = use_cuda
            self.recalculate_on_end = recalculate_on_end
            self.reverse_chances = reverse_chances
            
        def on_validation_epoch_end(self, trainer, module):
            if not self.recalculate_on_end and not hasattr(self.model, 'entropy'):
                return
            
            if not self.recalculate_on_end:
                entropy = self.model.entropy
            else:
                entropy = calculate_entropy(self.dataloader, self.model, self.use_cuda)
            augment_chances = {}
            for i in entropy:
                if self.reverse_chances:
                    augment_chances[i] = 1 - np.array(entropy[i]).mean()
                else:
                    augment_chances[i] = np.array(entropy[i]).mean()
            
            self.augmentation.augment_chances = augment_chances


class UpdateChancesBasedOnEntropyTakeTopCallback(Callback):
        def __init__(self, dataloader, model, augmentation, top_to_take, recalculate_on_end = False, use_cuda = True, reverse_chances = False):
            self.augmentation = augmentation
            self.dataloader = dataloader
            self.model = model
            self.use_cuda = use_cuda
            self.recalculate_on_end = recalculate_on_end
            self.top_to_take = top_to_take
            self.reverse_chances = reverse_chances
            
        def on_validation_epoch_end(self, trainer, module):
            if not self.recalculate_on_end and not hasattr(self.model, 'entropy'):
                return
            
            if not self.recalculate_on_end:
                entropy = self.model.entropy
            else:
                entropy = calculate_entropy(self.dataloader, self.model, self.use_cuda)
            average_entropy = {}
            for i in entropy:
                average_entropy[i] = np.array(entropy[i]).mean()
            
            taken = sorted(average_entropy.keys(), lambda k : average_entropy[k], reverse=not reverse_chances)[0:int(len(average_entropy) * self.top_to_take)]
            augment_chances = {}
            for i in average_entropy:
                if i in taken:
                    augment_chances[i] = 1.
                else:
                    augment_chances[i] = 0.
            
            self.augmentation.augment_chances = augment_chances


class UnwrapTupled(object):
        def __call__(self, sample):
            if type(sample) is not tuple:
                return sample
            image, backgroundless, target = sample
            return image
