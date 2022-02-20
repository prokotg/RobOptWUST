import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import torchmetrics
import time


class TIMMModel(pl.LightningModule):
    def __init__(self, timm_model, save_entropy=False):
        super().__init__()
        self.model = timm_model
        self.accuracy = torchmetrics.Accuracy()
        self.save_entropy = save_entropy

    def forward(self, x):
        return self.model(x)

    def training_step(self, train_batch, batch_idx):
        if self.save_entropy:
            (path, x, indices), y = train_batch
        else:
            (path, x), y = train_batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y, reduction='none')
        self.log('train_loss', loss.mean())
        self.log('train_acc', self.accuracy(y_hat, y), prog_bar=True)
        return {'loss' : loss.mean(), 'indices' : indices, 'losses' : loss.detach()}

    def validation_step(self, val_batch, batch_idx):
        (path, x), y = val_batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss)
        self.log('val_acc', self.accuracy(y_hat, y), prog_bar=True)
        return loss

    def training_epoch_end(self, outs):
        self.log('train_acc_epoch', self.accuracy.compute())
        if self.save_entropy:
            self.entropy = {}
            for out in outs:
                print(out)
                for loss, index in zip(out['losses'], out['indices']):
                    if index not in self.entropy:
                        self.entropy[index] = []
                    self.entropy[index].append(loss)
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1, weight_decay=1e-4, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

        return [optimizer], [scheduler]
