import torch
from torch.nn import functional as F
import pytorch_lightning as pl
import torchmetrics


class TIMMModel(pl.LightningModule):
    def __init__(self, timm_model):
        super().__init__()
        self.model = timm_model
        self.accuracy = torchmetrics.Accuracy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, train_batch, batch_idx):
        (path, x), y = train_batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        self.log('train_acc', self.accuracy(y_hat, y), prog_bar=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        (path, x), y = val_batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss)
        self.log('val_acc', self.accuracy(y_hat, y), prog_bar=True)
        return loss

    def training_epoch_end(self, outs):
        self.log('train_acc_epoch', self.accuracy.compute())
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1, weight_decay=1e-4, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

        return [optimizer], [scheduler]
