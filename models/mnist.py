import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import torchmetrics
from overrides import overrides


class MNISTPlainModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(1, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )

        self.accuracy = torchmetrics.Accuracy()

    def forward(self, x):
        features = self.net(x)
        return F.log_softmax(features)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        self.log('train_acc', self.accuracy(y_hat, y), prog_bar=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss)
        self.log('val_acc', self.accuracy(y_hat, y), prog_bar=True)

        return loss

    def training_epoch_end(self, outs):
        # log epoch metric
        self.log('train_acc_epoch', self.accuracy.compute())


class MNISTPlainModelGrad(MNISTPlainModel):
    def __init__(self):
        super().__init__()
        self.automatic_optimization = False

    @overrides
    def training_step(self, train_batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()
        x, y = train_batch
        x.requires_grad = True
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        batch_size = x.shape[0]
        grad_y_hat_x, = torch.autograd.grad(loss, x, retain_graph=True, create_graph=True)
        grad_y_hat_x_confouding = grad_y_hat_x.reshape(batch_size, -1)[list(range(batch_size)), y]
        loss = loss + 1e8 * torch.linalg.vector_norm(grad_y_hat_x_confouding)

        self.manual_backward(loss, retain_graph=True)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.accuracy(y_hat, y), prog_bar=True)
        self.log('train_grad_norm', torch.linalg.vector_norm(grad_y_hat_x), prog_bar=True)
        self.log('train_grad_norm_confouding', torch.linalg.vector_norm(grad_y_hat_x_confouding), prog_bar=True)
        opt.step()
        return loss

    @overrides
    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss)
        self.log('val_acc', self.accuracy(y_hat, y), prog_bar=True)
        return loss


class ColoredMNISTPlainModel(MNISTPlainModel):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )

    def training_step(self, train_batch, batch_idx):
        x, y, mask = train_batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        self.log('train_acc', self.accuracy(y_hat, y), prog_bar=True)

        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y, mask = val_batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss)
        self.log('val_acc', self.accuracy(y_hat, y), prog_bar=True)

        return loss


class ColoredMNISTPlainModelGrad(MNISTPlainModelGrad):
    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )

    def training_step(self, train_batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()
        x, y, mask = train_batch
        x.requires_grad = True
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        grad_y_hat_x, = torch.autograd.grad(loss, x, retain_graph=True, create_graph=True)
        grad_y_hat_x_confouding = grad_y_hat_x[mask.unsqueeze(1).repeat(1, 3, 1, 1)]
        # grad_y_hat_x_confouding = grad_y_hat_x[:, :, 0:15, 0:15].flatten() # penalizing certain region
        loss = loss + 1e7 * grad_y_hat_x_confouding.norm(2)

        self.manual_backward(loss, retain_graph=True)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.accuracy(y_hat, y), prog_bar=True)
        self.log('train_grad_norm', torch.linalg.vector_norm(grad_y_hat_x), prog_bar=True)
        self.log('train_grad_norm_confouding', torch.linalg.vector_norm(grad_y_hat_x_confouding), prog_bar=True)
        opt.step()
        return loss

    @overrides
    def validation_step(self, val_batch, batch_idx):
        x, y, mask = val_batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss)
        self.log('val_acc', self.accuracy(y_hat, y), prog_bar=True)
        return loss


class ColoredMNISTPlainModelGradProp(MNISTPlainModelGrad):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 10, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(10, 20, kernel_size=5),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(320, 50),
            nn.ReLU(),
            nn.Linear(50, 10)
        )

    def training_step(self, train_batch, batch_idx):
        opt = self.optimizers()
        opt.zero_grad()
        x, y, mask = train_batch
        x.requires_grad = True
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        grad_y_hat_x, = torch.autograd.grad(loss, x, retain_graph=True, create_graph=True)
        grad_y_hat_x_confouding = grad_y_hat_x[mask.unsqueeze(1).repeat(1, 3, 1, 1)]
        grad_y_hat_x_object = grad_y_hat_x[~mask.unsqueeze(1).repeat(1, 3, 1, 1)]
        loss = loss + torch.abs(
            (torch.linalg.vector_norm(grad_y_hat_x_confouding) / torch.linalg.vector_norm(grad_y_hat_x_object) - 1))
        loss = loss + torch.abs(
            (torch.linalg.vector_norm(grad_y_hat_x_confouding) / torch.linalg.vector_norm(grad_y_hat_x_object) - 1))

        self.manual_backward(loss, retain_graph=True)
        self.log('train_loss', loss, prog_bar=True)
        self.log('train_acc', self.accuracy(y_hat, y), prog_bar=True)
        self.log('train_grad_norm', torch.linalg.vector_norm(grad_y_hat_x), prog_bar=True)
        self.log('train_grad_norm_confouding', torch.linalg.vector_norm(grad_y_hat_x_confouding), prog_bar=True)
        self.log('train_grad_norm_object', torch.linalg.vector_norm(grad_y_hat_x_object), prog_bar=True)
        self.log('grad_norm_prop',
                 torch.linalg.vector_norm(grad_y_hat_x_object) / torch.linalg.vector_norm(grad_y_hat_x_confouding),
                 prog_bar=True)
        opt.step()
        return loss

    @overrides
    def validation_step(self, val_batch, batch_idx):
        x, y, mask = val_batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)
        self.log('val_loss', loss)
        self.log('val_acc', self.accuracy(y_hat, y), prog_bar=True)
        return loss
