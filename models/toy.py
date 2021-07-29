import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import torchmetrics


class MNISTToyModel(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        # self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

        self.accuracy = torchmetrics.Accuracy()

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d((self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x)

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


class MNISTToyModelGrad(MNISTToyModel):
    def __init__(self):
        super().__init__()
        self.automatic_optimization = False

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

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        self.log('val_loss', loss)
        self.log('val_acc', self.accuracy(y_hat, y), prog_bar=True)

        return loss

