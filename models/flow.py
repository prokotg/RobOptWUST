import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import torchmetrics
from nflows.transforms.coupling import AffineCouplingTransform
from nflows import transforms, distributions, flows
from nflows.flows.base import Flow
from nflows.nn.nets import MLP
from nflows.transforms.base import CompositeTransform
from nflows.distributions.normal import ConditionalDiagonalNormal


class ConditionalNICE(Flow):
    def __init__(self, features, hidden_sizes, num_layers, conditional_count):
        mask = torch.ones(features)
        layers = []
        mask[::2] = -1
        for _ in range(num_layers):
            layers.append(AffineCouplingTransform(mask=mask, transform_net_create_fn=lambda x, y: MLP(torch.zeros(x).shape, torch.zeros(y).shape, hidden_sizes)))
            mask *= -1
        
        super().__init__(
            transform=CompositeTransform(layers),
            distribution=ConditionalDiagonalNormal(shape=[features], context_encoder=nn.Linear(conditional_count, 2 * features))
        )


class EmbeddingNICEModel(pl.LightningModule):
    def __init__(self, base_model, flow, num_classes, zs_count, care_about_path=True):
        self.base_model = base_model
        self.flow = flow
        self.num_classes = num_classes
        self.zs_count = zs_count
        self.care_about_path = care_about_path

    def forward(self, x):
        if self.care_about_path:
            path, x = x
        embeddings = self.base_model(x)
        zs = self.flow.sample(self.zs_count - 1, context=embeddings)
        return torch.cat((x.unsqueeze(0), zs), dim=0)
        
    def training_step(self, train_batch, batch_idx):
        if self.care_about_path:
            (path, x, swapped), context = train_batch
        else:
            (x, swapped), context = train_batch
        context = self.base_model(x)
        inputs = self.base_model(swapped)
        loss = -self.flow.log_prob(inputs=inputs, context=context)
        self.log('train_loss', loss.mean())
        return loss


class FlowModel(pl.LightningModule):
        
    def _default_y_selector(z):
        return torch.sum(z, dim=2)

    def __init__(self, base_model, embedding_model, classifier_model, class_count, flow, embedding_size=100, z_count=100, y_selector=_default_y_selector):
        super().__init__()
        self.base_model = base_model
        self.embedding_model = embedding_model
        self.classifier_model = classifier_model
        self.accuracy = torchmetrics.Accuracy()
        self.z_count = z_count
        self.y_selector = y_selector
        self.class_count = class_count
        self.embedding_size = embedding_size
        self.flow = flow
    
    def get_context(self, embeddings):
        #class_context = torch.range(start=0, end=self.class_count, step=self.class_count / self.embedding_size) * self.class_count / self.embedding_size
        #class_context = nn.functional.one_hot(class_context.long()).type_as(embeddings).repeat((embeddings.shape[0], 1))
        #print(embeddings.shape)
        #print(class_context.shape)
        #context = torch.cat((embeddings, class_context), dim=1)
        return embeddings
    
    def forward(self, x):
        x = self.base_model(x)
        x = self.embedding_model(x)
        z = self.generate_zs(x)
        ys = self.classifier_model(z)
        y = self.y_selector(ys)
        return y
    
    def generate_zs(self, z0):
        context = self.get_context(z0)
        #zs = self.flow.sample(self.z_count - 1, context=context)
        return z0.unsqueeze(1) #torch.cat((z0.unsqueeze(0), zs), dim=0)

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
