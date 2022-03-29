import torch
from torch import nn
from torch.nn import functional as F
import pytorch_lightning as pl
import torchmetrics
from torchvision.utils import save_image, make_grid
from nflows.transforms.coupling import AffineCouplingTransform
from nflows import transforms, distributions, flows
from nflows.flows.base import Flow
from nflows.nn.nets import MLP
from nflows.transforms.base import CompositeTransform
from nflows.distributions.normal import ConditionalDiagonalNormal


class FlowableMLP(MLP):
    def forward(self, input, context):
        return super().forward(torch.cat((input, context), dim=1))


class ConditionalNICE(Flow):
    def __init__(self, features, hidden_sizes, num_layers, conditional_count):
        mask = torch.ones(features)
        layers = []
        mask[::2] = 0
        for _ in range(num_layers):
            layers.append(AffineCouplingTransform(mask=mask, transform_net_create_fn=lambda x, y: FlowableMLP(torch.zeros(x + features).shape, torch.zeros(y).shape, hidden_sizes)))
            mask = 1 - mask
        
        super().__init__(
            transform=CompositeTransform(layers),
            distribution=ConditionalDiagonalNormal(shape=[features], context_encoder=nn.Linear(conditional_count, 2 * features))
        )


class FlowModel(pl.LightningModule):
        
    def _default_y_selector(z):
        if len(z.shape) == 2:
            return z
        return z.mean(dim=1)

    def __init__(self, base_model, embedding_model, classifier_model, class_count, flow, use_flow=True, embedding_size=100, z_count=100, y_selector=_default_y_selector):
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
        self.use_flow = use_flow
    
    def get_context(self, embeddings):
        #class_context = torch.range(start=0, end=self.class_count, step=self.class_count / self.embedding_size) * self.class_count / self.embedding_size
        #class_context = nn.functional.one_hot(class_context.long()).type_as(embeddings).repeat((embeddings.shape[0], 1))
        #print(embeddings.shape)
        #print(class_context.shape)
        #context = torch.cat((embeddings, class_context), dim=1)
        embeddings = embeddings.detach()
        #embeddings.requires_grad = True
        embeddings.requires_grad = False
        return embeddings
    
    def forward_without_flow(self, x):
        x = self.base_model(x)
        x = self.embedding_model(x)
        ys = self.classifier_model(x)
        y = self.y_selector(ys)
        return y

    def forward(self, x):
        if self.use_flow:
            return self.forward_flow(x)
        else:
            return self.forward_without_flow(x), 0
    
    def forward_flow(self, x):
        shape = x.shape
        if len(shape) == 5:
            x = x.flatten(0, 1)
        x = self.base_model(x)
        x = self.embedding_model(x)
        if len(shape) == 5:
            x = torch.reshape(x, shape[0:2] + x.shape[1:])
        
        z, log_probs = self.generate_zs(x[:, 0], x) if len(x.shape) == 3 else self.generate_zs(x)
        ys = self.classifier_model(z)
        y = self.y_selector(ys)
        return F.softmax(y, dim=1), torch.mean(log_probs)
        
    def generate_zs(self, z0s, background_batch=None):
        context = self.get_context(z0s)
        if background_batch is not None:
            log_prob = self.flow.log_prob(background_batch.flatten(0, 1).detach(), context=context.repeat([background_batch.shape[1], 1]))
            log_prob = log_prob.mean()
            return background_batch, log_prob

        log_prob = torch.zeros(1).to(self.device)
        zs = self.flow.sample(self.z_count - 1, context=context)
        return torch.cat((z0s.unsqueeze(1), zs), dim=1), log_prob

    def training_step(self, train_batch, batch_idx):
        (path, x), y = train_batch
        y_hat, log_prob = self(x)
        loss = F.cross_entropy(y_hat, y) - log_prob
        self.log('train_loss', loss)
        self.log('train_acc', self.accuracy(y_hat, y), prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        (path, x), y = val_batch
        y_hat, log_prob = self(x)
        loss = F.cross_entropy(y_hat, y) - log_prob
        y_nf = self.forward_without_flow(x)
        nonflow_loss = F.cross_entropy(y_nf, y)
        self.log('val_loss', loss)
        self.log('val_loss_nf', nonflow_loss)
        self.log('val_acc', self.accuracy(y_hat, y), prog_bar=True)
        self.log('val_acc_nf', self.accuracy(y_nf, y), prog_bar=True)
        return loss

    def training_epoch_end(self, outs):
        self.log('train_acc_epoch', self.accuracy.compute())
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=0.0001)
        return optimizer
