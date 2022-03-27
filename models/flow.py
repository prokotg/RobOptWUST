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


class FlowableMLP(MLP):
    def forward(self, input, context):
        return super().forward(torch.cat((input, context), dim=1))


class ConditionalNICE(Flow):
    def __init__(self, features, hidden_sizes, num_layers, conditional_count):
        mask = torch.ones(features)
        layers = []
        mask[::2] = -1
        for _ in range(num_layers):
            layers.append(AffineCouplingTransform(mask=mask, transform_net_create_fn=lambda x, y: FlowableMLP(torch.zeros(x + features).shape, torch.zeros(y).shape, hidden_sizes)))
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
        if len(z.shape) == 2:
            return z
        return z.sum(dim=1)

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
        shape = x.shape
        if len(shape) == 5:
            x = x.flatten(0, 1)
        x = self.base_model(x)
        x = self.embedding_model(x)
        if len(shape) == 5:
            x = torch.reshape(x, shape[0:2] + x.shape[1:])
        
        log_prob = None
        result = []
        for x_ in x:
            z, l_p = self.generate_zs(x_[0], x_) if len(x_.shape) == 2 else self.generate_zs(x_)
            ys = self.classifier_model(z)
            y = self.y_selector(ys)
            result.append(y)
            if log_prob is None:
                log_prob = l_p
            else:
                log_prob += l_p
        return F.softmax(torch.stack(result).sum(dim=1), dim=1), log_prob
        
    def generate_zs(self, z0, additional_examples=None):
        context = self.get_context(z0)
        if additional_examples is not None:
            log_prob = self.flow.log_prob(additional_examples, context=context.repeat([len(additional_examples), 1]))
            log_prob = log_prob.sum()
        else:
            log_prob = 0
        zs = self.flow.sample(self.z_count - 1, context=context.unsqueeze(0))
        return torch.cat((z0.unsqueeze(0), zs.squeeze(0)), dim=0), log_prob

    def training_step(self, train_batch, batch_idx):
        (path, x), y = train_batch
        y_hat, log_prob = self(x)
        loss = - log_prob * 0.000001 + F.cross_entropy(y_hat, y)
        self.log('train_loss', loss)
        self.log('train_acc', self.accuracy(y_hat, y), prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        (path, x), y = val_batch
        y_hat, log_prob = self(x)
        loss = F.cross_entropy(y_hat, y) - log_prob * 0.00001
        self.log('val_loss', loss)
        self.log('val_acc', self.accuracy(y_hat, y), prog_bar=True)
        return loss

    def training_epoch_end(self, outs):
        self.log('train_acc_epoch', self.accuracy.compute())
    
    def configure_optimizers(self):
        optimizer = torch.optim.SGD(self.parameters(), lr=0.1, weight_decay=1e-4, momentum=0.9)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

        return [optimizer], [scheduler]
