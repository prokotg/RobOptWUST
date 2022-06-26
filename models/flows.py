import torch
from torch import nn
from torch.nn import functional as F
from nflows.transforms.coupling import AffineCouplingTransform
from nflows.flows.base import Flow
from nflows.nn.nets import MLP
from nflows.distributions.normal import ConditionalDiagonalNormal
from nflows.transforms.autoregressive import MaskedAffineAutoregressiveTransform
from nflows.transforms.base import CompositeTransform
from nflows.transforms.normalization import BatchNorm
from nflows.transforms.permutations import RandomPermutation, ReversePermutation


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


class ConditionalMAF(Flow):
    def __init__(self, features, hidden_features, num_layers, num_blocks_per_layer, conditional_count,
                 use_residual_blocks=True, use_random_masks=False, use_random_permutations=False, activation=F.relu,
                 dropout_probability=0.0, batch_norm_within_layers=False, batch_norm_between_layers=False):
        if use_random_permutations:
            permutation_constructor = RandomPermutation
        else:
            permutation_constructor = ReversePermutation

        layers = []
        for _ in range(num_layers):
            layers.append(permutation_constructor(features))
            layers.append(
                MaskedAffineAutoregressiveTransform(
                    features=features,
                    context_features=conditional_count,
                    hidden_features=hidden_features,
                    num_blocks=num_blocks_per_layer,
                    use_residual_blocks=use_residual_blocks,
                    random_mask=use_random_masks,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=batch_norm_within_layers,
                )
            )
            if batch_norm_between_layers:
                layers.append(BatchNorm(features))

        super().__init__(
            transform=CompositeTransform(layers),
            distribution=ConditionalDiagonalNormal(shape=[features], context_encoder=nn.Linear(conditional_count, 2 * features))
        )


class MAF(Flow):
    def __init__(self, features, hidden_features, num_layers, num_blocks_per_layer, conditional_count,
                 use_residual_blocks=True, use_random_masks=False, use_random_permutations=False, activation=F.relu,
                 dropout_probability=0.0, batch_norm_within_layers=False, batch_norm_between_layers=False):
        if use_random_permutations:
            permutation_constructor = RandomPermutation
        else:
            permutation_constructor = ReversePermutation

        layers = []
        for _ in range(num_layers):
            layers.append(permutation_constructor(features))
            layers.append(
                MaskedAffineAutoregressiveTransform(
                    features=features,
                    context_features=conditional_count,
                    hidden_features=hidden_features,
                    num_blocks=num_blocks_per_layer,
                    use_residual_blocks=use_residual_blocks,
                    random_mask=use_random_masks,
                    activation=activation,
                    dropout_probability=dropout_probability,
                    use_batch_norm=batch_norm_within_layers,
                )
            )
            if batch_norm_between_layers:
                layers.append(BatchNorm(features))

        super().__init__(
            transform=CompositeTransform(layers),
            distribution=ConditionalDiagonalNormal(shape=[features], context_encoder=nn.Linear(conditional_count, 2 * features))
        )
