import torch
from torch import nn
from typing import Optional
from .utils import (
    masked_mean,
    masked_variance,
    masked_min,
    masked_max,
)


class InputNet(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout_rate: float = 0.0,
    ):
        super(InputNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.layers = nn.Sequential()
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.Dropout(p=dropout_rate))
        self.layers.append(nn.Linear(hidden_dim, output_dim))

    def forward(self, x):
        output = self.layers(x)
        return output


class MolecularAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 1,
        add_mlp: bool = False,
        dim_feedforward: int = 250,
        dropout_rate: float = 0.0,
    ):
        super(MolecularAttention, self).__init__()

        self.self_attn_layer = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )

        self.norm1 = nn.LayerNorm(embed_dim)

        self.add_mlp = add_mlp
        if self.add_mlp:
            self.linear_net = nn.Sequential(
                nn.Linear(embed_dim, dim_feedforward),
                nn.Dropout(dropout_rate),
                nn.ReLU(inplace=True),
                nn.Linear(dim_feedforward, embed_dim)
            )
            self.norm2 = nn.LayerNorm(embed_dim)
            self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x, key_padding_mask):

        # Self-attention
        attn_x, _ = self.self_attn_layer(
            query=x,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )

        # Residual connection
        x = x + attn_x
        x = self.norm1(x)

        # MLP
        if self.add_mlp:
            x = self.linear_net(x)

            # Residual connection
            x = x + self.dropout(x)
            x = self.norm2(x)

        return x


class MeanAggregation(nn.Module):
    def __init__(self):
        super(MeanAggregation, self).__init__()

    def forward(self, x, key_padding_mask):
        # Masked average
        global_emb = masked_mean(x, ~key_padding_mask)

        global_emb = global_emb.squeeze(1)

        return global_emb


class AttentionAggregation(nn.Module):
    def __init__(
        self,
        embed_dim: int = 196,
        num_heads: int = 1,
        dropout_rate: float = 0.0,
    ):
        super(AttentionAggregation, self).__init__()

        self.cross_attn_layer = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout_rate,
            batch_first=True,
        )

    def forward(self, x, key_padding_mask):
        # Masked average
        global_emb = masked_mean(x, ~key_padding_mask)
        global_emb = global_emb.unsqueeze(1)

        # Cross-attention
        global_emb, _ = self.cross_attn_layer(
            query=global_emb,
            key=x,
            value=x,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        global_emb = global_emb.squeeze(1)

        return global_emb


class PrincipalNeighborhoodAggregation(nn.Module):
    def __init__(self):
        super(PrincipalNeighborhoodAggregation, self).__init__()

    def forward(self, x, key_padding_mask):
        mean = masked_mean(x, ~key_padding_mask).unsqueeze(1)
        var = masked_variance(x, ~key_padding_mask).unsqueeze(1)
        minimum = masked_min(x, ~key_padding_mask).unsqueeze(1)
        maximum = masked_max(x, ~key_padding_mask).unsqueeze(1)

        global_emb = torch.cat(
            (mean, var, minimum, maximum), dim=-1
        )

        global_emb = global_emb.squeeze(1)

        return global_emb


class MixtureNet(nn.Module):
    def __init__(
        self,
        mol_aggregation: nn.Module,
        num_layers: int = 1,
        **mol_attn_args,
    ):
        super(MixtureNet, self).__init__()

        layers = [MolecularAttention(**mol_attn_args) for _ in range(num_layers)] if num_layers > 0 else [nn.Identity()]
        self.mol_attn_layers = nn.ModuleList(layers)
        self.mol_aggregation = mol_aggregation

    def forward(self, x, key_padding_mask):
        for layer in self.mol_attn_layers:
            if isinstance(layer, nn.Identity):
                x = layer(x)
            else:
                x = layer(x, key_padding_mask)

        global_emb = self.mol_aggregation(x, key_padding_mask)

        return global_emb


class Regressor(nn.Module):
    def __init__(
        self,
        hidden_dim: int,
        output_dim: int,
        num_layers: int = 2,
        dropout_rate: float = 0.0,
    ):
        super(Regressor, self).__init__()
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        self.dropout_rate = dropout_rate

        self.layers = nn.Sequential()
        for _ in range(num_layers - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.layers.append(nn.ReLU())
            self.layers.append(nn.LayerNorm(hidden_dim))
            self.layers.append(nn.Dropout(p=dropout_rate))
        self.layers.append(nn.Linear(hidden_dim, output_dim))
        # self.layers.append(nn.Sigmoid())

    def forward(self, x):
        output = self.layers(x)
        return output


class Chemix(nn.Module):
    def __init__(
        self,
        regressor: nn.Module,
        mixture_net: nn.Module,
        unk_token: int = -999,
        aggr: str = 'concat',
        input_net: Optional[nn.Module] = None,
    ):
        super(Chemix, self).__init__()
        self.input_net = input_net
        self.mixture_net = mixture_net
        self.regressor = regressor
        self.unk_token = unk_token
        self.aggr = aggr
        self.final_activation = nn.Hardsigmoid()

    def embed(self, x):
        # Mixture embedding
        # x_all = torch.Tensor()
        x_all = []
        for i in range(x.shape[-1]):
            mix = x[:, :, :, i]

            # Masking
            key_padding_mask = (mix == self.unk_token).all(dim=2)

            # Input projection
            if self.input_net is not None:
                mix = self.input_net(mix)

            emb_mix = self.mixture_net(mix, key_padding_mask)

            # if emb_mix.device != x_all.device:
            #     x_all = x_all.to(emb_mix.device)
            x_all.append(emb_mix)
        x_all = torch.stack(x_all, -1)      # [batch_size, embedding_size, num_mixtures]
        return x_all
    

    def forward(self, x):
        # Mixture embedding
        x_all = self.embed(x)

        if self.aggr == 'concat':
            x_all = torch.cat([x_all[:,:, i] for i in range(x_all.shape[-1])], dim=-1)
        elif self.aggr == 'sum':
            x_all = torch.sum(x_all, dim=-1)
        else:
            raise NotImplementedError

        pred = self.regressor(x_all)
        pred = self.final_activation(pred/2.)  # hard sigmoid from [-6,6]
        return pred