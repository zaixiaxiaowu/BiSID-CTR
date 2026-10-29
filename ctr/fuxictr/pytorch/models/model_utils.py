import torch
from torch import nn

from ..layers.embedding import EmbeddingDictLayer
from ..layers.interaction import InnerProductLayer_v2


def categorical_fields(feature_map):
    return [name for name, spec in feature_map.feature_specs.items() if spec["type"] == "categorical"]


def sequence_mask(X, feature_map, field):
    spec = feature_map.feature_specs[field]
    padding_idx = spec.get("padding_idx", spec["vocab_size"] - 1)
    seq = X[:, spec["index"]].long()
    return seq != padding_idx


def stack_fields(emb_dict, fields):
    return torch.stack([emb_dict[name] for name in fields], dim=1)


def concat_fields(emb_dict, fields):
    return torch.cat([emb_dict[name] for name in fields], dim=-1)


class DeepFMBackbone(nn.Module):
    def __init__(self, num_fields, embedding_dim, hidden_units, hidden_activations="ReLU", net_dropout=0, batch_norm=False):
        super(DeepFMBackbone, self).__init__()
        from ..layers import DNN_Layer
        self.inner_product = InnerProductLayer_v2(num_fields, output="sum")
        self.dnn = DNN_Layer(
            input_dim=embedding_dim * num_fields,
            output_dim=None,
            hidden_units=hidden_units,
            hidden_activations=hidden_activations,
            final_activation=None,
            dropout_rates=net_dropout,
            batch_norm=batch_norm,
            use_bias=True,
        )
        self.hidden_dim = hidden_units[-1] if hidden_units else embedding_dim * num_fields

    def forward(self, feature_emb):
        fm_logit = self.inner_product(feature_emb)
        h_ctr = self.dnn(feature_emb.flatten(start_dim=1))
        return h_ctr, fm_logit


class SIDSequenceEncoder(nn.Module):
    def __init__(self, n_levels, in_dim, out_dim, max_len):
        super(SIDSequenceEncoder, self).__init__()
        self.n_levels = n_levels
        self.proj = nn.Linear(in_dim * n_levels, out_dim)
        self.pos_embedding = nn.Embedding(max_len, out_dim)

    def forward(self, level_embs):
        z = self.proj(torch.cat(level_embs, dim=-1))
        batch_size, seq_len, _ = z.size()
        positions = torch.arange(seq_len, device=z.device).unsqueeze(0).expand(batch_size, seq_len)
        return z + self.pos_embedding(positions)


def make_embedding_dict(feature_map, embedding_dim):
    return EmbeddingDictLayer(feature_map, embedding_dim)
