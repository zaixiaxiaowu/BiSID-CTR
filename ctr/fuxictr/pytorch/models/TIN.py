import torch
from torch import nn

from ..layers import DNN_Layer
from ..layers.attention import DINAttention
from .base_model import BaseModel
from .model_utils import categorical_fields, make_embedding_dict, sequence_mask, stack_fields


class TIN(BaseModel):
    """Target-Interest Network style sequential CTR with semantic-temporal attention."""

    def __init__(
        self,
        feature_map,
        model_id="TIN",
        gpu=-1,
        task="binary_classification",
        learning_rate=1e-3,
        embedding_initializer="torch.nn.init.normal_(std=1e-4)",
        embedding_dim=32,
        hidden_units=(128, 128),
        hidden_activations="ReLU",
        attention_hidden_units=(64,),
        max_seq_len=20,
        target_field="parent_asin",
        sequence_field="item_hist",
        net_dropout=0,
        batch_norm=False,
        embedding_regularizer=None,
        net_regularizer=None,
        **kwargs,
    ):
        super(TIN, self).__init__(
            feature_map,
            model_id=model_id,
            gpu=gpu,
            embedding_regularizer=embedding_regularizer,
            net_regularizer=net_regularizer,
            **kwargs,
        )
        self.target_field = target_field
        self.sequence_field = sequence_field
        self.cate_fields = [f for f in categorical_fields(feature_map) if f != sequence_field]
        self.embedding_layer = make_embedding_dict(feature_map, embedding_dim)
        self.time_embedding = nn.Embedding(max_seq_len, embedding_dim)
        self.semantic_attn = DINAttention(
            embedding_dim,
            hidden_units=attention_hidden_units,
            hidden_activations="Dice",
            use_softmax=True,
        )
        self.temporal_gate = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.Sigmoid(),
        )
        self.dnn = DNN_Layer(
            input_dim=embedding_dim * (len(self.cate_fields) + 1),
            output_dim=1,
            hidden_units=list(hidden_units),
            hidden_activations=hidden_activations,
            final_activation=None,
            dropout_rates=net_dropout,
            batch_norm=batch_norm,
        )
        self.final_activation = self.get_final_activation(task)
        self.compile(kwargs["optimizer"], loss=kwargs["loss"], lr=learning_rate)
        self.init_weights(embedding_initializer=embedding_initializer)

    def forward(self, inputs):
        X, y = self.inputs_to_device(inputs)
        emb_dict = self.embedding_layer(X)
        hist = emb_dict[self.sequence_field]
        target = emb_dict[self.target_field]
        mask = sequence_mask(X, self._feature_map, self.sequence_field)
        seq_len = hist.size(1)
        recency = torch.arange(seq_len - 1, -1, -1, device=hist.device).clamp(max=self.time_embedding.num_embeddings - 1)
        time_emb = self.time_embedding(recency).unsqueeze(0).expand(hist.size(0), -1, -1)
        hist = hist + time_emb
        semantic = self.semantic_attn(target, hist, mask)
        gate = self.temporal_gate(torch.cat([semantic, target], dim=-1))
        pooling = gate * semantic + (1 - gate) * target
        feature_emb = torch.cat([stack_fields(emb_dict, self.cate_fields), pooling.unsqueeze(1)], dim=1)
        y_pred = self.dnn(feature_emb.flatten(start_dim=1))
        if self.final_activation is not None:
            y_pred = self.final_activation(y_pred)
        return {"loss": self.loss_with_reg(y_pred, y), "y_pred": y_pred}
