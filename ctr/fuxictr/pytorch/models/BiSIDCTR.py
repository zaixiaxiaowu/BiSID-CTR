import torch
from torch import nn

from ..layers import DNN_Layer
from ..layers.attention import TargetQueryAttention
from .base_model import BaseModel
from .model_utils import (
    SIDSequenceEncoder,
    categorical_fields,
    concat_fields,
    make_embedding_dict,
    sequence_mask,
    stack_fields,
)


class BiSIDCTR(BaseModel):
    """Bidirectional Semantic ID sequence CTR model."""

    def __init__(
        self,
        feature_map,
        model_id="BiSIDCTR",
        gpu=-1,
        task="binary_classification",
        learning_rate=1e-3,
        embedding_initializer="torch.nn.init.normal_(std=1e-4)",
        embedding_dim=32,
        attention_dim=128,
        hidden_units=(128, 128),
        hidden_activations="ReLU",
        num_heads=4,
        n_levels=3,
        max_seq_len=20,
        use_user_side=True,
        use_item_side=True,
        use_cross_attn=True,
        keep_pooled=True,
        use_sid=True,
        user_field="user_id",
        item_field="parent_asin",
        net_dropout=0,
        batch_norm=False,
        embedding_regularizer=None,
        net_regularizer=None,
        **kwargs,
    ):
        super(BiSIDCTR, self).__init__(
            feature_map,
            model_id=model_id,
            gpu=gpu,
            embedding_regularizer=embedding_regularizer,
            net_regularizer=net_regularizer,
            **kwargs,
        )
        self.use_user_side = use_user_side
        self.use_item_side = use_item_side
        self.use_cross_attn = use_cross_attn
        self.keep_pooled = keep_pooled
        self.use_sid = use_sid
        self.n_levels = n_levels
        self.user_field = user_field
        self.item_field = item_field
        self.item_hist_fields = (
            [f"item_hist_sid{i}" for i in range(1, n_levels + 1)] if use_sid else ["item_hist"]
        )
        self.user_hist_fields = (
            [f"user_hist_sid{i}" for i in range(1, n_levels + 1)] if use_sid else ["user_hist"]
        )
        self.target_item_sid_fields = [f"item_sid{i}" for i in range(1, n_levels + 1)]
        self.target_user_sid_fields = [f"user_sid{i}" for i in range(1, n_levels + 1)]
        seq_fields = set(self.item_hist_fields + self.user_hist_fields)
        self.cate_fields = [f for f in categorical_fields(feature_map) if f not in seq_fields]
        self.embedding_layer = make_embedding_dict(feature_map, embedding_dim)

        context_fields = [
            f for f in self.cate_fields
            if f not in {user_field, item_field} and not f.startswith("item_sid") and not f.startswith("user_sid")
        ]
        self.context_fields = context_fields
        sid_dim = embedding_dim * n_levels if use_sid else embedding_dim
        query_in = embedding_dim * (2 + len(context_fields)) + sid_dim * 2
        self.W_q = nn.Linear(query_in, attention_dim)
        self.item_seq_encoder = SIDSequenceEncoder(
            n_levels if use_sid else 1, embedding_dim, attention_dim, max_seq_len
        )
        self.user_seq_encoder = SIDSequenceEncoder(
            n_levels if use_sid else 1, embedding_dim, attention_dim, max_seq_len
        )
        self.user_side_attn = TargetQueryAttention(attention_dim, num_heads=num_heads)
        self.item_side_attn = TargetQueryAttention(attention_dim, num_heads=num_heads)
        self.cross_item_attn = TargetQueryAttention(attention_dim, num_heads=num_heads)
        self.cross_user_attn = TargetQueryAttention(attention_dim, num_heads=num_heads)
        self.fm_inner = None
        from ..layers.interaction import InnerProductLayer_v2
        self.fm_inner = InnerProductLayer_v2(len(self.cate_fields), output="sum")
        self.dnn_backbone = DNN_Layer(
            input_dim=embedding_dim * len(self.cate_fields),
            output_dim=None,
            hidden_units=list(hidden_units),
            hidden_activations=hidden_activations,
            final_activation=None,
            dropout_rates=net_dropout,
            batch_norm=batch_norm,
        )
        pred_in = hidden_units[-1] + attention_dim
        if use_user_side and keep_pooled:
            pred_in += attention_dim
        if use_item_side and keep_pooled:
            pred_in += attention_dim
        if use_user_side and use_item_side and use_cross_attn:
            pred_in += attention_dim * 2
        self.pred_mlp = DNN_Layer(
            input_dim=pred_in,
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

    def _sid_or_id_seq(self, emb_dict, fields):
        return [emb_dict[name] for name in fields]

    def _target_sid(self, emb_dict, fields, fallback_field):
        if self.use_sid:
            return concat_fields(emb_dict, fields)
        return emb_dict[fallback_field]

    def forward(self, inputs):
        X, y = self.inputs_to_device(inputs)
        emb_dict = self.embedding_layer(X)
        feature_emb = stack_fields(emb_dict, self.cate_fields)
        h_ctr = self.dnn_backbone(feature_emb.flatten(start_dim=1))
        fm_logit = self.fm_inner(feature_emb)

        e_u = emb_dict[self.user_field]
        e_i = emb_dict[self.item_field]
        e_c = concat_fields(emb_dict, self.context_fields) if self.context_fields else e_u.new_zeros(e_u.size(0), 0)
        phi_u = self._target_sid(emb_dict, self.target_user_sid_fields, self.user_field)
        phi_i = self._target_sid(emb_dict, self.target_item_sid_fields, self.item_field)
        query = self.W_q(torch.cat([e_u, e_i, e_c, phi_u, phi_i], dim=-1))

        pieces = [h_ctr, query]
        item_seq = user_seq = item_mask = user_mask = None
        h_pref = h_aud = None
        if self.use_user_side:
            item_seq = self.item_seq_encoder(self._sid_or_id_seq(emb_dict, self.item_hist_fields))
            item_mask = sequence_mask(X, self._feature_map, self.item_hist_fields[0])
            h_pref = self.user_side_attn(query, item_seq, item_mask)
            if self.keep_pooled:
                pieces.append(h_pref)
        if self.use_item_side:
            user_seq = self.user_seq_encoder(self._sid_or_id_seq(emb_dict, self.user_hist_fields))
            user_mask = sequence_mask(X, self._feature_map, self.user_hist_fields[0])
            h_aud = self.item_side_attn(query, user_seq, user_mask)
            if self.keep_pooled:
                pieces.append(h_aud)

        if self.use_user_side and self.use_item_side and self.use_cross_attn:
            pieces.append(self.cross_item_attn(h_pref, user_seq, user_mask))
            pieces.append(self.cross_user_attn(h_aud, item_seq, item_mask))

        y_pred = self.pred_mlp(torch.cat(pieces, dim=-1)) + fm_logit
        if self.final_activation is not None:
            y_pred = self.final_activation(y_pred)
        return {"loss": self.loss_with_reg(y_pred, y), "y_pred": y_pred}


class SIDCTR(BiSIDCTR):
    """User-side item SID sequence only."""

    def __init__(self, feature_map, model_id="SIDCTR", **kwargs):
        kwargs["use_user_side"] = True
        kwargs["use_item_side"] = False
        kwargs["use_cross_attn"] = False
        super(SIDCTR, self).__init__(feature_map, model_id=model_id, **kwargs)
