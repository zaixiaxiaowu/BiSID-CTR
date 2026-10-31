"""Instantiate BiSID-CTR on a tiny fake feature map."""
import os
import sys
from collections import OrderedDict

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "ctr"))

from fuxictr.features import FeatureMap  # noqa: E402
from fuxictr.pytorch.models import BiSIDCTR, DIN  # noqa: E402


def add_cat(fm, name, vocab, idx):
    fm.feature_specs[name] = {
        "source": "",
        "type": "categorical",
        "vocab_size": vocab,
        "index": idx,
        "min_categr_count": 1,
    }
    return idx + 1


def add_seq(fm, name, vocab, idx, max_len=4, share=None):
    fm.feature_specs[name] = {
        "source": "",
        "type": "sequence",
        "vocab_size": vocab,
        "index": list(range(idx, idx + max_len)),
        "max_len": max_len,
        "encoder": "none",
        "padding_idx": vocab - 1,
        "min_categr_count": 1,
    }
    if share:
        fm.feature_specs[name]["share_embedding"] = share
    return idx + max_len


def main():
    fm = FeatureMap("gift_sid")
    idx = 0
    cats = [
        "parent_asin",
        "user_id",
        "timestamp",
        "main_category",
        "title",
        "average_rating",
        "rating_number",
        "features",
        "description",
        "price",
        "store",
        "categories",
        "details",
        "item_sid1",
        "item_sid2",
        "item_sid3",
        "user_sid1",
        "user_sid2",
        "user_sid3",
    ]
    for name in cats:
        idx = add_cat(fm, name, 16, idx)
    idx = add_seq(fm, "item_hist", 16, idx, share="parent_asin")
    idx = add_seq(fm, "user_hist", 16, idx, share="user_id")
    for i in range(1, 4):
        idx = add_seq(fm, f"item_hist_sid{i}", 16, idx, share=f"item_sid{i}")
    for i in range(1, 4):
        idx = add_seq(fm, f"user_hist_sid{i}", 16, idx, share=f"user_sid{i}")
    fm.num_fields = len(fm.feature_specs)
    fm.feature_len = idx
    fm.num_features = 16 * fm.num_fields

    params = dict(
        optimizer="adam",
        loss="binary_crossentropy",
        model_root="./checkpoint_smoke",
        metrics=["logloss", "AUC"],
        verbose=0,
        gpu=-1,
        embedding_dim=8,
        attention_dim=16,
        hidden_units=[16],
        num_heads=2,
        n_levels=3,
        max_seq_len=4,
        learning_rate=1e-3,
    )
    model = BiSIDCTR(fm, **params)
    batch = 3
    X = torch.zeros(batch, fm.feature_len)
    y = torch.tensor([[1.0], [0.0], [1.0]])
    out = model.forward((X, y))
    assert out["y_pred"].shape == (batch, 1), out["y_pred"].shape
    out["loss"].backward()
    print("BiSIDCTR smoke ok", float(out["loss"]))

    replace = BiSIDCTR(fm, keep_pooled=False, **params)
    out = replace.forward((X, y))
    assert out["y_pred"].shape == (batch, 1), out["y_pred"].shape
    out["loss"].backward()
    print("BiSIDCTR -TA smoke ok", float(out["loss"]))

    din_fm = FeatureMap("gift_seq")
    idx = 0
    for name in cats[:13]:
        idx = add_cat(din_fm, name, 16, idx)
    idx = add_seq(din_fm, "item_hist", 16, idx, share="parent_asin")
    din_fm.num_fields = len(din_fm.feature_specs)
    din_fm.feature_len = idx
    din = DIN(din_fm, embedding_dim=8, hidden_units=[16], gpu=-1, optimizer="adam",
              loss="binary_crossentropy", model_root="./checkpoint_smoke", metrics=["logloss", "AUC"], verbose=0)
    X = torch.zeros(batch, din_fm.feature_len)
    out = din.forward((X, y))
    out["loss"].backward()
    print("DIN smoke ok", float(out["loss"]))


if __name__ == "__main__":
    main()
