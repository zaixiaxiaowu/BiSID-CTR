"""Attach bidirectional histories and Semantic IDs to LLaCTR CTR splits.

For each target (u, i, t), this writes:
- item_hist: user-side item ID sequence with timestamp < t
- user_hist: item-side user ID sequence with timestamp < t
- SID features and SID sequences at each RQ level
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict

import pandas as pd
from tqdm import tqdm

SPLITS = ("train", "valid", "test")
SID_LEVELS = (1, 2, 3)


def load_sids(path):
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("sids", payload)


def join_seq(values, max_len):
    values = [str(v) for v in values if v is not None and str(v) != ""]
    if max_len > 0:
        values = values[-max_len:]
    return "^".join(values)


def sid_tuple(mapping, key, n_levels):
    codes = mapping.get(key)
    if not codes:
        return ["0"] * n_levels
    codes = [str(c) for c in codes[:n_levels]]
    while len(codes) < n_levels:
        codes.append("0")
    return codes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--max_len", type=int, default=50)
    parser.add_argument("--n_levels", type=int, default=3)
    args = parser.parse_args()

    frames = []
    for split in SPLITS:
        path = os.path.join(args.data_dir, f"{split}.csv")
        df = pd.read_csv(path, dtype=str)
        df["split"] = split
        frames.append(df)
    data = pd.concat(frames, ignore_index=True)
    data["timestamp"] = data["timestamp"].astype("int64")
    data = data.sort_values("timestamp", kind="mergesort").reset_index(drop=True)

    item_sids = load_sids(os.path.join(args.data_dir, "item_sids.json"))
    user_sids = load_sids(os.path.join(args.data_dir, "user_sids.json"))
    n_levels = args.n_levels

    user_item_hist = defaultdict(list)
    item_user_hist = defaultdict(list)
    user_item_sid_hist = {lvl: defaultdict(list) for lvl in range(n_levels)}
    item_user_sid_hist = {lvl: defaultdict(list) for lvl in range(n_levels)}

    records = {split: [] for split in SPLITS}
    for row in tqdm(data.itertuples(index=False), total=len(data), desc="histories"):
        user_id = row.user_id
        item_id = row.parent_asin
        item_codes = sid_tuple(item_sids, item_id, n_levels)
        user_codes = sid_tuple(user_sids, user_id, n_levels)

        out = row._asdict()
        out["item_hist"] = join_seq(user_item_hist[user_id], args.max_len)
        out["user_hist"] = join_seq(item_user_hist[item_id], args.max_len)
        for lvl in range(n_levels):
            out[f"item_sid{lvl + 1}"] = item_codes[lvl]
            out[f"user_sid{lvl + 1}"] = user_codes[lvl]
            out[f"item_hist_sid{lvl + 1}"] = join_seq(user_item_sid_hist[lvl][user_id], args.max_len)
            out[f"user_hist_sid{lvl + 1}"] = join_seq(item_user_sid_hist[lvl][item_id], args.max_len)

        records[out["split"]].append(out)

        user_item_hist[user_id].append(item_id)
        item_user_hist[item_id].append(user_id)
        for lvl in range(n_levels):
            user_item_sid_hist[lvl][user_id].append(item_codes[lvl])
            item_user_sid_hist[lvl][item_id].append(user_codes[lvl])

    drop_cols = ["split"]
    for split, rows in records.items():
        df = pd.DataFrame(rows).drop(columns=drop_cols)
        out_path = os.path.join(args.data_dir, f"{split}_sid.csv")
        df.to_csv(out_path, index=False)
        print(f"{os.path.basename(out_path)} num:", len(df))


if __name__ == "__main__":
    main()
