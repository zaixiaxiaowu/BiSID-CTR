"""LLM semantic profiling for items and users (BiSID-CTR Sec. 2.2)."""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict

from tqdm import tqdm

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from sid.prompts import (  # noqa: E402
    ITEM_PROFILE_PROMPT,
    USER_PROFILE_PROMPT,
    format_item_metadata,
    format_user_history,
)


def load_jsonl(path):
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def dump_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def raw_item_profile(item):
    return format_item_metadata(item)


def raw_user_profile(history_items, reviews):
    return format_user_history(history_items, reviews)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--skip_llm", action="store_true", help="Use raw text instead of LLM profiles")
    parser.add_argument("--max_user_items", type=int, default=20)
    parser.add_argument("--limit_items", type=int, default=None)
    parser.add_argument("--limit_users", type=int, default=None)
    args = parser.parse_args()

    items = {row["parent_asin"]: row for row in load_jsonl(os.path.join(args.data_dir, "items.jsonl"))}
    reviews = load_jsonl(os.path.join(args.data_dir, "reviews_text.jsonl"))
    reviews.sort(key=lambda x: int(x["timestamp"]))

    user_hist_items = defaultdict(list)
    user_hist_reviews = defaultdict(list)
    for review in reviews:
        asin = review["parent_asin"]
        user_hist_items[review["user_id"]].append(items.get(asin, {"title": asin}))
        user_hist_reviews[review["user_id"]].append(review)

    item_ids = list(items.keys())
    user_ids = list(user_hist_items.keys())
    if args.limit_items:
        item_ids = item_ids[: args.limit_items]
    if args.limit_users:
        user_ids = user_ids[: args.limit_users]

    chat_complete = None
    if not args.skip_llm:
        from sid.llm_client import chat_complete as _chat

        chat_complete = _chat

    item_profiles = []
    for asin in tqdm(item_ids, desc="item profiles"):
        raw = raw_item_profile(items[asin])
        if args.skip_llm:
            text = raw
        else:
            text = chat_complete(ITEM_PROFILE_PROMPT, raw)
        item_profiles.append({"parent_asin": asin, "profile": text, "raw": raw})
    dump_jsonl(item_profiles, os.path.join(args.data_dir, "item_profiles.jsonl"))

    user_profiles = []
    for user_id in tqdm(user_ids, desc="user profiles"):
        raw = raw_user_profile(
            user_hist_items[user_id][-args.max_user_items :],
            user_hist_reviews[user_id][-args.max_user_items :],
        )
        if args.skip_llm:
            text = raw
        else:
            text = chat_complete(USER_PROFILE_PROMPT, raw)
        user_profiles.append({"user_id": user_id, "profile": text, "raw": raw})
    dump_jsonl(user_profiles, os.path.join(args.data_dir, "user_profiles.jsonl"))


if __name__ == "__main__":
    main()
