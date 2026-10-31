"""Amazon Reviews 2023 CTR preprocessing, following LLaCTR (arxiv:2505.14057).

Expect the official 5-core jsonl dumps. Rating >= 4 is positive.
Interactions are sorted by timestamp and split 8:1:1.
This script also dumps item metadata and review text for offline SID construction.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from multiprocessing import Pool

from tqdm import tqdm

DOMAINS = {
    "gift": {
        "meta": "meta_Gift_Cards.jsonl",
        "review": "Gift_Cards.jsonl",
    },
    "game": {
        "meta": "meta_Video_Games.jsonl",
        "review": "Video_Games.jsonl",
    },
    "music": {
        "meta": "meta_Digital_Music.jsonl",
        "review": "Digital_Music.jsonl",
    },
}

FIELD_LIST = [
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
]

CSV_KEYS = ["rating"] + FIELD_LIST


def _stringify(value):
    if value is None:
        return "null"
    return str(value)


def load_meta(meta_path):
    meta_data_dict = {}
    with open(meta_path, "r", encoding="utf-8") as meta_file:
        for line in tqdm(meta_file, desc="meta"):
            meta_item = json.loads(line)
            meta_data_dict[meta_item["parent_asin"]] = meta_item
    print("meta data num:", len(meta_data_dict))
    return meta_data_dict


def iter_reviews(review_path, meta_data_dict):
    processed_list = []
    review_rows = []
    user_set = set()
    item_set = set()
    missing_meta = 0

    with open(review_path, "r", encoding="utf-8") as review_file:
        for line in tqdm(review_file, desc="reviews"):
            data = json.loads(line)
            parent_asin = data.get("parent_asin")
            meta_item = meta_data_dict.get(parent_asin)
            if meta_item is None:
                missing_meta += 1
                continue

            rating_modified = 1 if (data.get("rating") or 0) >= 4 else 0
            user_id = data.get("user_id")
            timestamp = data.get("timestamp")
            row = [
                rating_modified,
                parent_asin,
                user_id,
                timestamp,
                meta_item.get("main_category"),
                meta_item.get("title"),
                meta_item.get("average_rating"),
                meta_item.get("rating_number"),
                meta_item.get("features"),
                meta_item.get("description"),
                meta_item.get("price"),
                meta_item.get("store"),
                meta_item.get("categories"),
                meta_item.get("details"),
            ]
            processed_list.append(row)
            user_set.add(user_id)
            item_set.add(parent_asin)
            review_rows.append(
                {
                    "user_id": user_id,
                    "parent_asin": parent_asin,
                    "timestamp": timestamp,
                    "rating": data.get("rating"),
                    "title": data.get("title") or "",
                    "text": data.get("text") or "",
                }
            )

    print("data num:", len(processed_list))
    print("user num:", len(user_set))
    print("item num:", len(item_set))
    if user_set and item_set:
        print("sparsity:", len(processed_list) / (len(user_set) * len(item_set)))
    print("missing meta:", missing_meta)
    return processed_list, review_rows


def dump_items(meta_data_dict, used_items, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for asin in used_items:
            item = meta_data_dict[asin]
            record = {
                "parent_asin": asin,
                "main_category": item.get("main_category"),
                "title": item.get("title"),
                "average_rating": item.get("average_rating"),
                "rating_number": item.get("rating_number"),
                "features": item.get("features"),
                "description": item.get("description"),
                "price": item.get("price"),
                "store": item.get("store"),
                "categories": item.get("categories"),
                "details": item.get("details"),
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print("wrote", out_path)


def dump_jsonl(rows, out_path):
    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("wrote", out_path, "num:", len(rows))


def write_csv(filename, header, data):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)
    print(f"{os.path.basename(filename)} num:", len(data))


def run(data_dir, domain):
    spec = DOMAINS[domain]
    meta_path = os.path.join(data_dir, spec["meta"])
    review_path = os.path.join(data_dir, spec["review"])
    print(meta_path)
    print(review_path)

    meta_data_dict = load_meta(meta_path)
    processed_list, review_rows = iter_reviews(review_path, meta_data_dict)

    used_items = sorted({row[1] for row in processed_list})
    dump_items(meta_data_dict, used_items, os.path.join(data_dir, "items.jsonl"))
    dump_jsonl(review_rows, os.path.join(data_dir, "reviews_text.jsonl"))

    sorted_data = sorted(processed_list, key=lambda x: int(x[3]))
    total_samples = len(sorted_data)
    train_samples = int(0.8 * total_samples)
    valid_samples = int(0.1 * total_samples)
    train_data = sorted_data[:train_samples]
    valid_data = sorted_data[train_samples:train_samples + valid_samples]
    test_data = sorted_data[train_samples + valid_samples:]

    jobs = [
        (os.path.join(data_dir, "train.csv"), CSV_KEYS, train_data),
        (os.path.join(data_dir, "valid.csv"), CSV_KEYS, valid_data),
        (os.path.join(data_dir, "test.csv"), CSV_KEYS, test_data),
    ]
    with Pool(processes=min(8, os.cpu_count() or 1)) as pool:
        pool.starmap(write_csv, jobs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True, choices=sorted(DOMAINS))
    parser.add_argument("--data_dir", default=None, help="Directory that contains the Amazon jsonl files")
    args = parser.parse_args()
    data_dir = args.data_dir or os.getcwd()
    run(data_dir, args.domain)


if __name__ == "__main__":
    main()
