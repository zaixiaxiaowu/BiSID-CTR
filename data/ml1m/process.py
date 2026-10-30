"""MovieLens-1M CTR preprocessing, following LLaCTR (arxiv:2505.14057).

Ratings > 3 are positive, ratings < 3 are negative, and rating == 3 is dropped.
Interactions are sorted by timestamp and split 8:1:1.
"""
from __future__ import annotations

import csv
import json
import os
import re

GENDER_MAP = {"F": "female", "M": "male"}
AGE_MAP = {
    "1": "Under 18",
    "18": "18-24",
    "25": "25-34",
    "35": "35-44",
    "45": "45-49",
    "50": "50-55",
    "56": "56+",
}
OCCUPATION_MAP = {
    "0": "other or not specified",
    "1": "academic/educator",
    "2": "artist",
    "3": "clerical/admin",
    "4": "college/grad student",
    "5": "customer service",
    "6": "doctor/health care",
    "7": "executive/managerial",
    "8": "farmer",
    "9": "homemaker",
    "10": "K-12 student",
    "11": "lawyer",
    "12": "programmer",
    "13": "retired",
    "14": "sales/marketing",
    "15": "scientist",
    "16": "self-employed",
    "17": "technician/engineer",
    "18": "tradesman/craftsman",
    "19": "unemployed",
    "20": "writer",
}
CSV_FIELDS = [
    "UserID",
    "Gender",
    "Age",
    "Occupation",
    "MovieID",
    "Title",
    "Genres",
    "Rating",
    "Timestamp",
]


def _find_ml1m_dir(data_dir):
    candidates = [
        os.path.join(data_dir, "ml-1m"),
        data_dir,
    ]
    for path in candidates:
        if os.path.isfile(os.path.join(path, "ratings.dat")):
            return path
    raise FileNotFoundError(
        "Put MovieLens-1M files in data/ml1m/ml-1m/ (ratings.dat, users.dat, movies.dat). "
        "Download: https://grouplens.org/datasets/movielens/1m/"
    )


def load_users(path):
    user_info = {}
    with open(path, "r", encoding="latin-1") as infile:
        for line in infile:
            user_id, gender, age, occupation, _ = re.split(r"::", line.strip())
            user_info[user_id] = {
                "Gender": GENDER_MAP.get(gender, gender),
                "Age": AGE_MAP.get(age, age),
                "Occupation": OCCUPATION_MAP.get(occupation, occupation),
            }
    print("user num (raw):", len(user_info))
    return user_info


def load_movies(path):
    movie_info = {}
    with open(path, "r", encoding="latin-1") as infile:
        for line in infile:
            match = re.match(r"(\d+)::(.*)::(.*)", line.strip())
            if not match:
                continue
            movie_id, title, genres = match.groups()
            movie_info[movie_id] = {"Title": title, "Genres": genres}
    print("movie num (raw):", len(movie_info))
    return movie_info


def write_csv(rows, filename):
    with open(filename, "w", newline="", encoding="utf-8") as outfile:
        writer = csv.DictWriter(outfile, fieldnames=CSV_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(os.path.basename(filename), "num:", len(rows))


def dump_jsonl(rows, path):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print("wrote", path, "num:", len(rows))


def run(data_dir):
    src = _find_ml1m_dir(data_dir)
    user_info = load_users(os.path.join(src, "users.dat"))
    movie_info = load_movies(os.path.join(src, "movies.dat"))

    data = []
    review_rows = []
    used_movies = set()
    with open(os.path.join(src, "ratings.dat"), "r", encoding="latin-1") as infile:
        for line in infile:
            user_id, movie_id, rating, timestamp = line.strip().split("::")
            raw_rating = int(rating)
            if raw_rating == 3:
                continue
            label = 1 if raw_rating > 3 else 0
            user = user_info.get(user_id, {})
            movie = movie_info.get(movie_id, {})
            data.append(
                {
                    "UserID": user_id,
                    "Gender": user.get("Gender", ""),
                    "Age": user.get("Age", ""),
                    "Occupation": user.get("Occupation", ""),
                    "MovieID": movie_id,
                    "Title": movie.get("Title", ""),
                    "Genres": movie.get("Genres", ""),
                    "Rating": label,
                    "Timestamp": timestamp,
                }
            )
            used_movies.add(movie_id)
            review_rows.append(
                {
                    "user_id": user_id,
                    "parent_asin": movie_id,
                    "timestamp": timestamp,
                    "rating": raw_rating,
                    "title": movie.get("Title", ""),
                    "text": "",
                }
            )

    items = []
    for movie_id in sorted(used_movies, key=lambda x: int(x)):
        movie = movie_info.get(movie_id, {})
        items.append(
            {
                "parent_asin": movie_id,
                "title": movie.get("Title", ""),
                "genres": movie.get("Genres", ""),
                "main_category": "Movies",
                "categories": movie.get("Genres", ""),
            }
        )
    dump_jsonl(items, os.path.join(data_dir, "items.jsonl"))
    dump_jsonl(review_rows, os.path.join(data_dir, "reviews_text.jsonl"))

    data = sorted(data, key=lambda x: int(x["Timestamp"]))
    total_samples = len(data)
    train_samples = int(0.8 * total_samples)
    valid_samples = int(0.1 * total_samples)
    write_csv(data[:train_samples], os.path.join(data_dir, "train.csv"))
    write_csv(data[train_samples:train_samples + valid_samples], os.path.join(data_dir, "valid.csv"))
    write_csv(data[train_samples + valid_samples:], os.path.join(data_dir, "test.csv"))

    users = {row["UserID"] for row in data}
    movies = {row["MovieID"] for row in data}
    print("all_data_num=", total_samples)
    print("user_num=", len(users))
    print("item_num=", len(movies))
    print("sparsity=", total_samples / (len(users) * len(movies)))


if __name__ == "__main__":
    run(os.path.dirname(os.path.abspath(__file__)))
