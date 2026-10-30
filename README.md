# BiSID-CTR

Bidirectional Semantic ID Sequences for Click-Through Rate Prediction.

This repository starts from the open-source [LLaCTR](https://github.com/istarryn/LLaCTR) pipeline (Amazon Reviews 2023 preprocessing + FuxiCTR training) and adds the BiSID-CTR model: LLM profiling, RQ-KMeans Semantic IDs, bidirectional SID sequences, and target-aware cross-attention.

数据集与 LLaCTR / arXiv:2505.14057 一致，共四个：Amazon Reviews 2023 的 Gift Cards、Video Games、Digital Music（评分 ≥ 4 为正样本），以及 MovieLens-1M（丢掉 rating == 3，>3 为正、<3 为负）。均按时间戳 8:1:1 划分。

## Method

1. **SID construction (offline).** Qwen3-Max writes item/user profiles; Qwen3-Embedding encodes them; RQ-KMeans (L=3, 256 centroids) produces SIDs.
2. **Bidirectional sequences.** For a target pair `(u, i)` at time `t`, the user-side item SID sequence is preference evidence and the item-side user SID sequence is audience evidence. Both are truncated to timestamps `< t`.
3. **Target-aware fusion.** The target pair is the query that aggregates both sequences; the two views then interact by cross-attention and are fed into a DeepFM backbone.

Baselines implemented on the same split: DeepFM, DIN, BST, TIN, SID-Feat, SID-CTR.

## Setup

```bash
conda create -n bisid python=3.10
conda activate bisid
pip install -r requirements.txt
```

Download Amazon Reviews 2023 jsonl files from [amazon-reviews-2023](https://amazon-reviews-2023.github.io/) into the domain folders:

```text
data/gift/meta_Gift_Cards.jsonl
data/gift/Gift_Cards.jsonl
data/game/meta_Video_Games.jsonl
data/game/Video_Games.jsonl
data/music/meta_Digital_Music.jsonl
data/music/Digital_Music.jsonl
```

Download [MovieLens-1M](https://grouplens.org/datasets/movielens/1m/) and unzip so that these files exist:

```text
data/ml1m/ml-1m/ratings.dat
data/ml1m/ml-1m/users.dat
data/ml1m/ml-1m/movies.dat
```

## Data pipeline

The first step is LLaCTR's processor (Windows-safe rewrite of `process.py`). Later steps add BiSID features.

```powershell
# Gift Cards example
cd data/gift
python process.py

cd ../..
# LLM profiles (needs LLM_API_KEY or DASHSCOPE_API_KEY)
python data/build_profiles.py --data_dir data/gift
python data/build_sids.py --data_dir data/gift
python data/prepare_bisid.py --data_dir data/gift
```

Without an LLM API you can still run the pipeline (this matches the paper's `-LLM` ablation, plus hashed embeddings for a dry run):

```powershell
python data/build_profiles.py --data_dir data/gift --skip_llm
python data/build_sids.py --data_dir data/gift --hash_embed
python data/prepare_bisid.py --data_dir data/gift
```

Or use `scripts/prepare_domain.ps1 -Domain gift -SkipLlm -HashEmbed`.

MovieLens-1M uses `UserID` / `MovieID` / `Timestamp` instead of Amazon's `user_id` / `parent_asin` / `timestamp`:

```powershell
cd data/ml1m
python process.py

cd ../..
python data/build_profiles.py --data_dir data/ml1m
python data/build_sids.py --data_dir data/ml1m
python data/prepare_bisid.py --data_dir data/ml1m --user_col UserID --item_col MovieID --time_col Timestamp
```

Or `scripts/prepare_domain.ps1 -Domain ml1m`.

Environment variables for the official setting:

| Variable | Default | Meaning |
| --- | --- | --- |
| `LLM_API_KEY` / `DASHSCOPE_API_KEY` | — | API key |
| `LLM_BASE_URL` | DashScope compatible mode | OpenAI-compatible endpoint |
| `LLM_MODEL` | `qwen3-max` | Profiling model (Qwen3-Max) |
| `EMBED_MODEL` | `text-embedding-v4` | Embedding model (Qwen3-Embedding) |

## Training

From `ctr/`:

```powershell
python run_expid.py --config ./config --expid DeepFM_gift --gpu 0
python run_expid.py --config ./config --expid DIN_gift --gpu 0
python run_expid.py --config ./config --expid BST_gift --gpu 0
python run_expid.py --config ./config --expid TIN_gift --gpu 0
python run_expid.py --config ./config --expid SIDFeat_gift --gpu 0
python run_expid.py --config ./config --expid SIDCTR_gift --gpu 0
python run_expid.py --config ./config --expid BiSIDCTR_gift --gpu 0
```

Replace `gift` with `game`, `music`, or `ml1m`. Paper defaults: batch size 1024, lr 0.001, hidden size 128, 4 attention heads, sequence length 50, SID length 3, patience 10, Adam (dense) + FTRL (sparse).

Ablations on Video Games:

```powershell
python run_expid.py --config ./config --expid BiSIDCTR_game_woUS --gpu 0
python run_expid.py --config ./config --expid BiSIDCTR_game_woIS --gpu 0
python run_expid.py --config ./config --expid BiSIDCTR_game_woXA --gpu 0
python run_expid.py --config ./config --expid BiSIDCTR_game_AID --gpu 0
```

## Layout

```text
data/           LLaCTR preprocess (Amazon + MovieLens-1M) + SID/sequence construction
sid/            LLM client, prompts, RQ-KMeans
ctr/fuxictr/    FuxiCTR v1 from LLaCTR, plus DIN/BST/TIN/SID-Feat/SID-CTR/BiSID-CTR
ctr/config/     dataset and model configs
scripts/        PowerShell helpers
```

## Acknowledgements

CTR training and Amazon preprocessing are adapted from [LLaCTR](https://github.com/istarryn/LLaCTR) and [FuxiCTR](https://github.com/reczoo/FuxiCTR).
