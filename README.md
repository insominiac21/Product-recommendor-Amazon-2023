# 🛒 Amazon Hybrid Recommender — Notebook & Pipeline

> **Project:** Hybrid retrieval + rerank recommender built for Amazon-style product review data  
> **Notebook:** `amazon-hybrid-recommender.ipynb`  
> **Output directory:** `/kaggle/working` (by default assigned to `OUT_DIR`)  
> **Dataset:** [McAuley-Lab / Amazon-Reviews-2023 — Electronics config](https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023)

---

## 🚀 Project overview (elevator pitch)

This repository implements a **two-stage hybrid recommender**:

1. **Retrieval / Candidate generation**
   - Semantic retrieval (SBERT embeddings + FAISS ANN or TF-IDF+SVD fallback)
   - Collaborative retrieval (ALS or SVD on the item-user matrix)
   - Popularity-based retrieval
   - Union & deduplication into a candidate pool per user

2. **Reranking (Learning-to-Rank)**
   - Feature engineering for each (user, candidate) pair: semantic score, MF score, popularity, recency, text length, user statistics, rank features, etc.
   - Hard-negative mining (hard + random negatives)
   - Train a LightGBM reranker (`lambdarank`) on the meta dataset to optimize ranking (NDCG@K)

This architecture balances **high recall** at retrieval and **precise ranking** by the reranker, and is designed to scale from 10k to 500k rows with CPU/GPU-friendly fallbacks.

---

## 📁 Output files & folders (what the notebook writes)

By default all outputs are saved to `OUT_DIR` (`/kaggle/working` on Kaggle). Typical artifact names:

| Artifact | Description |
|---|---|
| `sampled_reviews_{SCALE_N}.parquet` | Cached sampled input rows used for the run (SCALE_N = 100000, 200000, 500000, etc.) |
| `item_emb_{SCALE_N}.joblib` | Item embeddings (SBERT or TF-IDF+SVD fallback) |
| `item_emb_norm_{SCALE_N}.npy` | Normalized embeddings used for ANN |
| `faiss_hnsw_{SCALE_N}.index` | FAISS HNSW index (if FAISS available) |
| `als_item_factors_{SCALE_N}.joblib` | Item latent factors from MF (ALS or SVD fallback) |
| `als_user_factors_{SCALE_N}.joblib` | User latent factors from MF |
| `meta_enhanced_scale_{SCALE_N}.parquet` | Hard-negative LTR training meta dataset |
| `lgbm_reranker_final_{SCALE_N}.joblib` | Trained LightGBM LTR model |
| `final_scale_summary_metrics_{SCALE_N}.csv` | Final evaluation metrics (P@10, R@10, NDCG@10, MAP@10, MPR@10, EvalUsers) |

---

## 🧭 Dataset link & notes

- Dataset: **Amazon Reviews 2023** (McAuley-Lab)  
  HF dataset link:  
  `https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023`  
  The notebook uses the `raw_review_Electronics` configuration but is written to generalize to other categories.

**Important:** streaming downloads may fail under network restrictions — the notebook caches sampled rows (`sampled_reviews_{SCALE_N}.parquet`) so you can resume work without re-downloading.

---

## 🧰 Requirements

Minimum recommended packages (use your environment manager / `pip`):

```bash
pip install datasets sentence-transformers scikit-learn lightgbm optuna joblib tqdm pandas scipy faiss-cpu implicit
