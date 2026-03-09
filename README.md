# CineMatch — Hybrid Movie Recommender System

[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://python.org)
[![Tests](https://img.shields.io/badge/tests-24%20passed-brightgreen.svg)](#testing)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A production-ready recommender system built from scratch in Python, implementing and comparing three recommendation strategies — Collaborative Filtering, Matrix Factorization, and Content-Based Filtering — fused into a weighted hybrid ensemble.

> **Highlights:** Implements 4 algorithms, 7 evaluation metrics, a full train/test pipeline, 24 unit tests, and a model explainability interface — all without black-box libraries like Surprise or LightFM.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Dataset](#dataset)
- [Models](#models)
- [Results](#results)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Evaluation Metrics](#evaluation-metrics)
- [Testing](#testing)
- [Explainability](#explainability)
- [Design Decisions](#design-decisions)

---

## Project Overview

CineMatch recommends movies to users based on their past rating history. It solves the core challenges of recommendation systems:

| Challenge | Approach |
|-----------|----------|
| Data sparsity (93.7% missing) | Truncated SVD learns dense latent representations |
| Cold-start items | Content-Based Filtering via genre TF-IDF |
| User rating bias | Mean-centering before similarity computation |
| Single-model weakness | Weighted ensemble blends 3 complementary signals |
| Evaluation beyond accuracy | Ranking metrics (NDCG, MAP, Precision/Recall @K) |

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     HybridRecommender                       │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │Collaborative │  │   Matrix     │  │  Content-Based   │  │
│  │  Filtering   │  │Factorization │  │    Filtering      │  │
│  │  (Item-KNN)  │  │    (SVD)     │  │  (TF-IDF genres) │  │
│  │   w = 0.30   │  │   w = 0.50   │  │    w = 0.20      │  │
│  └──────┬───────┘  └──────┬───────┘  └───────┬──────────┘  │
│         │                 │                   │             │
│         └─────────────────┴──────────┬────────┘             │
│                                      ▼                      │
│                          Min-Max Normalize Each             │
│                          Score → Weighted Sum               │
│                                      │                      │
│                                      ▼                      │
│                           Ranked Recommendation List        │
└─────────────────────────────────────────────────────────────┘
```

---

## Dataset

The dataset is a synthetic MovieLens-style corpus generated to match the statistical properties of the real MovieLens 100K benchmark:

| Property | Value |
|----------|-------|
| Users | 943 |
| Movies | 1,682 |
| Ratings | ~88,000 (after deduplication) |
| Rating scale | 1.0 – 5.0 |
| Mean rating | 3.54 |
| Matrix density | 6.33% |
| Genres | 18 categories |

**Why synthetic data?**  
The MovieLens dataset requires agreeing to distribution terms. This project generates equivalent data programmatically so the full pipeline is fully reproducible without any external downloads.

```bash
python generate_data.py          # regenerate data/
python generate_data.py --seed 7 # different random seed
```

---

## Models

### 1. Item-Based Collaborative Filtering (`src/collaborative_filter.py`)

Computes cosine similarity between item vectors in user-rating space. For each candidate item, finds the K most similar items the target user has rated and takes a weighted average.

**Key design choices:**
- Mean-centers user ratings before similarity computation (reduces high/low rater bias)
- Supports both user-based and item-based modes
- Handles unrated items gracefully via fallback to user mean

### 2. Matrix Factorization via Truncated SVD (`src/matrix_factorization.py`)

Decomposes the mean-centered user-item matrix into user and item latent factor matrices using `scipy.sparse.linalg.svds`. The dot product of user and item vectors gives the predicted rating.

```
R ≈ U · Σ · Vᵀ   →   ŷ(u, i) = uᵤ · vᵢ + μᵤ
```

- **50 latent factors** (configurable via `--n_factors`)
- Vectorized batch prediction with `np.einsum`
- Supports model persistence via `pickle`

### 3. Content-Based Filtering (`src/content_based.py`)

Builds a TF-IDF item profile matrix from pipe-separated genre tags. A user profile is constructed as the weighted average of profiles of items the user has rated (weights = rating − threshold). Cosine similarity ranks unseen items.

**Advantage:** Works for new items with no rating history (item cold-start).

### 4. Hybrid Ensemble (`src/hybrid_recommender.py`)

Each model's scores are independently min-max normalized to [0, 1], then combined via weighted sum:

```python
score = 0.30 * cf_score + 0.50 * mf_score + 0.20 * cb_score
```

The weights reflect MF's superior empirical performance while preserving the diversity and cold-start benefits of the other models.

---

## Results

Evaluated on a held-out 20% test split:

### Rating Prediction (MF)

| Metric | Score |
|--------|-------|
| RMSE | 1.17 |
| MAE | 0.97 |

### Ranking Quality (Hybrid, K=10)

| Metric | Score |
|--------|-------|
| Precision@10 | 0.029 |
| Recall@10 | 0.031 |
| NDCG@10 | 0.037 |
| MAP@10 | 0.014 |

*Note: Low absolute values are expected and normal for sparse implicit-preference evaluation on 1,500+ item catalogs — a random baseline would score near 0.003.*

### Visualizations

| EDA | Latent Space |
|-----|-------------|
| ![EDA](docs/eda_plots.png) | ![Latent](docs/latent_factors.png) |

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/YOUR_USERNAME/cinematch.git
cd cinematch
pip install -r requirements.txt
```

### 2. Generate data

```bash
python generate_data.py
```

### 3. Train all models

```bash
python train.py --model all --n_factors 50
```

### 4. Get recommendations for a specific user

```python
from src.data_loader import DataLoader
from src.hybrid_recommender import HybridRecommender
from scipy.sparse import csr_matrix

loader = DataLoader().load().preprocess()
uim = loader.build_user_item_matrix()

hybrid = HybridRecommender()
hybrid.fit(uim, loader.movies, loader.ratings,
           loader.user_id_map, loader.item_id_map)

recs = hybrid.recommend(user_id=42, n=10)
print(recs)
```

### 5. Run the notebook

```bash
jupyter notebook notebooks/01_eda_and_modeling.ipynb
```

---

## Project Structure

```
cinematch/
├── src/
│   ├── data_loader.py          # Data I/O, preprocessing, matrix construction
│   ├── collaborative_filter.py # User/Item KNN collaborative filtering
│   ├── matrix_factorization.py # Truncated SVD with latent factors
│   ├── content_based.py        # TF-IDF genre profiles + user taste vectors
│   ├── hybrid_recommender.py   # Weighted ensemble of all three models
│   └── evaluator.py            # RMSE, MAE, Precision/Recall/NDCG/MAP @K
├── tests/
│   └── test_recommender.py     # 24 unit + integration tests (pytest)
├── notebooks/
│   └── 01_eda_and_modeling.ipynb
├── docs/
│   ├── eda_plots.png
│   ├── matrix_heatmap.png
│   └── latent_factors.png
├── data/                       # Generated by generate_data.py (gitignored)
├── outputs/                    # Evaluation results, sample recs (gitignored)
├── generate_data.py            # Reproducible synthetic dataset generator
├── train.py                    # CLI training and evaluation pipeline
├── requirements.txt
├── setup.py
└── README.md
```

---

## Evaluation Metrics

| Metric | Type | What it measures |
|--------|------|-----------------|
| RMSE | Rating prediction | Penalizes large errors more than MAE |
| MAE | Rating prediction | Average absolute error in predicted rating |
| Precision@K | Ranking | Fraction of top-K recs that are relevant |
| Recall@K | Ranking | Fraction of all relevant items retrieved in top-K |
| F1@K | Ranking | Harmonic mean of Precision and Recall |
| NDCG@K | Ranking | Position-aware: rewards relevant items ranked higher |
| MAP@K | Ranking | Mean Average Precision across users |

---

## Testing

```bash
pytest tests/ -v
```

```
24 passed in 4.00s
```

Tests cover:
- Metric correctness (RMSE, MAE, NDCG edge cases)
- CF similarity matrix properties (diagonal = 1, shape, seen-item exclusion)
- MF output shapes, prediction range, score ordering
- Content-based profile construction and self-exclusion
- Hybrid fit → recommend → explain pipeline
- Evaluator coverage calculation

---

## Explainability

The hybrid model supports per-prediction score breakdown:

```python
explanation = hybrid.explain(user_id=42, movie_id=356)
# {
#   'userId': 42, 'movieId': 356,
#   'cf_score': 3.812,
#   'mf_score': 4.103,
#   'cb_score': 0.741,
#   'weights': {'cf': 0.3, 'mf': 0.5, 'cb': 0.2}
# }
```

---

## Design Decisions

**Why not use Surprise or LightFM?**  
Building from scratch demonstrates understanding of the underlying math rather than API usage. Every formula — cosine similarity, SVD decomposition, TF-IDF weighting — is explicitly implemented and testable.

**Why mean-center before similarity?**  
Users have different rating scales (some always rate 4–5, others 1–3). Subtracting each user's mean rating isolates preference signal from scale bias, leading to more meaningful similarity scores.

**Why weighted hybrid?**  
Each model has complementary failure modes: CF fails on cold-start items, MF struggles with very sparse users, CB ignores collaborative patterns. Combining them improves robustness without complex meta-learning.

**Why synthetic data?**  
Full reproducibility. Running `python generate_data.py` gives any reviewer an identical dataset in under 2 seconds with no authentication, downloads, or license agreements.

---

## License

MIT License — see [LICENSE](LICENSE) for details.
