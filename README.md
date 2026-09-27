# DSN Mart Sales Prediction — DSN AI Bootcamp Qualification Hackathon 2026

This repo contains my full solution to the **DSN Mart Sales Prediction Hackathon**, the qualifying competition for the 2026 DSN AI Bootcamp. The task: predict `total_sales` for a given product at a given store, using only what's known about the product and the outlet it's sold in.

Leaderboard rank feeds directly into Bootcamp selection, so this notebook isn't just about chasing a score — it's built to actually show the reasoning behind every decision: what I found in the data, what I tried that didn't work, and why I ended up with the model I did.

---

## The problem

DSN Mart runs everything from small corner shops to flagship hypermarkets across major cities, state capitals, and smaller towns in Nigeria. They want to understand what actually drives product-level sales across these different formats and locations, so they can plan stock, pricing, and store investment more intelligently.

Given historical product-store sales records, the job is to build a model that predicts `total_sales` for unseen product-store combinations. It's a straight regression problem, scored on **RMSE** (lower is better) — no partial credit for being close on some rows and wildly off on others, since RMSE punishes big misses harder than small ones.

## Dataset

| Column | Description |
|---|---|
| `id` | Unique row identifier |
| `product_code` | Unique code for the product |
| `product_weight_kg` | Weight of the product in kg (some missing) |
| `fat_content` | "Low Fat" or "Regular" |
| `shelf_visibility` | Proportion of total display area allocated to this product |
| `product_category` | Product's category (capitalization is inconsistent — real-world data, not a bug) |
| `product_price` | Listed price |
| `store_code` | Unique code for the store |
| `store_age_years` | How long the store has been operating |
| `store_size` | Small / Medium / Large (some missing) |
| `store_location_tier` | Tier_1 (major urban centers), Tier_2 (state capitals), Tier_3 (smaller towns) |
| `store_format` | Corner Shop, Standard Supermarket, Superstore, or Flagship Hypermarket |
| `total_sales` | **Target.** Train only. |

`train.csv` has the real `total_sales` values. `test.csv` is the same structure with `total_sales` stripped out — that's what gets predicted and submitted.

---

## How I actually approached this

### 1. Understanding the data before touching a model

Before writing a single line of modeling code, I spent time just interrogating the dataset — checking what's missing, what's inconsistent, and what actually correlates with sales. A few things stood out:

- **`product_price` is doing almost all the heavy lifting.** Correlation with `total_sales` sits around 0.57 — nothing else comes close on its own. `product_weight_kg` and `store_age_years` are both essentially noise (correlations near 0.02–0.04).
- **`fat_content` had a labeling quirk** — non-food categories (like household or health-and-hygiene items) were still tagged "Low Fat" or "Regular," which obviously doesn't mean anything for those products.
- **Three stores had no recoverable `store_size` at all** — every single row for those stores was missing. That had to be handled deliberately rather than just dropped.
- **`store_location_tier` and `store_format` showed a real, non-random ranking** in average sales, which shaped how I treated those columns.
- **`shelf_visibility` had a batch of exact-zero values**, which don't make physical sense (a product being sold somewhere can't have literally zero display space) — treated as missing rather than real measurements.
- **`store_code` turned out to be more replaceable than expected.** With only 10 stores in the whole dataset, `store_format`, `store_location_tier`, `store_size`, and store-level aggregates I engineered (below) already capture most of what the raw store identity was contributing. I tested dropping `store_code` outright — CV barely moved, and it improved the leaderboard score twice in a row on a 50/50 public/private split, so it's out of the final feature set.

### 2. Cleaning and preprocessing

- `store_size`: imputed using the most common size *for that specific store* (group-mode fill), with an explicit `"unknown"` fallback for the 3 stores that had nothing to recover.
- `product_weight_kg`: imputed using the average weight *for that specific product* across the stores it appears in, falling back to the global mean only when a product had no recoverable weight at all.
- `shelf_visibility`: zeros converted to missing, then filled using product-level average, falling back to category-level average, then global mean as a last resort.
- `product_category`, `store_format`, `store_location_tier`, `fat_content`: lowercased and stripped to fix inconsistent capitalization.
- All cleaning and feature engineering is done on train and test **combined** (never separately), so derived features mean the same thing on both sides.

### 3. Feature engineering — what actually helped, and what didn't

Each product in this dataset shows up in roughly 4–5 different stores on average, which opens the door to features that summarize a product *across* stores, not just within one row. The features currently in the pipeline:

- **`product_num_stores`** — how many distinct stores carry a given product.
- **`product_avg_price`** — a product's average price across every store that sells it.
- **`visibility_ratio_to_cat`** — a product's shelf visibility relative to its category's average.
- **`store_format_tier`** — a combined store-format + location-tier feature.
- **`price_band`**, **`price_discount_ratio`**, **`price_discount_diff`**, **`price_to_cat_mean`**, **`price_per_weight`** — price relative to a product's own average, and relative to its category.
- **`store_item_count`**, **`store_cat_share`**, **`store_price_level`** — store-level aggregates that substitute for the raw store identity once `store_code` was dropped.
- **`visibility_ratio_to_product`** — visibility relative to a product's own average visibility.
- Frequency encodings for `product_code`, `product_category`, `store_code`, `store_format_tier`.

Not everything I tried made the cut. The first four features above were tested individually against a clean CatBoost baseline before being kept. The rest were added as a batch and judged on the combined result rather than one at a time — a few ideas from that batch (a plain shelf-visibility ratio, a store-format × category interaction, weight bins) moved CV by less than a point, inside the noise of the folds, and were dropped.

**Out-of-fold target encoding** is applied on top of this for `product_category` and `store_format_tier` (smoothed toward the global mean, built only from each fold's training rows) — used for the XGBoost and LightGBM models specifically, since they don't have CatBoost's native categorical handling.

### 4. Model selection and stacking

I compared model families rather than committing to one upfront, and eventually moved from a single CatBoost model to a small stack:

| Model | Approach | Local CV RMSE |
|---|---|---|
| XGBoost | Native `category` dtype + target encoding | ~1077 |
| CatBoost | Native categorical handling | **~1074** |
| LightGBM | Native categorical handling + target encoding | ~1083 |
| Sales-per-price (volume decomposition) | LightGBM predicting `total_sales / product_price`, scaled back by price | ~1076–1083 |
| **Stacked (Ridge, non-negative weights)** | Combines all four via out-of-fold predictions | **~1073.7** |

**CatBoost remains the strongest single model**, and the stack — a Ridge meta-model with weights forced non-negative, trained on each base model's out-of-fold predictions — gives a small, genuine improvement over CatBoost alone. The Ridge weights consistently favor CatBoost and XGBoost, with LightGBM often getting close to zero weight.

I keep two final submission paths from this notebook rather than one:
- `submission_cat.csv` — CatBoost alone, trained on all training rows.
- `submission_stack.csv` — the 4-model stack.

Both are selected as final submissions on Kaggle, so the platform's private leaderboard (also a 50% split) decides between them rather than me guessing upfront.

### 5. Tuning — and catching my own overfitting

I tuned manually first, then ran a search on top of it. Most reasonable configurations landed in a tight cluster around RMSE 1071–1080, regardless of fairly different parameter combinations — a signal that the feature set has a real ceiling, not that I hadn't searched hard enough.

At one point, pushing tree depth and iteration count up sharply produced a very strong local CV score. On the leaderboard, that same model scored noticeably worse than a shallower, more conservative version. That gap is a textbook overfitting signature: extra capacity was fitting patterns specific to my training folds that didn't generalize. I reverted to the shallower configuration below, which has shown consistent agreement between local CV and the leaderboard.

**Final CatBoost configuration:**
```python
CatBoostRegressor(
    iterations=500,
    learning_rate=0.03,
    depth=3,
    l2_leaf_reg=3,
    min_data_in_leaf=10,
    bagging_temperature=1,
    boosting_type='Ordered',
    cat_features=categorical_columns,
    loss_function='RMSE',
    random_state=42
)
```

No model in this pipeline uses early stopping on the validation fold — every model trains for a fixed number of iterations, so the reported cross-validation score isn't inflated by a model peeking at the exact rows it's being scored on.

### 6. What didn't work

- **A plain average of CatBoost and XGBoost** scored worse on the leaderboard than CatBoost alone — a simple average pulls a strong model toward a meaningfully weaker one instead of improving on it. This is what pushed me toward proper stacking (a learned, non-negative-weighted blend) instead of a naive average.
- Several batch-added features, tested in isolation later, showed no CV benefit on their own — kept anyway as part of the full combined set, since the full combination outperforms the sum of its individually-tested parts.
- Removing `store_code` was expected (by my own early testing) to hurt — it instead helped, twice, on a 50/50 public/private leaderboard split, which is large enough that this is very unlikely to be luck.

### 7. Validation strategy throughout

Every feature, every parameter choice, every "this didn't help" claim was checked against **10-fold cross-validation**, not a single train/test split and not gut feeling. That's also what caught the deep-tree overfitting case before it became a final submission by accident.

---

## Results

- **Local CV RMSE:** ~1073–1074 (CatBoost alone), ~1073.7 (stack, honestly cross-validated)
- **Leaderboard RMSE:** ~1067.5 (stack), ~1069.6 (CatBoost alone), both on a 50% public split
- Leaderboard rank improved from ~40th to ~17th over the course of this process.
- Two submissions are locked in as Kaggle finals (`submission_stack.csv` and `submission_cat.csv`), so the private leaderboard — not a guess — decides which one counts.

## Repo structure
