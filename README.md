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

- **`product_price` is doing almost all the heavy lifting.** Correlation with `total_sales` sits around 0.57 — nothing else comes close. `product_weight_kg` and `store_age_years` are both essentially noise (correlations near 0.02–0.04).
- **`fat_content` had a labeling quirk** — non-food categories (like household or electronics-adjacent items) were still tagged "Low Fat" or "Regular," which obviously doesn't mean anything for those products. Worth knowing, even if it didn't end up being a feature I leaned on heavily.
- **Three stores had no recoverable `store_size` at all** — every single row for those stores was missing. That had to be handled deliberately rather than just dropped.
- **`store_location_tier` and `store_format` showed a real, non-random ranking** in average sales (Tier_2 > Tier_3 > Tier_1; Flagship Hypermarket > Superstore > Standard Supermarket > Corner Shop), which shaped how I treated those columns.
- **`store_size` surprised me.** I initially assumed it was basically redundant with `store_code` (since each store has one fixed size), and even considered dropping it. Testing that directly — removing it and rerunning cross-validation — proved the opposite: performance dropped noticeably without it. My best guess is that it's doing real work distinguishing those 3 stores with unknown/missing size, information `store_code` alone doesn't hand the model as directly.
- **`shelf_visibility` had a batch of exact-zero values**, which don't make physical sense (a product being sold somewhere can't have literally zero display space) — treated as missing rather than real measurements.

### 2. Cleaning and preprocessing

- `store_size`: imputed using the most common size *for that specific store* (group-mode fill), with an explicit `"unknown"` fallback for the 3 stores that had nothing to recover.
- `product_weight_kg`: imputed using the average weight *for that specific product* across the stores it appears in, falling back to the global mean only when a product had no recoverable weight at all.
- `shelf_visibility`: zeros converted to missing, then filled using product-level average, falling back to category-level average, then global mean as a last resort.
- `product_category`: lowercased and stripped to fix inconsistent capitalization.

### 3. Feature engineering — what actually helped, and what didn't

Each product in this dataset shows up in roughly 4–5 different stores on average, which opens the door to features that summarize a product *across* stores, not just within one row:

- **`product_avg_price`** — a product's average price across every store that sells it.
- **`product_num_stores`** — how many distinct stores carry a given product (a rough proxy for how "mainstream" vs. niche it is).
- **`price_deviation`** — the difference between this row's specific price and that product's own average price, so the model doesn't have to rediscover that comparison on its own.

These three were tested individually against a clean baseline using 5-fold cross-validation, not just added on faith. Several other ideas I tried — a shelf-visibility ratio relative to a product's own average visibility, price relative to category average, a store-format × category interaction, weight bins — moved the RMSE by less than a point, well within the noise of the cross-validation folds (fold standard deviation was consistently around 20). I didn't keep those; adding features that don't demonstrably help just adds fragility for no real gain.

### 4. Model selection

I compared three model families rather than committing to one upfront:

| Model | Approach | Local CV RMSE |
|---|---|---|
| XGBoost | One-hot encoded categoricals | ~1081–1085 |
| XGBoost | Native `category` dtype | ~1081 |
| CatBoost | Native categorical handling | **~1073** |
| CatBoost | One-hot encoded (for comparison) | ~1075 |
| LightGBM | Native categorical handling | ~1076 |

**CatBoost came out ahead consistently**, and one-hot encoding underperformed native categorical handling across the board — worth knowing given how tempting it is to default to `pd.get_dummies()`. I kept the comparison in the notebook rather than deleting it, since "I tried the alternatives and here's why I didn't use them" is more convincing than just presenting the winner.

### 5. Tuning — and catching my own overfitting

This is the part of the process I think matters most, and I didn't want to sand it out of the final notebook.

I tuned manually first, then ran an Optuna search on top of it. Both landed in a tight cluster around **RMSE 1071–1075**, regardless of fairly different parameter combinations — a strong signal that the model had hit a real ceiling given the current feature set, not that I hadn't searched hard enough.

At one point, pushing `depth` up to 10 produced my *best-ever local CV score* (1071.23). On the actual leaderboard, that exact model scored **1083** — worse than almost every other version I'd submitted. That gap is a textbook overfitting signature: a deeper tree was finding patterns specific to my particular training folds that didn't generalize. I reverted to a shallower, more conservative configuration (`depth=3`), which had already shown *consistent* agreement between local CV and the leaderboard (1073.22 CV → 1074.48 leaderboard, later improved to ~1072 with additional regularization tuning). I trust that consistency more than a single flattering number.

**Final model configuration:**
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

### 6. What didn't work — the ensemble

I tried averaging this CatBoost model's predictions with an XGBoost model's predictions, expecting the usual ensembling bump. It scored **worse** on the leaderboard (1072.34) than CatBoost alone. The reason: a simple average blends a strong model with a meaningfully weaker one, which pulls the result toward the *weaker* model rather than improving on the stronger one. Ensembling only reliably helps when the models being combined are similarly strong to begin with — I didn't have that here, so I dropped the ensemble and stuck with CatBoost on its own.

### 7. Validation strategy throughout

Every claim in this notebook — every feature, every parameter choice, every "this didn't help" — was checked against **5-fold cross-validation**, not a single train/test split and not gut feeling. That's also what caught the depth=10 overfitting case before it became my final submission by accident.

---

## Results

- **Local CV RMSE:** ~1072
- **Leaderboard RMSE:** ~1072
- The close agreement between the two is the result I'm most confident in — it means the score reflects a model that actually generalizes, not one that got lucky on a particular validation split.

## Repo structure