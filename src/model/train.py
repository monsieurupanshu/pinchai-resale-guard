"""
Train the reseller-detection model.

Key design choices, and why:

1. STRATIFIED SPLIT BY SEGMENT, not just by the binary `is_reseller` label.
   A plain stratified split on the binary label could easily put most
   `shared_address_legit` customers in train and almost none in test (or
   vice versa), silently hiding whether the model actually generalizes on
   the hard false-positive/false-negative pairs. Splitting by the 6-way
   `segment` guarantees every archetype is represented proportionally in
   both sets.

2. scale_pos_weight for class imbalance. Resellers are a minority class
   (~29% of customers here), which is realistic for fraud-adjacent
   problems. Without this, a model can get high accuracy by just
   predicting "not a reseller" for everyone.

3. Features going into the model are exactly the 16 non-label columns in
   features.csv. `segment` itself is NEVER a feature — it's the thing
   being predicted (via `is_reseller`), and including a proxy for it
   would be catastrophic leakage.
"""

import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))

LABEL_COLS = ["customer_id", "segment", "is_reseller", "is_loyal_bulk"]


def load_features(data_dir: str = DATA_DIR) -> pd.DataFrame:
    return pd.read_csv(os.path.join(data_dir, "features.csv"))


def split_data(df: pd.DataFrame, test_size: float = 0.25, random_state: int = 42):
    feature_cols = [c for c in df.columns if c not in LABEL_COLS]

    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=df["segment"],  # stratify by 6-way segment, not binary label
    )
    return train_df, test_df, feature_cols


def train_model(train_df: pd.DataFrame, feature_cols: list, random_state: int = 42) -> lgb.LGBMClassifier:
    X_train = train_df[feature_cols]
    y_train = train_df["is_reseller"]

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    scale_pos_weight = n_neg / n_pos

    model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        random_state=random_state,
        verbosity=-1,
    )
    model.fit(X_train, y_train)
    return model


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)

    df = load_features()
    print(f"Loaded {len(df)} customers, {df['is_reseller'].sum()} resellers "
          f"({df['is_reseller'].mean():.1%})")

    train_df, test_df, feature_cols = split_data(df)
    print(f"\nTrain: {len(train_df)}  Test: {len(test_df)}")
    print("\nSegment distribution — train vs test (should be proportional):")
    comp = pd.DataFrame({
        "train_pct": train_df["segment"].value_counts(normalize=True).round(3),
        "test_pct": test_df["segment"].value_counts(normalize=True).round(3),
    })
    print(comp)

    model = train_model(train_df, feature_cols)

    model_path = os.path.join(MODEL_DIR, "reseller_model.txt")
    model.booster_.save_model(model_path)
    print(f"\nSaved model to {model_path}")

    # persist train/test splits + feature column order for evaluate.py
    train_df.to_csv(os.path.join(DATA_DIR, "train_split.csv"), index=False)
    test_df.to_csv(os.path.join(DATA_DIR, "test_split.csv"), index=False)
    with open(os.path.join(MODEL_DIR, "feature_cols.txt"), "w") as f:
        f.write("\n".join(feature_cols))
    print("Saved train/test splits and feature column list.")


if __name__ == "__main__":
    main()
