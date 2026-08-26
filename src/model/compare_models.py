import os

import lightgbm as lgb
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_score, recall_score
from sklearn.preprocessing import StandardScaler

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))
LABEL_COLS = ["customer_id", "segment", "is_reseller", "is_loyal_bulk"]
THRESHOLD = 0.5

def load_split():
    train_df = pd.read_csv(os.path.join(DATA_DIR, "train_split.csv"))
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test_split.csv"))
    with open(os.path.join(MODEL_DIR, "feature_cols.txt")) as f:
        feature_cols = f.read().splitlines()
    return train_df, test_df, feature_cols

def evasive_recall(test_df, y_pred):
    """Recall specifically on stealth_reseller + ring_reseller — the two
    segments designed to evade obvious signals. This is the metric that
    actually distinguishes a good model from a shortcut-taking one here."""
    mask = test_df["segment"].isin(["stealth_reseller", "ring_reseller"])
    y_true_hard = test_df.loc[mask, "is_reseller"]
    y_pred_hard = pd.Series(y_pred, index=test_df.index).loc[mask]
    return recall_score(y_true_hard, y_pred_hard)


def legit_false_positive_rate(test_df, y_pred):
    """False-positive rate specifically on loyal_bulk + shared_address_legit
    — the cost of getting evasive-recall too aggressive."""
    mask = test_df["segment"].isin(["loyal_bulk", "shared_address_legit"])
    y_pred_legit = pd.Series(y_pred, index=test_df.index).loc[mask]
    return y_pred_legit.mean()

def main():
    train_df, test_df, feature_cols = load_split()

    X_train, y_train = train_df[feature_cols], train_df["is_reseller"]
    X_test, y_test = test_df[feature_cols], test_df["is_reseller"]

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    n_pos = y_train.sum()
    n_neg = len(y_train) - n_pos
    spw = n_neg / n_pos

    scores = {}

    # 1. Logistic Regression — interpretable linear baseline
    lr = LogisticRegression(max_iter=1000, class_weight="balanced", random_state=42)
    lr.fit(X_train_scaled, y_train)
    scores["Logistic Regression"] = lr.predict_proba(X_test_scaled)[:, 1]

    # 2. Random Forest — bagged trees
    rf = RandomForestClassifier(n_estimators=300, max_depth=8, class_weight="balanced",
                                 random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    scores["Random Forest"] = rf.predict_proba(X_test)[:, 1]

    # 3. LightGBM — matches src/model/train.py config, our current production model
    lgbm = lgb.LGBMClassifier(n_estimators=300, max_depth=5, learning_rate=0.05,
                               scale_pos_weight=spw, random_state=42, verbosity=-1)
    lgbm.fit(X_train, y_train)
    scores["LightGBM"] = lgbm.predict_proba(X_test)[:, 1]

    results = {}
    for name, score in scores.items():
        pred = (score >= THRESHOLD).astype(int)
        results[name] = {
            "PR-AUC": average_precision_score(y_test, score),
            "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred),
            "Evasive Recall": evasive_recall(test_df, pred),
            "Legit FP Rate": legit_false_positive_rate(test_df, pred),
        }

    results_df = pd.DataFrame(results).T.round(3)
    print(results_df.to_string())

    winner = results_df["Evasive Recall"].idxmax()
    print(f"\nWinner by evasive-case recall: {winner}")


if __name__ == "__main__":
    main()