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