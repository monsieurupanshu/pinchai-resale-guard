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