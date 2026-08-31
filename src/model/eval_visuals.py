import os

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, precision_recall_curve, average_precision_score

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))
IMG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "docs", "images"))


def load_predictions():
    model = lgb.Booster(model_file=os.path.join(MODEL_DIR, "reseller_model.txt"))
    with open(os.path.join(MODEL_DIR, "feature_cols.txt")) as f:
        feature_cols = f.read().splitlines()
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test_split.csv"))
    test_df["score"] = model.predict(test_df[feature_cols])
    test_df["pred"] = (test_df["score"] >= 0.5).astype(int)
    return test_df