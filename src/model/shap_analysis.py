import os

import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
import shap

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))
IMG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "docs", "images"))


def load_model_and_data():
    model = lgb.Booster(model_file=os.path.join(MODEL_DIR, "reseller_model.txt"))
    with open(os.path.join(MODEL_DIR, "feature_cols.txt")) as f:
        feature_cols = f.read().splitlines()
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test_split.csv"))
    return model, feature_cols, test_df

def compute_shap_values(model, feature_cols, test_df):
    """TreeExplainer is exact (not approximated) for tree-based models
    like LightGBM — it computes true Shapley values efficiently by
    exploiting the tree structure, rather than the sampling-based
    approximation SHAP uses for arbitrary black-box models."""
    explainer = shap.TreeExplainer(model)
    X_test = test_df[feature_cols]
    shap_values = explainer.shap_values(X_test)
    return explainer, shap_values, X_test