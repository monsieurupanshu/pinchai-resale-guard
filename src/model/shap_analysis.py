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

def plot_global_summary(shap_values, X_test):
    """Global view: which features matter most, and in which direction.
    Unlike plain gain-based importance, this shows whether HIGH values
    of a feature push toward reseller (red) or toward legit (blue)."""
    fig = plt.figure(figsize=(9, 7))
    shap.summary_plot(shap_values, X_test, show=False, plot_size=None)
    plt.title("SHAP Summary — Feature Impact Direction", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(IMG_DIR, "shap_summary.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("Saved docs/images/shap_summary.png")


def plot_customer_waterfall(explainer, shap_values, X_test, test_df, customer_id):
    """Per-customer view: exactly how each feature pushed THIS specific
    customer's score up or down from the model's baseline expectation."""
    idx = test_df[test_df["customer_id"] == customer_id].index
    if len(idx) == 0:
        print(f"Customer {customer_id} not found in test set")
        return
    row_pos = test_df.index.get_loc(idx[0])

    fig = plt.figure(figsize=(9, 6))
    shap.waterfall_plot(
        shap.Explanation(
            values=shap_values[row_pos],
            base_values=explainer.expected_value,
            data=X_test.iloc[row_pos],
            feature_names=X_test.columns.tolist(),
        ),
        show=False,
    )
    plt.title(f"SHAP Waterfall — {customer_id}", fontsize=12)
    plt.tight_layout()
    safe_name = customer_id.replace("-", "_")
    plt.savefig(os.path.join(IMG_DIR, f"shap_waterfall_{safe_name}.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved docs/images/shap_waterfall_{safe_name}.png")