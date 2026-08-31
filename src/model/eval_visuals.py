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

def plot_confusion_matrix(test_df):
    cm = confusion_matrix(test_df["is_reseller"], test_df["pred"])
    fig, ax = plt.subplots(figsize=(5, 4.5))
    im = ax.imshow(cm, cmap="Blues")

    labels = ["Legit", "Reseller"]
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix (test set, threshold=0.5)")

    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "confusion_matrix.png"), dpi=150)
    plt.close(fig)
    print("Saved docs/images/confusion_matrix.png")
    
def plot_pr_curve(test_df):
    precision, recall, _ = precision_recall_curve(test_df["is_reseller"], test_df["score"])
    ap = average_precision_score(test_df["is_reseller"], test_df["score"])

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(recall, precision, color="#2a78d6", linewidth=2)
    ax.fill_between(recall, precision, alpha=0.15, color="#2a78d6")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(f"Precision-Recall Curve (AP = {ap:.3f})")
    ax.set_xlim([0, 1.02])
    ax.set_ylim([0, 1.02])
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "pr_curve.png"), dpi=150)
    plt.close(fig)
    print("Saved docs/images/pr_curve.png")