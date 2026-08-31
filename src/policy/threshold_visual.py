import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))
IMG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "docs", "images"))


def load_scored_data():
    model = lgb.Booster(model_file=os.path.join(MODEL_DIR, "reseller_model.txt"))
    with open(os.path.join(MODEL_DIR, "feature_cols.txt")) as f:
        feature_cols = f.read().splitlines()
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test_split.csv"))
    test_df["score"] = model.predict(test_df[feature_cols])
    return test_df

def plot_threshold_zones(test_df):
    """Shows the score axis with the 4 action zones marked, and every
    real customer's actual score plotted on it — makes the threshold
    logic immediately visible instead of read as a table."""
    from src.policy.engine import THRESHOLDS

    segments = ["normal", "loyal_bulk", "shared_address_legit",
                "solo_reseller", "stealth_reseller", "ring_reseller"]
    colors = {"normal": "#2a78d6", "loyal_bulk": "#1baf7a", "shared_address_legit": "#199e70",
              "solo_reseller": "#e34948", "stealth_reseller": "#eb6834", "ring_reseller": "#d03b3b"}

    fig, ax = plt.subplots(figsize=(12, 5))

    # zone backgrounds
    zone_bounds = [0, THRESHOLDS["ALLOW"], THRESHOLDS["FLAG"], THRESHOLDS["LIMIT_QTY"], 1.0]
    zone_names = ["ALLOW", "FLAG", "LIMIT_QTY", "BLOCK"]
    zone_colors = ["#d4f7dc", "#fff3cd", "#ffe0b3", "#f8d7da"]
    for i in range(4):
        ax.axvspan(zone_bounds[i], zone_bounds[i + 1], color=zone_colors[i], alpha=0.6)
        mid = (zone_bounds[i] + zone_bounds[i + 1]) / 2
        ax.text(mid, len(segments) + 0.3, zone_names[i], ha="center", fontsize=10, fontweight="bold")

    # jittered scatter per segment, one row per segment
    rng = np.random.default_rng(42)
    for i, seg in enumerate(segments):
        scores = test_df[test_df["segment"] == seg]["score"].values
        y = i + rng.uniform(-0.15, 0.15, size=len(scores))
        ax.scatter(scores, y, color=colors[seg], s=25, alpha=0.7, edgecolor="white", linewidth=0.3)

    ax.set_yticks(range(len(segments)))
    ax.set_yticklabels(segments)
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.5, len(segments) + 0.7)
    ax.set_xlabel("Model Score")
    ax.set_title("Policy Thresholds vs. Real Customer Scores (test set)")
    ax.grid(axis="x", alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(IMG_DIR, "threshold_zones.png"), dpi=150)
    plt.close(fig)
    print("Saved docs/images/threshold_zones.png")


if __name__ == "__main__":
    os.makedirs(IMG_DIR, exist_ok=True)
    test_df = load_scored_data()
    plot_threshold_zones(test_df)