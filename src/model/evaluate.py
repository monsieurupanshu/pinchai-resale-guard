"""
Evaluate the reseller-detection model.

The headline output is NOT a single accuracy number — it's a per-segment
breakdown, because the whole point of this project is proving the model
doesn't punish legitimate bulk/shared-address customers while still
catching resellers who evade the obvious signals (timing, per-account
quantity limits).

Aggregate metrics (PR-AUC) are reported too, but the per-segment table is
what actually answers the question this project set out to answer.
"""

import os

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score, precision_score, recall_score,
    confusion_matrix, classification_report,
)

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))
DOCS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "docs"))

DECISION_THRESHOLD = 0.5  # used only for the segment-level ALLOW/FLAG view here;
                           # Phase 4's policy engine will use its own thresholds


def load_model_and_data():
    model = lgb.Booster(model_file=os.path.join(MODEL_DIR, "reseller_model.txt"))
    with open(os.path.join(MODEL_DIR, "feature_cols.txt")) as f:
        feature_cols = f.read().splitlines()
    test_df = pd.read_csv(os.path.join(DATA_DIR, "test_split.csv"))
    return model, feature_cols, test_df


def per_segment_report(test_df: pd.DataFrame, y_true: pd.Series, y_pred: np.ndarray, y_score: np.ndarray) -> pd.DataFrame:
    df = test_df.copy()
    df["y_true"] = y_true.values
    df["y_pred"] = y_pred
    df["y_score"] = y_score

    rows = []
    for seg, g in df.groupby("segment"):
        n = len(g)
        flagged = g["y_pred"].sum()
        flag_rate = flagged / n
        avg_score = g["y_score"].mean()
        # ground-truth intent: is this segment supposed to be flagged?
        should_flag = g["y_true"].iloc[0] == 1
        rows.append({
            "segment": seg,
            "n": n,
            "should_be_flagged": should_flag,
            "pct_flagged": round(flag_rate, 3),
            "avg_score": round(avg_score, 3),
        })
    return pd.DataFrame(rows).sort_values("segment")


def main():
    model, feature_cols, test_df = load_model_and_data()

    X_test = test_df[feature_cols]
    y_test = test_df["is_reseller"]

    y_score = model.predict(X_test)
    y_pred = (y_score >= DECISION_THRESHOLD).astype(int)

    print("=" * 70)
    print("AGGREGATE METRICS")
    print("=" * 70)
    pr_auc = average_precision_score(y_test, y_score)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    print(f"PR-AUC:    {pr_auc:.3f}")
    print(f"Precision: {precision:.3f}  (at threshold {DECISION_THRESHOLD})")
    print(f"Recall:    {recall:.3f}  (at threshold {DECISION_THRESHOLD})")

    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion matrix:\n{cm}")
    print("  [[TN  FP]\n   [FN  TP]]")

    print("\n" + "=" * 70)
    print("PER-SEGMENT BREAKDOWN  <-- the actual headline result")
    print("=" * 70)
    seg_report = per_segment_report(test_df, y_test, y_pred, y_score)
    print(seg_report.to_string(index=False))

    print("\nInterpretation:")
    for _, row in seg_report.iterrows():
        seg = row["segment"]
        flag_rate = row["pct_flagged"]
        should = row["should_be_flagged"]
        if not should and flag_rate > 0.10:
            verdict = f"  WARNING: {flag_rate:.0%} false-positive rate — legit segment over-flagged"
        elif not should:
            verdict = f"  GOOD: only {flag_rate:.0%} falsely flagged"
        elif should and flag_rate < 0.70:
            verdict = f"  WARNING: only {flag_rate:.0%} caught — recall gap on this reseller type"
        else:
            verdict = f"  GOOD: {flag_rate:.0%} caught"
        print(f"  {seg:24s} {verdict}")

    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE (gain-based)")
    print("=" * 70)
    importance = pd.DataFrame({
        "feature": feature_cols,
        "gain": model.feature_importance(importance_type="gain"),
    }).sort_values("gain", ascending=False)
    importance["gain_pct"] = (importance["gain"] / importance["gain"].sum() * 100).round(1)
    print(importance[["feature", "gain_pct"]].to_string(index=False))

    # write eval doc
    os.makedirs(DOCS_DIR, exist_ok=True)
    with open(os.path.join(DOCS_DIR, "01_model_eval.md"), "w") as f:
        f.write("# Model Evaluation\n\n")
        f.write(f"PR-AUC: **{pr_auc:.3f}**  |  Precision: **{precision:.3f}**  |  Recall: **{recall:.3f}**  "
                f"(decision threshold {DECISION_THRESHOLD})\n\n")
        f.write("## Per-Segment Breakdown\n\n")
        f.write("The headline result. `should_be_flagged` reflects ground truth "
                "(is this archetype a reseller by design). `pct_flagged` is what "
                "the model actually did.\n\n")
        f.write(seg_report.to_markdown(index=False))
        f.write("\n\n## Feature Importance (gain-based)\n\n")
        f.write(importance[["feature", "gain_pct"]].to_markdown(index=False))
        f.write("\n")
    print(f"\nWrote docs/01_model_eval.md")


if __name__ == "__main__":
    main()
