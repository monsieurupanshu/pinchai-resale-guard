"""
Policy Engine for Resale-Guard.

Converts a raw model score (0-1) into an actual decision — ALLOW / FLAG /
LIMIT_QTY / BLOCK — with human-readable reason codes. This is the
"product" layer: real fraud ops teams act on explainable decisions, not
bare probabilities.

Two layers of logic:
  1. Score-based thresholds (the default path)
  2. Hard-override rules that can escalate a decision regardless of score
     — this reflects real systems, where certain patterns (e.g. a brand-
     new cluster of accounts converging on one address) warrant review
     even if the model itself is under-confident. It also documents that
     pure ML isn't treated as the whole answer.
"""

import os

import lightgbm as lgb
import pandas as pd

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))

# Score thresholds — tuned against this project's actual score distribution
# (see docs/01_model_eval.md): most customers score near 0 or near 1, with
# a small number of genuinely ambiguous cases in between.
THRESHOLDS = {
    "ALLOW": 0.30,      # below this: no action
    "FLAG": 0.60,       # 0.30-0.60: flag for manual review
    "LIMIT_QTY": 0.85,  # 0.60-0.85: allow but cap quantity
    # >= 0.85: BLOCK
}

ACTIONS = ["ALLOW", "FLAG", "LIMIT_QTY", "BLOCK"]


def score_to_action(score: float) -> str:
    if score < THRESHOLDS["ALLOW"]:
        return "ALLOW"
    elif score < THRESHOLDS["FLAG"]:
        return "FLAG"
    elif score < THRESHOLDS["LIMIT_QTY"]:
        return "LIMIT_QTY"
    else:
        return "BLOCK"


def generate_reason_codes(row: pd.Series, score: float, action: str) -> list:
    """Plain-language reasons a reviewer can actually read — not a dump
    of every feature value, just the ones that meaningfully contributed."""
    reasons = []

    if row["avg_qty_per_order"] >= 8:
        reasons.append(f"High average order quantity ({row['avg_qty_per_order']:.1f} units/order)")
    elif row["avg_qty_per_order"] >= 3.5:
        reasons.append(f"Elevated average order quantity ({row['avg_qty_per_order']:.1f} units/order)")

    if row["pct_orders_discount_window"] >= 0.8:
        reasons.append(f"{row['pct_orders_discount_window']:.0%} of orders placed during active discount windows")

    if row["identity_cluster_size"] >= 3:
        reasons.append(
            f"Linked to {int(row['identity_cluster_size']) - 1} other account(s) via shared "
            f"device/payment/address"
        )

    if row["sku_concentration"] >= 0.85:
        reasons.append(f"Purchases concentrated on very few SKUs (concentration {row['sku_concentration']:.2f})")

    if row["account_age_at_first_order_days"] < 30 and row["avg_qty_per_order"] >= 3:
        reasons.append(f"New account ({row['account_age_at_first_order_days']:.0f} days old) with bulk buying")

    if not reasons:
        if action == "ALLOW":
            reasons.append("No risk signals detected — typical purchase pattern")
        else:
            reasons.append(f"Model score {score:.2f} exceeded threshold, though no single dominant signal")

    return reasons


def apply_hard_overrides(row: pd.Series, action: str, reasons: list) -> tuple:
    """Escalate regardless of model score for patterns real fraud ops
    teams would want a human to see, even if the model itself is
    under-confident (e.g. a brand-new coordinated cluster that hasn't
    built enough purchase history yet for the model to be sure)."""

    action_rank = {a: i for i, a in enumerate(ACTIONS)}

    # Override 1: new account, part of an identity cluster of 3+, already
    # escalate to at least FLAG — a fresh coordinated cluster is worth a
    # human look even before it accumulates a long order history.
    if row["identity_cluster_size"] >= 3 and row["account_age_at_first_order_days"] < 45:
        if action_rank[action] < action_rank["FLAG"]:
            action = "FLAG"
            reasons = reasons + ["OVERRIDE: new account within a large identity cluster — escalated for review"]

    # Override 2: an extreme single-order quantity spike always gets at
    # least LIMIT_QTY, even if the account's overall pattern looks mild —
    # protects against a one-off bulk-buy burst on an otherwise quiet
    # account.
    if row["max_qty_single_order"] >= 15:
        if action_rank[action] < action_rank["LIMIT_QTY"]:
            action = "LIMIT_QTY"
            reasons = reasons + [
                f"OVERRIDE: single order of {int(row['max_qty_single_order'])} units exceeds hard cap"
            ]

    return action, reasons


class PolicyEngine:
    def __init__(self, model_dir: str = MODEL_DIR):
        self.model = lgb.Booster(model_file=os.path.join(model_dir, "reseller_model.txt"))
        with open(os.path.join(model_dir, "feature_cols.txt")) as f:
            self.feature_cols = f.read().splitlines()

    def score(self, row: pd.Series) -> float:
        X = row[self.feature_cols].to_frame().T.astype(float)
        return float(self.model.predict(X)[0])

    def decide(self, row: pd.Series) -> dict:
        score = self.score(row)
        action = score_to_action(score)
        reasons = generate_reason_codes(row, score, action)
        action, reasons = apply_hard_overrides(row, action, reasons)
        return {
            "customer_id": row.get("customer_id", "unknown"),
            "score": round(score, 3),
            "action": action,
            "reasons": reasons,
        }


def demo():
    """Run the policy engine against a handful of real customers spanning
    every segment, so the decisions can be sanity-checked by eye."""
    engine = PolicyEngine()
    data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
    test_df = pd.read_csv(os.path.join(data_dir, "test_split.csv"))

    print("=" * 90)
    print("POLICY ENGINE DEMO — sample decisions across all segments")
    print("=" * 90)

    for seg in sorted(test_df["segment"].unique()):
        sample = test_df[test_df["segment"] == seg].iloc[0]
        result = engine.decide(sample)
        print(f"\n[{seg}]  customer={sample['customer_id']}")
        print(f"  Score: {result['score']}   Action: {result['action']}")
        for r in result["reasons"]:
            print(f"    - {r}")


if __name__ == "__main__":
    demo()
