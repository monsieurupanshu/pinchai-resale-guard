"""
Core logic for the Resale-Guard dashboard, kept separate from the
Streamlit UI code so it can be tested directly without launching a
server.

Two things live here beyond the policy engine's score/action/reasons:

1. TRUST SCORE — a positively-framed companion score for legitimate
   customers. This directly mirrors PinchAI's own stated product
   philosophy: don't just catch bad actors, actively recognize good ones
   (their public claim is +20% VIP retention alongside reduced return
   rates). A risk score alone only tells half the story.

2. NARRATIVE — a plain-English case summary a reviewer can read in
   seconds instead of parsing a feature table. Works with or without an
   Anthropic API key: if ANTHROPIC_API_KEY is set, it calls the real
   model; otherwise it falls back to a template. This means the
   dashboard is never blocked on API access to actually run.
"""

import os

import pandas as pd


def compute_trust_score(row: pd.Series, risk_score: float) -> int:
    """0-100 trust score. Starts from inverse risk, then adds bonus
    points for positive tenure/loyalty signals — so two customers with
    the same near-zero risk score can still be differentiated (a
    20-order, 2-year veteran vs. a first-time shopper both score "safe,"
    but only one has earned real trust)."""
    base = (1 - risk_score) * 70  # up to 70 points from being low-risk

    tenure_bonus = min(row.get("account_age_at_first_order_days", 0) / 500 * 15, 15)
    loyalty_bonus = min(row.get("n_orders", 0) / 10 * 10, 10)
    diversity_bonus = min((1 - row.get("sku_concentration", 1)) * 5, 5)

    score = base + tenure_bonus + loyalty_bonus + diversity_bonus
    return int(round(min(max(score, 0), 100)))


def generate_trust_signals(row: pd.Series) -> list:
    """Positive-framed signals for customers who are NOT flagged —
    the mirror image of the policy engine's risk reason codes."""
    signals = []

    age = row.get("account_age_at_first_order_days", 0)
    if age >= 365:
        signals.append(f"Established customer — account is {age:.0f} days old")
    elif age >= 90:
        signals.append(f"Returning customer — account is {age:.0f} days old")

    if row.get("n_orders", 0) >= 4:
        signals.append(f"Repeat purchaser — {int(row['n_orders'])} orders on record")

    if row.get("sku_concentration", 1) < 0.5:
        signals.append("Diverse purchase history across multiple products")

    if row.get("identity_cluster_size", 1) == 1:
        signals.append("No shared identity signals with any other account")

    if row.get("pct_orders_discount_window", 0) == 0:
        signals.append("Purchases not concentrated around discount events")

    if not signals:
        signals.append("Standard purchase pattern, no notable history yet")

    return signals


NARRATIVE_MODEL = "claude-haiku-4-5-20251001"  # fast/cheap — narratives are short, high-volume


def generate_narrative(row: pd.Series, decision: dict, api_key: str = None) -> str:
    """Plain-English case summary. Uses the real Anthropic API if a key
    is available; otherwise falls back to a template so the dashboard
    never breaks without one."""
    api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")

    if api_key:
        try:
            return _generate_narrative_llm(row, decision, api_key)
        except Exception as e:
            return _generate_narrative_template(row, decision) + f"\n\n_(LLM narrative unavailable: {e})_"
    return _generate_narrative_template(row, decision)


def _generate_narrative_llm(row: pd.Series, decision: dict, api_key: str) -> str:
    import anthropic

    client = anthropic.Anthropic(api_key=api_key)

    summary_facts = (
        f"Customer {decision['customer_id']}, segment (internal, not for reviewer): {row.get('segment', 'unknown')}. "
        f"Risk score: {decision['score']}. Decision: {decision['action']}. "
        f"Total orders: {row.get('n_orders', 0)}, avg quantity per order: {row.get('avg_qty_per_order', 0):.1f}, "
        f"% orders during discount windows: {row.get('pct_orders_discount_window', 0):.0%}, "
        f"SKU concentration: {row.get('sku_concentration', 0):.2f}, "
        f"account age at first order: {row.get('account_age_at_first_order_days', 0):.0f} days, "
        f"identity cluster size (accounts sharing device/payment/address): {int(row.get('identity_cluster_size', 1))}. "
        f"Reason codes: {'; '.join(decision['reasons'])}."
    )

    message = client.messages.create(
        model=NARRATIVE_MODEL,
        max_tokens=200,
        messages=[{
            "role": "user",
            "content": (
                "You are writing a short case note for a fraud-ops reviewer at an "
                "e-commerce retailer. Given these facts, write a 2-3 sentence plain-English "
                "summary explaining the decision. Be factual and specific, cite the actual "
                "numbers, no hedging language, no preamble. Facts:\n\n" + summary_facts
            ),
        }],
    )
    return message.content[0].text


def _generate_narrative_template(row: pd.Series, decision: dict) -> str:
    action = decision["action"]
    reasons_str = "; ".join(decision["reasons"])

    if action == "ALLOW":
        return (
            f"This customer's purchase pattern shows no significant risk signals "
            f"(score: {decision['score']}). {reasons_str}."
        )
    elif action == "BLOCK":
        return (
            f"This customer was blocked with a high risk score of {decision['score']}. "
            f"Key factors: {reasons_str}. Recommend confirming before any manual override."
        )
    elif action == "LIMIT_QTY":
        return (
            f"This customer's order was allowed but quantity-limited (score: {decision['score']}). "
            f"Contributing factors: {reasons_str}."
        )
    else:  # FLAG
        return (
            f"This customer was flagged for manual review (score: {decision['score']}). "
            f"Factors warranting a second look: {reasons_str}."
        )
