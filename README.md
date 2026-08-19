
# Resale-Guard

**Checkout-time detection of discount-arbitrage resellers** — extending PinchAI's
post-purchase buyer-intent thesis one step earlier, to the moment of purchase.

---

## The Problem

Retailers running time-boxed discount promotions (flash sales, seasonal
clearance) lose margin to buyers who purchase in bulk — not for personal use,
but to resell at a markup on secondary platforms shortly after. This behavior
is invisible to traditional fraud systems because **no rule is technically
broken**: it's a legitimate purchase, correctly paid for, correctly
delivered. The harm is economic — margin erosion, distorted "sold out"
signals, and unfair inventory access for genuine customers — not
transactional.

## Why This Problem (and why it extends PinchAI's thesis)

PinchAI's product connects signals across checkout, return initiation, and
warehouse operations to form a single view of buyer intent — but that lens is
currently anchored at *returns*. The bet this project makes: **the same
buyer-intent lens, applied one step earlier at checkout, catches a class of
abuse a returns-focused system structurally cannot see**, because the item
in question is never returned. This isn't a generic fraud model — it's a
proposed extension to a gap in an existing product surface.

## What Success Looks Like

The easy cases (obvious bulk resellers) are table stakes. The real test —
and the reason the synthetic data includes deliberately hard pairs — is:

- **Not flagging legitimate bulk/shared-identity customers** (a loyal
  repeat buyer, a household sharing one shipping address)
- **Still catching resellers who evade the obvious signals** (spreading
  purchases across non-sale days, splitting volume across coordinated
  accounts to stay under per-account limits)

A model that only catches the obvious cases isn't proving much. See
[`docs/00_problem_framing.md`](docs/00_problem_framing.md) for the full
scoping discussion, including explicit in/out-of-scope decisions.

## Architecture

```
1. Synthetic Data Generation   → 6 customer archetypes, realistic noise
2. Feature Engineering          → purchase / identity / network signals
3. Detection Model              → LightGBM, evaluated per-segment
4. Policy Engine                → score → action, with reason codes
5. Dashboard                    → Streamlit console + trust score +
                                    LLM-generated reviewer narratives
6. Docs                         → design rationale, eval results,
                                    roadmap (incl. agentic investigation
                                    layer as a documented next step)
```

## Status

- [x] Phase 0 — Problem framing
- [ ] Phase 1 — Synthetic data generation (6 customer archetypes)
- [ ] Phase 2 — Feature engineering
- [ ] Phase 3 — Detection model (LightGBM)
- [ ] Phase 4 — Policy engine
- [ ] Phase 5 — Dashboard (Streamlit)
- [ ] Phase 6 — Final docs + polish

## Quickstart

```bash
pip install -r requirements.txt
python3 src/data_gen/generate.py     # generates data/*.csv
python3 src/features/build_features.py  # generates data/features.csv
```

## Repo Structure

```
src/
  data_gen/     synthetic data generation
  features/     feature engineering
  model/        training + evaluation (Phase 3)
  policy/       score-to-action engine (Phase 4)
  dashboard/    Streamlit app (Phase 5)
docs/           design rationale, framing, roadmap
data/           generated CSVs (not versioned — regenerate via scripts)
```

## A Note on Scope

This is a self-directed project, not an official PinchAI assignment — built
after a March 2026 interview to explore a problem we discussed together, and
to demonstrate how I approach ambiguous, product-shaped ML problems
end-to-end. Explicit out-of-scope items (cross-retailer identity resolution,
adversarial evasion, live drift monitoring, privacy/compliance architecture)
are documented rather than silently skipped — see `docs/00_problem_framing.md`.
