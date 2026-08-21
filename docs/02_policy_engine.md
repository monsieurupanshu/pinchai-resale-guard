# Policy Engine

Converts the Phase 3 model's raw 0-1 score into an actual decision —
`ALLOW` / `FLAG` / `LIMIT_QTY` / `BLOCK` — with human-readable reason
codes. See `src/policy/engine.py`.

## Score Thresholds

Tuned against this project's actual score distribution (see
`docs/01_model_eval.md`), which is fairly polarized — most customers
score near 0.0 or near 1.0, with a small number of genuinely ambiguous
cases in between:

| Score range | Action |
|---|---|
| < 0.30 | ALLOW |
| 0.30 – 0.60 | FLAG (manual review) |
| 0.60 – 0.85 | LIMIT_QTY |
| ≥ 0.85 | BLOCK |

## Hard-Override Rules

Two rules can escalate a decision regardless of the model's score. This
reflects how real fraud ops systems work: certain patterns warrant a
human look even when the model itself is under-confident (e.g. a
brand-new cluster that hasn't accumulated enough order history yet).

1. **New account + large identity cluster** — if `identity_cluster_size`
   ≥ 3 and the account is less than 45 days old, escalate to at least
   `FLAG`. Catches a coordinated ring early, before it has enough
   purchase history for the model alone to be confident.
2. **Extreme single-order quantity** — if any single order is ≥ 15
   units, escalate to at least `LIMIT_QTY`, even if the account's
   overall pattern looks mild. Protects against a one-off bulk-buy burst
   on an otherwise quiet account.

## A Known, Deliberate Trade-off — Stated Honestly

Testing the overrides against the full test set surfaced a real cost:
**Override 1 causes 5 of 40 (12.5%) `shared_address_legit` customers to
be flagged**, even though the model itself correctly scored them near
0.0 as safe. This happens when a legitimate multi-person household
happens to include a newer account.

This is not being hidden as a flaw — it's a genuine, considered
trade-off:

- `FLAG` is not `BLOCK`. It routes the customer to human review, not
  denial. The cost is a review-queue item, not a lost sale.
- The alternative — not having this override — means a genuinely new
  coordinated ring could operate un-flagged until it accumulates enough
  order history for the model to catch it on signal alone, which could
  mean real fraud losses in the meantime.
- In a production setting, this threshold (45 days, cluster size 3)
  would be tuned against the actual cost of a review vs. the cost of a
  missed ring — a business decision, not a purely technical one.

## Full Action Distribution (test set, n=460)

| Segment | n | ALLOW | FLAG | LIMIT_QTY | BLOCK |
|---|---|---|---|---|---|
| loyal_bulk | 62 | 62 | 0 | 0 | 0 |
| normal | 240 | 240 | 0 | 0 | 0 |
| ring_reseller | 46 | 0 | 0 | 0 | 46 |
| shared_address_legit | 40 | 35 | 5 | 0 | 0 |
| solo_reseller | 49 | 0 | 0 | 0 | 49 |
| stealth_reseller | 23 | 0 | 1 | 0 | 22 |

Every reseller segment ends up in `FLAG` or `BLOCK` — none silently
`ALLOW`ed. Every legitimate segment stays overwhelmingly in `ALLOW`,
with the one documented exception above.
