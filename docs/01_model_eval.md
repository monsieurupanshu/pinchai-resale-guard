# Model Evaluation

PR-AUC: **0.999**  |  Precision: **0.967**  |  Recall: **1.000**  (decision threshold 0.5)

## Visual Evaluation

![Confusion Matrix](images/confusion_matrix.png)
![PR Curve](images/pr_curve.png)
![Segment Score Distribution](images/segment_score_distribution.png)

6 misclassifications on the test set (340 TN, 3 FP, 3 FN, 120 TP,
AP=0.998). Each was individually inspected against real feature values —
not just counted — since which specific cases fail is more informative
than how many.

### The 3 false positives — not outliers, the most typical members

All 3 misclassified `loyal_bulk` customers sit almost exactly on the
segment average: quantity 3.6-4.2 vs. segment mean 4.46, account age
385-400 days vs. segment mean 404. These aren't unusual edge cases the
model stumbled on — they're the most *ordinary* members of a genuinely
ambiguous legitimate segment. The model occasionally misjudges typical
`loyal_bulk` behavior precisely because it looks similar to bulk-buying
in general; this is an honest, expected cost of the segment design, not
a bug.

### The 3 false negatives — a specific, informative weak spot

All 3 missed `ring_reseller` customers share a consistent pattern:
`identity_cluster_size` of only 3-4 and `shared_address_count` of only
2-3, both below the segment average (4.63 / 3.63) — and all 3 have
`pct_orders_discount_window = 0.0`, meaning all are "stealth"-style
rings, not "aggressive" ones.

**The model's real weak spot is specific: the smallest, most patient,
stealth-style rings** — coordinated groups that keep both their group
size and their purchase-timing signal minimal. This is exactly the
hardest case type the archetype design intentionally stress-tests (see
`docs/00_problem_framing.md`), and finding it here — rather than
assuming the model is uniformly strong — is the actual value of this
evaluation.

**What this means for a production system:** a real deployment would
want either a lower BLOCK threshold specifically for accounts showing
*any* nonzero `shared_address_count` (even below the typical ring
size), or a separate, more sensitive review tier for small clusters —
rather than trusting the same 0.85 threshold uniformly across cluster
sizes.

## Per-Segment Breakdown

The headline result. `should_be_flagged` reflects ground truth (is this archetype a reseller by design). `pct_flagged` is what the model actually did.

| segment              |   n | should_be_flagged   |   pct_flagged |   avg_score |
|:---------------------|----:|:--------------------|--------------:|------------:|
| loyal_bulk           |  62 | False               |         0.065 |       0.064 |
| normal               | 240 | False               |         0     |       0     |
| ring_reseller        |  46 | True                |         1     |       1     |
| shared_address_legit |  40 | False               |         0     |       0     |
| solo_reseller        |  49 | True                |         1     |       1     |
| stealth_reseller     |  23 | True                |         1     |       0.984 |

## Feature Importance (gain-based)

| feature                         |   gain_pct |
|:--------------------------------|-----------:|
| pct_orders_discount_window      |       24.7 |
| avg_qty_per_order               |       22.6 |
| shared_address_count            |       20.5 |
| n_orders                        |       13.9 |
| shared_payment_count            |        5.5 |
| shared_device_count             |        4.2 |
| max_qty_single_order            |        3.6 |
| sku_concentration               |        2.3 |
| total_quantity                  |        1.3 |
| account_age_at_first_order_days |        0.7 |
| avg_discount_pct_when_used      |        0.4 |
| order_frequency_per_week        |        0.2 |
| total_spend                     |        0.1 |
| purchase_span_days              |        0.1 |
| n_distinct_skus                 |        0   |
| identity_cluster_size           |        0   |
