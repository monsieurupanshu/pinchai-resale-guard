# Model Evaluation

PR-AUC: **0.999**  |  Precision: **0.967**  |  Recall: **1.000**  (decision threshold 0.5)

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
