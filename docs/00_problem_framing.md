# Problem Framing

## 1. Problem Statement

Retailers running time-boxed discount promotions (flash sales, seasonal
clearance) lose margin to buyers who purchase in bulk not for personal use,
but to resell at a markup on secondary platforms. This behavior is invisible
to traditional fraud systems because no rule is technically broken — it's a
legitimate purchase, correctly paid for, correctly delivered. The harm is
economic (margin erosion, distorted "sold out" signals, unfair inventory
access for genuine customers), not transactional.

## 2. Why This Problem, Specifically

PinchAI's stated thesis connects signals across checkout, return initiation,
and warehouse operations to form a single view of buyer intent — but the
current product surface is anchored at *returns*. This project's bet: the
same buyer-intent lens, applied one step earlier at checkout, catches a
class of abuse a returns-focused system structurally cannot see, since the
item is never returned. The goal isn't "a fraud model" — it's identifying a
gap in an existing product's own stated thesis and building the missing
piece.

## 3. What Success Looks Like

Defined before building, not retrofitted after:

- A model that separates `solo_reseller` / `ring_reseller` /
  `stealth_reseller` from `normal` / `loyal_bulk` / `shared_address_legit` —
  where the **hard pairs matter more than the easy ones**. Beating the easy
  cases is table stakes; the real test is not misflagging
  `shared_address_legit` (which looks identical to a ring on the network
  signal alone) and not missing `stealth_reseller` (which looks identical to
  a loyal buyer on quantity and timing alone).
- A policy layer producing an actionable, explainable decision — not just a
  raw probability.
- A design doc that shows the reasoning, not just the artifact.

## 4. Scope Boundaries

**In scope:**
- Checkout-time detection using signals available at time of purchase
- Synthetic data with realistic, deliberately-hard archetypes
- Single-retailer assumption (no cross-retailer data sharing)

**Out of scope (acknowledged, not built):**
- **Cross-retailer identity resolution** — a real PinchAI deployment serves
  many retailers; device/address fingerprints would need to stay scoped
  per-retailer (or explicitly cross-retailer with consent) to avoid one
  retailer's legitimate customer being flagged due to unrelated activity
  elsewhere.
- **Adversarial evasion / arms race** — VPNs, device spoofing, and other
  active evasion once resellers suspect detection exists. A static model
  degrades over time against an adaptive adversary; this would need a
  monitoring + retraining cadence in production.
- **Privacy/legal compliance architecture** — device fingerprinting and
  cross-account identity linking touch real consent and data-minimization
  requirements (GDPR/CCPA-style). A production system would need
  consent-scoped identity linking; this project does not model that layer.
- **Concept drift / live retraining** — the model here is trained once on a
  static synthetic snapshot. A "what I'd do at scale" note covers this in
  the final docs, but no retraining pipeline is built.

## 5. The Core Assumption Worth Naming

Everything downstream assumes the 6-archetype synthetic data is a fair proxy
for real buyer behavior. That's a reasonable assumption for a self-directed
take-home — real transactional data isn't available — but it's worth stating
explicitly as a limitation rather than treating the synthetic data as ground
truth about the real world. The archetypes were designed adversarially
against each other (see below) specifically to avoid the trap of a model
that only performs well because the synthetic data made the problem easy.

## 6. The Six Archetypes and Why Each Exists

| Archetype | Purpose |
|---|---|
| `normal` | Baseline — no bulk, no pattern |
| `loyal_bulk` | **False-positive test.** Legitimate high-volume buyer (spread across time, diverse SKUs, established account) — must NOT be flagged |
| `shared_address_legit` | **False-positive test.** Unrelated cohabiting customers (roommates/family) sharing one address but independent everything else — must NOT be flagged despite triggering the network signal |
| `solo_reseller` | Easy positive — single account, concentrated bulk, timed to discount windows |
| `stealth_reseller` | **False-negative test.** Resells but spreads purchases across non-sale days too, defeating a timing-only detector |
| `ring_reseller` | Easy-to-miss positive if only judged per-account — coordinated accounts, each individually unremarkable, connected via a shared identity graph |

Two pairs are deliberately close to each other in feature space:
`ring_reseller` vs. `shared_address_legit` (similar network signal, differ
on timing + account age), and `stealth_reseller` vs. `loyal_bulk` (similar
quantity/timing, differ on SKU concentration + account age). A model that
can't separate these pairs isn't learning the right thing.
