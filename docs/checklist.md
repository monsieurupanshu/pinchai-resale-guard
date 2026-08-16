# Project Checklist — Resale-Guard

Tracks what's done, what's next, and *why* each phase exists. Use this as
the single source of truth instead of scrolling back through chat history.

---

## Phase 0 — Problem Framing ✅ DONE

**What:** Define the problem precisely, justify why it matters to PinchAI
specifically, define success criteria, set explicit scope boundaries.

**Why:** A Head of DS needs to see the *reasoning* before the code — this
is what separates "I built a fraud model" from "I found a gap in your
product surface and built the missing piece."

**How it was done:** Written up in `docs/00_problem_framing.md` — problem
statement, why-PinchAI rationale, success criteria (hard pairs matter more
than easy ones), in/out-of-scope table, archetype rationale table.

**Artifacts produced:** `docs/00_problem_framing.md`, `README.md`

---

## Phase 1 — Synthetic Data Generation ✅ DONE

**What:** Generate customers + orders + resale listings with 6 realistic
archetypes baked in, so we have honest ground truth.

**Why:** Real transactional data isn't available for a take-home. Synthetic
data lets us control ground truth (needed for evaluation) — but only if
the archetypes are designed adversarially against each other, not just
"obviously different," or the model's success would be trivial and
unconvincing.

**How it was done:** `src/data_gen/generate.py` — 6 segments:
- `normal` — baseline
- `loyal_bulk` — false-positive test (legit high volume)
- `shared_address_legit` — false-positive test (legit shared address)
- `solo_reseller` — easy positive
- `stealth_reseller` — false-negative test (evades timing signal)
- `ring_reseller` — false-negative test if judged per-account only

Resale listings generated but used **only** for ground-truth labels —
never fed to the model as a feature (would be label leakage; a real
checkout-time system doesn't have future resale-market data).

**Artifacts produced:** `data/customers.csv`, `data/orders.csv`,
`data/resale_listings.csv`, `data/skus.csv`, `data/discount_events.csv`,
`data/labels.csv`

**Validation done:** confirmed segment stats match design intent (e.g.
`loyal_bulk` at 0% discount-window purchases vs. resellers at 100%;
`shared_address_legit` avg qty matches `normal`; ring members share
exactly 1 address each).

---

## Phase 2 — Feature Engineering ✅ DONE

**What:** Convert raw orders/customers into a model-ready signal table.

**Why:** This is where the actual detection logic lives. The model is only
as good as these features — and this is where label-leakage discipline and
signal-design judgment are demonstrated.

**How it was done:** `src/features/build_features.py` — 3 feature groups:
- **Purchase:** total/avg quantity, SKU concentration (Herfindahl index),
  % orders during discount windows, purchase span, order frequency
- **Identity:** account age at first order
- **Network/graph:** shared device/payment/address counts, plus
  `identity_cluster_size` via `networkx` connected components (catches
  rings even when no single shared attribute looks unusual alone)

**Artifacts produced:** `data/features.csv`

**Validation done:** confirmed two deliberately hard pairs exist in
feature space — `ring_reseller` vs `shared_address_legit` (similar
network signal, differ on timing + account age) and `stealth_reseller`
vs `loyal_bulk` (similar quantity, differ on SKU concentration + account
age). This proves the model has to learn combinations, not shortcuts.

---

## Phase 3 — Detection Model ⏭️ NEXT

**What:** Train a classifier to score each customer 0–1 on reseller
likelihood.

**Why:** LightGBM matches the model family used in real fraud-detection
systems (and in your actual PinchAI interview prep) — gradient-boosted
trees handle tabular data with mixed feature types well, and support
class-imbalance handling natively.

**How to do it:**
1. Stratified train/test split **by segment** (not just binary label) —
   ensures `loyal_bulk` and `shared_address_legit` are represented in
   both sets
2. Train LightGBM with `scale_pos_weight` for class imbalance
3. Evaluate with PR-AUC (not just accuracy — meaningless on imbalanced
   data) and a **per-segment precision/recall breakdown** — the headline
   proof point:
   - Is `loyal_bulk` avoiding false flags?
   - Is `shared_address_legit` avoiding false flags?
   - Is `stealth_reseller` still being caught?
   - Is `ring_reseller` still being caught?
4. Feature importance report — which signals actually drove decisions
5. Save trained model artifact for Phases 4–5

**Artifacts to produce:** `src/model/train.py`, `src/model/evaluate.py`,
trained model file, `docs/01_model_eval.md` (results writeup)

**Definition of done:** per-segment eval table exists and shows the model
does NOT punish the two legit-but-suspicious-looking segments while still
catching the two evasive reseller segments.

---

## Phase 4 — Policy Engine ⏳ NOT STARTED

**What:** Convert a raw model score into an actual decision: `ALLOW` /
`FLAG` / `LIMIT_QTY` / `BLOCK`, with human-readable reason codes.

**Why:** This is the actual "product" layer — real fraud ops teams don't
work off a bare probability, they work off explainable decisions. It also
mirrors how PinchAI's own product is described (decisions + explanations,
not just scores).

**How to do it:**
1. Threshold-based score-to-action mapping
2. Hard-override rules layered on top (e.g., `identity_cluster_size` above
   some N forces manual review regardless of model score) — shows judgment
   that pure ML isn't always the full answer
3. Reason-code generation (which features/thresholds triggered the
   decision, in plain terms)

**Artifacts to produce:** `src/policy/engine.py`

**Definition of done:** given any customer's feature row, the engine
returns an action + a short list of human-readable reasons.

---

## Phase 5 — Dashboard + Differentiators ⏳ NOT STARTED

**What:** Streamlit console — customer lookup, score, signal breakdown,
trust score for good customers, LLM-generated reviewer narrative.

**Why:** This is the 5-minute walkthrough surface — what the interviewer
actually sees and remembers. The trust-score + narrative combo directly
echoes PinchAI's own stated product philosophy (catch bad actors without
punishing good ones).

**How to do it (in priority order, given the timebox):**
1. Static view — search a customer, see score, reason codes, action
2. Trust score — a companion, positively-framed score for legitimate high
   -volume customers (`loyal_bulk`, `shared_address_legit`)
3. LLM-generated plain-English case narrative (Anthropic API call)
4. *(Stretch, time permitting)* Investigation Agent — tool-calling loop
   for free-form reviewer questions ("is this part of a bigger ring?").
   If not built, documented as a Phase 2/roadmap item instead — with the
   explicit guardrail noted that it would be advisory-only, never able to
   execute a BLOCK/ALLOW decision itself.

**Artifacts to produce:** `src/dashboard/app.py`

**Definition of done:** `streamlit run src/dashboard/app.py` works
end-to-end against the trained model + policy engine.

---

## Phase 6 — Docs, Polish, Final Push ⏳ NOT STARTED

**What:** Finalize README, write the roadmap/limitations doc, screenshot
the dashboard, clean commit history, confirm everything runs from a fresh
clone.

**Why:** This is what gets read *before* the code, and it's what makes the
eventual outreach message land as a pitch, not a portfolio dump.

**How to do it:**
1. `docs/02_roadmap_and_limitations.md` — cross-retailer identity
   resolution, adversarial evasion/VPN, drift monitoring, privacy/legal —
   the "how I'd actually solve this at scale" writeup (already drafted in
   conversation, needs to be committed to the repo)
2. Update README status checklist to all-complete
3. Add dashboard screenshots/GIF to README
4. Fresh-clone test: does `pip install -r requirements.txt` +
   the quickstart commands actually work end-to-end for a stranger?
5. Final commit + push

**Artifacts to produce:** `docs/02_roadmap_and_limitations.md`, polished
`README.md`

**Definition of done:** a stranger could clone the repo, follow the
README, and get the dashboard running without asking you a single
question.

---

## Quick Status Snapshot

| Phase | Status |
|---|---|
| 0 — Problem Framing | ✅ Done |
| 1 — Synthetic Data | ✅ Done |
| 2 — Feature Engineering | ✅ Done |
| 3 — Detection Model | ⏭️ Next |
| 4 — Policy Engine | ⏳ Not started |
| 5 — Dashboard | ⏳ Not started |
| 6 — Docs & Polish | ⏳ Not started |
