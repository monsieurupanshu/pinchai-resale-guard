# Resale-Guard

Discount-arbitrage / bulk-resale detection at the point of purchase — extending
PinchAI's post-purchase buyer-intent thesis one step earlier, to checkout.

> Full design doc (problem framing, signal taxonomy, edge cases, eval results,
> and roadmap) lands in Phase 6. This README is a work-in-progress scaffold
> during the build.

## Status
- [x] Phase 1 — Synthetic data generation (6 customer archetypes)
- [ ] Phase 2 — Feature engineering
- [ ] Phase 3 — Detection model (LightGBM)
- [ ] Phase 4 — Policy engine
- [ ] Phase 5 — Dashboard (Streamlit)
- [ ] Phase 6 — Final README + polish

## Quickstart
```bash
pip install -r requirements.txt
python3 src/data_gen/generate.py
```
