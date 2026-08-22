# Roadmap & Known Limitations

Items deliberately built as design/documentation rather than code, and
why — consistent with the Phase 0 principle of naming scope boundaries
explicitly rather than silently skipping them.

## Investigation Agent (Phase 5 stretch goal — not built)

**What it would be:** instead of a reviewer only seeing a static score +
reason codes, they could ask a free-form question ("why was this
customer flagged, and is this part of a bigger ring?") and an agent would
autonomously call tools to build an answer — pulling the customer's
feature vector, querying the identity-cluster graph, and if a cluster
exists, pulling every linked account's order history to assess combined
volume.

**Why it fits this project specifically:** this mirrors what PinchAI's
own product likely already does at some level (their public claim of
"80% of return reviews handled automatically" implies some automated
investigation/triage layer). Building even a lightweight version would
have shown alignment with where their product is headed, not just where
it is today.

**Why it wasn't built:** a real tool-calling reasoning loop — even a
minimal one — is meaningfully more engineering time than the trust-score
and narrative features that were prioritized instead, given the 7-day
timebox for this project. The trust score and LLM narrative already
demonstrate the same underlying skill (turning model output into
something a human reviewer can act on) with a much smaller build cost.

**The guardrail that would matter most if built:** the agent would be
**advisory only** — it could read data and simulate scenarios, but it
would never be able to execute a BLOCK/ALLOW/FLAG decision itself. Final
action authority stays with the deterministic policy engine (`src/policy/
engine.py`). This is a deliberate safety boundary, not an oversight: in a
fraud/trust domain, letting an LLM-driven agent make the final call
introduces failure modes (hallucinated reasoning, inconsistent
thresholds across runs) that a deterministic rules layer avoids. This is
the same "does the agent make the final call?" question a fraud-focused
interviewer would likely ask, and the honest answer is no.

**What a minimal version would look like, if built next:**
```
Tools available to the agent (read-only):
  get_customer_features(customer_id)
  get_network_cluster(customer_id)
  get_policy_decision(customer_id)
  get_cluster_orders(ring_id)          # only if cluster_size > 1
  simulate_policy(new_threshold)        # for "what if we lowered the
                                         # block threshold" questions

The agent may READ and SIMULATE. It may never WRITE a decision.
```

## Cross-Retailer Identity Resolution (out of scope, Phase 0)

See `docs/00_problem_framing.md` for the original scoping decision. In
short: a real PinchAI deployment serves multiple retailers, and any
cross-retailer identity signal-sharing would need a federated,
privacy-preserving design (hashed/tokenized identifiers, not raw PII) and
a contractual data-sharing framework between retailers — genuine
infrastructure work, not something to simulate against one synthetic
retailer's data.

## Adversarial Evasion / VPN & Device Spoofing (out of scope, Phase 0)

Also detailed in `docs/00_problem_framing.md`. The short version: this
project's synthetic data models static customer archetypes, not an
adaptive adversary. A production system facing real evasion attempts
would need signal diversification (already partially reflected in this
project's feature set — no single feature is load-bearing, per the
Phase 3 eval), drift monitoring (e.g. PSI on key features over time), and
a retraining cadence. Simulating a red-team/blue-team adaptive loop is a
fundamentally larger project than a self-directed take-home.

## Concept Drift / Live Retraining (out of scope)

The model here is trained once on a static synthetic snapshot. A
production system would need scheduled retraining as both genuine
customer behavior and reseller tactics shift over time — not built here,
but a natural extension of the pipeline already in place (`src/model/
train.py` could be re-run on a rolling data window with minimal changes).

## Privacy / Legal Compliance Architecture (out of scope, Phase 0)

Device fingerprinting and cross-account identity linking touch real
consent and data-minimization requirements (GDPR/CCPA-style). A
production system would need consent-scoped identity linking; this
project does not model that layer.
