import os

import pandas as pd
from rank_bm25 import BM25Okapi
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def build_case_corpus() -> pd.DataFrame:
    """Turns every customer's actual policy decision into a searchable
    'case' — a short natural-language narrative of what happened and
    why. This isn't synthetic/fabricated text — it's a plain-English
    rendering of real, already-computed decisions from Phase 4."""
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from src.policy.engine import PolicyEngine

    df = pd.read_csv(os.path.join(DATA_DIR, "features.csv"))
    engine = PolicyEngine()

    cases = []
    for _, row in df.iterrows():
        decision = engine.decide(row)
        narrative = (
            f"Customer {decision['customer_id']}: {decision['action']} "
            f"(risk score {decision['score']}). "
            f"{'. '.join(decision['reasons'])}."
        )
        cases.append({
            "customer_id": decision["customer_id"],
            "action": decision["action"],
            "score": decision["score"],
            "narrative": narrative,
        })
    return pd.DataFrame(cases)