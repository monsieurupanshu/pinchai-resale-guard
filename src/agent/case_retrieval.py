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

def build_search_index(corpus: pd.DataFrame):
    """Builds both retrieval indexes over the case corpus:
    - BM25 (sparse, keyword-based)
    - TF-IDF (dense-ish, term-weighted similarity)
    Returns everything needed to search later."""
    tokenized = [doc.lower().split() for doc in corpus["narrative"]]
    bm25 = BM25Okapi(tokenized)

    tfidf = TfidfVectorizer(stop_words="english")
    tfidf_matrix = tfidf.fit_transform(corpus["narrative"])

    return {"bm25": bm25, "tfidf": tfidf, "tfidf_matrix": tfidf_matrix, "corpus": corpus}


def hybrid_search(query: str, index: dict, top_k: int = 5) -> pd.DataFrame:
    """Combines BM25 and TF-IDF rankings using Reciprocal Rank Fusion
    (RRF) — the same fusion technique used in production hybrid
    retrieval systems. Neither ranking alone is used; RRF blends them."""
    corpus = index["corpus"]
    n = len(corpus)

    # BM25 ranking
    bm25_scores = index["bm25"].get_scores(query.lower().split())
    bm25_rank = pd.Series(bm25_scores).rank(ascending=False, method="min")

    # TF-IDF ranking
    query_vec = index["tfidf"].transform([query])
    tfidf_scores = cosine_similarity(query_vec, index["tfidf_matrix"]).flatten()
    tfidf_rank = pd.Series(tfidf_scores).rank(ascending=False, method="min")

    # Reciprocal Rank Fusion: RRF_score = sum(1 / (k + rank)) across rankers
    k = 60  # standard RRF constant
    rrf_score = (1 / (k + bm25_rank)) + (1 / (k + tfidf_rank))

    result = corpus.copy()
    result["rrf_score"] = rrf_score.values
    return result.sort_values("rrf_score", ascending=False).head(top_k)