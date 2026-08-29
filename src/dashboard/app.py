"""
Resale-Guard Dashboard — the reviewer-facing console.

Run with: streamlit run src/dashboard/app.py

Works fully without an ANTHROPIC_API_KEY (falls back to a template
narrative). If you set one, the case narrative uses the real LLM instead.
"""

import os
import sys

import pandas as pd
import streamlit as st

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.policy.engine import PolicyEngine
from src.dashboard.logic import compute_trust_score, generate_trust_signals, generate_narrative
from src.agent.investigation_agent import ask_agent

st.set_page_config(page_title="Resale-Guard", page_icon="🛡️", layout="wide")

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))

ACTION_COLORS = {
    "ALLOW": "🟢",
    "FLAG": "🟡",
    "LIMIT_QTY": "🟠",
    "BLOCK": "🔴",
}


@st.cache_resource
def load_engine():
    return PolicyEngine()


@st.cache_data
def load_customers():
    return pd.read_csv(os.path.join(DATA_DIR, "test_split.csv"))


def main():
    st.title("🛡️ Resale-Guard")
    st.caption("Checkout-time discount-arbitrage detection — reviewer console")

    engine = load_engine()
    df = load_customers()

    with st.sidebar:
        st.header("Find a customer")
        segment_filter = st.selectbox(
            "Filter by segment (ground truth, for demo browsing only — "
            "a real reviewer wouldn't see this)",
            ["All"] + sorted(df["segment"].unique().tolist()),
        )
        filtered = df if segment_filter == "All" else df[df["segment"] == segment_filter]

        customer_id = st.selectbox("Customer ID", filtered["customer_id"].tolist())

        st.divider()
        api_key = st.text_input(
            "Anthropic API key (optional)",
            type="password",
            help="If provided, case narratives use the real LLM. Otherwise a template is used.",
        )

    row = df[df["customer_id"] == customer_id].iloc[0]
    decision = engine.decide(row)
    trust = compute_trust_score(row, decision["score"])

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Risk Score", f"{decision['score']:.3f}")
    with col2:
        st.metric("Decision", f"{ACTION_COLORS[decision['action']]} {decision['action']}")
    with col3:
        st.metric("Trust Score", f"{trust} / 100")

    st.divider()

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Why this decision")
        for reason in decision["reasons"]:
            prefix = "⚠️" if "OVERRIDE" in reason else "•"
            st.write(f"{prefix} {reason}")

        if decision["action"] == "ALLOW":
            st.subheader("Trust signals")
            for signal in generate_trust_signals(row):
                st.write(f"✓ {signal}")

    with right:
        st.subheader("Case narrative")
        with st.spinner("Generating..."):
            narrative = generate_narrative(row, decision, api_key=api_key or None)
        st.info(narrative)
        if not api_key:
            st.caption("Using template narrative — add an API key in the sidebar for LLM-generated narratives.")

    st.divider()
    st.subheader("Raw signal values")
    feature_cols = [c for c in df.columns if c not in ("customer_id", "segment", "is_reseller", "is_loyal_bulk", "ring_id")]
    display_df = row[feature_cols].to_frame().T
    st.dataframe(display_df, use_container_width=True, hide_index=True)
    
    st.divider()
    st.subheader("🕵️ Ask the Investigation Agent")
    st.caption("Free-form questions — the agent can look up customers, check for linked accounts, "
               "search past cases, and simulate policy changes. Advisory only — it never makes decisions itself.")

    agent_question = st.text_input(
        "Ask a question",
        placeholder=f"e.g. Why was {customer_id} flagged, and is this part of a bigger ring?",
    )
    if st.button("Ask") and agent_question:
        if not os.environ.get("GROQ_API_KEY"):
            st.error("GROQ_API_KEY environment variable not set. The agent needs this to run.")
        else:
            with st.spinner("Investigating..."):
                try:
                    contextualized_question = f"[Currently viewing customer: {customer_id}] {agent_question}"   
                    answer = ask_agent(agent_question)
                    st.success(answer)
                except Exception as e:
                    st.error(f"Agent error: {e}")

    with st.expander("Ground truth (demo only — not available to a real reviewer)"):
        st.write(f"Actual segment: `{row['segment']}`")
        st.write(f"Is reseller (label): `{bool(row['is_reseller'])}`")


if __name__ == "__main__":
    main()
