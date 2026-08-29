import json
import os
import pandas as pd
from groq import Groq
from src.agent.case_retrieval import build_case_corpus, build_search_index, hybrid_search

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
MODEL_NAME = "openai/gpt-oss-20b"

_SEARCH_INDEX = None  # built once, lazily, on first use


def _get_search_index():
    global _SEARCH_INDEX
    if _SEARCH_INDEX is None:
        print("  [building case search index, one-time setup...]")
        corpus = build_case_corpus()
        _SEARCH_INDEX = build_search_index(corpus)
    return _SEARCH_INDEX


def get_customer_features(customer_id: str) -> dict:
    """Tool 1: pull a customer's full signal profile."""
    df = pd.read_csv(os.path.join(DATA_DIR, "features.csv"))
    row = df[df["customer_id"] == customer_id]
    if row.empty:
        return {"error": f"Customer {customer_id} not found"}
    return row.iloc[0].to_dict()

def get_network_cluster(customer_id: str) -> dict:
    """Tool 2: find accounts linked to this customer via shared identity
    (device, payment, address, or IP)."""
    customers = pd.read_csv(os.path.join(DATA_DIR, "customers.csv"))
    target = customers[customers["customer_id"] == customer_id]
    if target.empty:
        return {"error": f"Customer {customer_id} not found"}

    target = target.iloc[0]
    linked = customers[
        (customers["customer_id"] != customer_id) & (
            (customers["device_id"] == target["device_id"]) |
            (customers["payment_fingerprint"] == target["payment_fingerprint"]) |
            (customers["shipping_address_id"] == target["shipping_address_id"]) |
            (customers["home_ip"] == target["home_ip"])
        )
    ]
    return {
        "customer_id": customer_id,
        "linked_account_count": len(linked),
        "linked_account_ids": linked["customer_id"].tolist(),
        "shared_address": bool((linked["shipping_address_id"] == target["shipping_address_id"]).any()),
        "shared_device": bool((linked["device_id"] == target["device_id"]).any()),
        "shared_payment": bool((linked["payment_fingerprint"] == target["payment_fingerprint"]).any()),
        "shared_ip": bool((linked["home_ip"] == target["home_ip"]).any()),
    }
    
def get_policy_decision(customer_id: str) -> dict:
    """Tool 3: get the actual score/action/reasons from the policy engine
    for this customer."""
    import sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
    from src.policy.engine import PolicyEngine

    df = pd.read_csv(os.path.join(DATA_DIR, "features.csv"))
    row = df[df["customer_id"] == customer_id]
    if row.empty:
        return {"error": f"Customer {customer_id} not found"}

    engine = PolicyEngine()
    decision = engine.decide(row.iloc[0])
    return decision

def get_cluster_orders(customer_ids: list) -> dict:
    """Tool 4: pull combined order history across a set of linked accounts
    — shows the TRUE combined volume a ring is moving, which no single
    account's view would reveal."""
    orders = pd.read_csv(os.path.join(DATA_DIR, "orders.csv"))
    cluster_orders = orders[orders["customer_id"].isin(customer_ids)]

    return {
        "accounts_in_cluster": len(customer_ids),
        "total_orders": len(cluster_orders),
        "total_quantity": int(cluster_orders["quantity"].sum()),
        "total_spend": float((cluster_orders["quantity"] * cluster_orders["unit_price_paid"]).sum()),
        "distinct_skus": cluster_orders["sku_id"].nunique(),
    }
    
def simulate_policy(new_threshold: float, action_name: str = "BLOCK") -> dict:
    """Tool 5: 'what if' analysis — if we changed a policy threshold,
    how many test-set customers would move into a given action?
    Read-only simulation, does not change any real policy."""
    import lightgbm as lgb

    MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "models"))
    model = lgb.Booster(model_file=os.path.join(MODEL_DIR, "reseller_model.txt"))
    with open(os.path.join(MODEL_DIR, "feature_cols.txt")) as f:
        feature_cols = f.read().splitlines()

    test_df = pd.read_csv(os.path.join(DATA_DIR, "test_split.csv"))
    scores = model.predict(test_df[feature_cols])
    test_df["score"] = scores

    current_count = (test_df["score"] >= 0.85).sum()  # current BLOCK threshold
    new_count = (test_df["score"] >= new_threshold).sum()

    return {
        "current_threshold": 0.85,
        "current_customers_at_or_above": int(current_count),
        "simulated_threshold": new_threshold,
        "simulated_customers_at_or_above": int(new_count),
        "change": int(new_count - current_count),
    }
    
def find_similar_cases(query: str) -> dict:
    """Tool 6: hybrid search (BM25 + TF-IDF + RRF) over past case
    decisions — answers 'have we seen this pattern before?'"""
    index = _get_search_index()
    results = hybrid_search(query, index, top_k=3)
    return {
        "query": query,
        "similar_cases": results[["customer_id", "action", "score", "narrative"]].to_dict("records"),
    }
    
TOOLS_SCHEMA = [
    {
        "type": "function",
        "function": {
            "name": "get_customer_features",
            "description": "Get a customer's full behavioral signal profile (quantity, timing, SKU concentration, account age, etc.)",
            "parameters": {
                "type": "object",
                "properties": {"customer_id": {"type": "string"}},
                "required": ["customer_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_network_cluster",
            "description": "Find other accounts linked to this customer via shared device, payment, address, or IP",
            "parameters": {
                "type": "object",
                "properties": {"customer_id": {"type": "string"}},
                "required": ["customer_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_policy_decision",
            "description": "Get the actual risk score, action (ALLOW/FLAG/LIMIT_QTY/BLOCK), and reasons for a customer from the real policy engine",
            "parameters": {
                "type": "object",
                "properties": {"customer_id": {"type": "string"}},
                "required": ["customer_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_cluster_orders",
            "description": "Get combined order volume across a list of linked customer IDs — shows true ring-level activity",
            "parameters": {
                "type": "object",
                "properties": {
                    "customer_ids": {"type": "array", "items": {"type": "string"}}
                },
                "required": ["customer_ids"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "simulate_policy",
            "description": "Simulate how many customers would be affected if the BLOCK threshold were changed. Does not change any real policy.",
            "parameters": {
                "type": "object",
                "properties": {"new_threshold": {"type": "number"}},
                "required": ["new_threshold"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_similar_cases",
            "description": "Search past case decisions for similar patterns using hybrid retrieval. Use this to answer 'have we seen this before' questions.",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        },
    },
]

TOOL_FUNCTIONS = {
    "get_customer_features": get_customer_features,
    "get_network_cluster": get_network_cluster,
    "get_policy_decision": get_policy_decision,
    "get_cluster_orders": get_cluster_orders,
    "simulate_policy": simulate_policy,
    "find_similar_cases": find_similar_cases,
}

SYSTEM_PROMPT = """You are an investigation assistant for a fraud-ops reviewer
at an e-commerce retailer. You have READ-ONLY tools to look up customer data,
network connections, and policy decisions. You can also SIMULATE policy
changes. You must NEVER claim to make or change an actual decision — you only
investigate and report facts. The real policy engine (via get_policy_decision)
is the sole source of truth for any action taken. Be concise and factual."""


def ask_agent(question: str, max_turns: int = 5) -> str:
    """The core agentic loop: send the question + tool descriptions to the
    LLM, let it decide which tools to call, run them, feed results back,
    repeat until it has a final answer (or max_turns is reached)."""
    client = Groq(api_key=os.environ["GROQ_API_KEY"])
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]

    for _ in range(max_turns):
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            tools=TOOLS_SCHEMA,
        )
        msg = response.choices[0].message

        if not msg.tool_calls:
            return msg.content

        messages.append(msg)
        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            fn_args = json.loads(tool_call.function.arguments)
            print(f"  [agent called: {fn_name}({fn_args})]")
            result = TOOL_FUNCTIONS[fn_name](**fn_args)
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result),
            })

    return "Reached max turns without a final answer."


if __name__ == "__main__":
    q = input("Ask the investigation agent a question: ")
    answer = ask_agent(q)
    print("\n" + answer)