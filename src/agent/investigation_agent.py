import json
import os

import pandas as pd
from groq import Groq

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
MODEL_NAME = "openai/gpt-oss-20b"


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