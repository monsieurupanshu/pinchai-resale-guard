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