"""
Feature engineering for Resale-Guard.

Builds one row per customer with the derived signals from the taxonomy:
  1. Purchase behavior  - quantity, concentration, discount-timing
  2. Identity            - account age relative to order volume
  3. Network/graph        - shared address/device/payment across accounts

IMPORTANT: resale_listings.csv is NEVER read in this module. That table
exists purely to generate ground-truth labels (see src/data_gen/generate.py
-> labels.csv). A real checkout-time system would not have resale-market
data available at decision time, so including it here would be label
leakage — the model would be "cheating" with information it can't actually
have in production. This exclusion is deliberate and should be called out
in the README.
"""

import os

import networkx as nx
import numpy as np
import pandas as pd

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "data"))


def _herfindahl(qty_by_sku: pd.Series) -> float:
    """SKU concentration index: sum of squared shares. 1.0 = all purchases
    on a single SKU (max concentration), lower = spread across many SKUs."""
    total = qty_by_sku.sum()
    if total == 0:
        return 0.0
    shares = qty_by_sku / total
    return float((shares ** 2).sum())


def build_purchase_features(orders: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cust_id, g in orders.groupby("customer_id"):
        g = g.sort_values("order_timestamp")
        total_qty = g["quantity"].sum()
        n_orders = len(g)
        qty_by_sku = g.groupby("sku_id")["quantity"].sum()

        span_days = (g["order_timestamp"].max() - g["order_timestamp"].min()).total_seconds() / 86400
        span_days = max(span_days, 0.0)

        discount_orders = g[g["discount_pct_applied"] > 0]

        rows.append({
            "customer_id": cust_id,
            "total_quantity": total_qty,
            "n_orders": n_orders,
            "avg_qty_per_order": total_qty / n_orders,
            "max_qty_single_order": g["quantity"].max(),
            "n_distinct_skus": g["sku_id"].nunique(),
            "sku_concentration": _herfindahl(qty_by_sku),
            "pct_orders_discount_window": len(discount_orders) / n_orders,
            "avg_discount_pct_when_used": (
                discount_orders["discount_pct_applied"].mean() if len(discount_orders) else 0.0
            ),
            "purchase_span_days": span_days,
            "order_frequency_per_week": (n_orders / max(span_days, 1)) * 7,
            "total_spend": (g["quantity"] * g["unit_price_paid"]).sum(),
            "n_distinct_ips": g["ip_address"].nunique(),
        })
    return pd.DataFrame(rows)


def build_identity_features(customers: pd.DataFrame, orders: pd.DataFrame) -> pd.DataFrame:
    first_order = orders.groupby("customer_id")["order_timestamp"].min().rename("first_order_ts")
    merged = customers.merge(first_order, on="customer_id", how="left")

    merged["account_age_at_first_order_days"] = (
        (merged["first_order_ts"] - merged["account_created"]).dt.total_seconds() / 86400
    ).clip(lower=0)

    return merged[["customer_id", "account_age_at_first_order_days"]]


def build_network_features(customers: pd.DataFrame) -> pd.DataFrame:
    """Graph-based identity clustering: connect customers who share a
    device, payment fingerprint, OR shipping address. Connected-component
    size captures coordinated rings even when each individual shared
    attribute alone looks unremarkable."""

    # per-attribute sharing counts (excluding self)
    def _sharing_count(col):
        counts = customers.groupby(col)["customer_id"].transform("count") - 1
        return counts

    customers = customers.copy()
    customers["shared_device_count"] = _sharing_count("device_id")
    customers["shared_payment_count"] = _sharing_count("payment_fingerprint")
    customers["shared_address_count"] = _sharing_count("shipping_address_id")
    customers["shared_ip_count"] = _sharing_count("home_ip")
    # union graph across all three identity attributes
    G = nx.Graph()
    G.add_nodes_from(customers["customer_id"])
    for col in ["device_id", "payment_fingerprint", "shipping_address_id", "home_ip"]:
        for _, g in customers.groupby(col):
            ids = g["customer_id"].tolist()
            if len(ids) > 1:
                first = ids[0]
                for other in ids[1:]:
                    G.add_edge(first, other)

    cluster_size = {}
    for component in nx.connected_components(G):
        size = len(component)
        for node in component:
            cluster_size[node] = size

    customers["identity_cluster_size"] = customers["customer_id"].map(cluster_size).fillna(1).astype(int)

    return customers[[
        "customer_id", "shared_device_count", "shared_payment_count",
        "shared_address_count", "shared_ip_count", "identity_cluster_size",
    ]]


def build_feature_table(data_dir: str = DATA_DIR) -> pd.DataFrame:
    customers = pd.read_csv(os.path.join(data_dir, "customers.csv"), parse_dates=["account_created"])
    orders = pd.read_csv(os.path.join(data_dir, "orders.csv"), parse_dates=["order_timestamp"])
    labels = pd.read_csv(os.path.join(data_dir, "labels.csv"))

    purchase_feats = build_purchase_features(orders)
    identity_feats = build_identity_features(customers, orders)
    network_feats = build_network_features(customers)

    feats = (
        purchase_feats
        .merge(identity_feats, on="customer_id", how="left")
        .merge(network_feats, on="customer_id", how="left")
        .merge(labels[["customer_id", "segment", "is_reseller", "is_loyal_bulk"]], on="customer_id", how="left")
    )
    return feats


if __name__ == "__main__":
    out = build_feature_table()
    out_path = os.path.join(DATA_DIR, "features.csv")
    out.to_csv(out_path, index=False)
    print(f"wrote {out_path}  shape={out.shape}")
    print("\nFeature columns:")
    print([c for c in out.columns if c not in ("customer_id", "segment", "is_reseller", "is_loyal_bulk")])
    print("\nMean feature values by segment (sanity check):")
    numeric_cols = out.select_dtypes(include=[np.number]).columns.tolist()
    print(out.groupby("segment")[numeric_cols].mean().round(2).T)
