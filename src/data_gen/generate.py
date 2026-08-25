"""
Synthetic data generator for the PinchAI Resale-Guard project.

Simulates a retail e-commerce environment (Adidas-style flash sales) with
four customer archetypes baked in at generation time, so we have honest
ground truth for evaluation:

    1. normal_shopper     (~65%) - everyday customer, 1-3 units, no pattern
    2. loyal_bulk_buyer    (~12%) - buys in volume for LEGITIMATE reasons
                                     (family, small gym/team, reseller-adjacent
                                     but NOT flagged) — the "don't punish good
                                     customers" case PinchAI cares about
    3. solo_reseller        (~11%) - one account, concentrated bulk buys
                                     timed to discount windows, resells later
    4. ring_reseller        (~12%) - 2-6 coordinated accounts sharing
                                     device/payment/address to evade
                                     per-account quantity limits

Resale listings are generated ONLY for solo/ring resellers and used purely
as ground truth for labels + evaluation — NOT as a live feature, since a
real detection system at checkout time would not have access to future
resale-market data. This avoids label leakage in the feature set.
"""

import json
import random
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

RNG_SEED = 42
random.seed(RNG_SEED)
np.random.seed(RNG_SEED)

# ---------------------------------------------------------------------------
# Reference data: SKUs and discount calendar
# ---------------------------------------------------------------------------

SKU_CATALOG = [
    {"sku_id": "ADI-SAMBA-BLK-42", "name": "Adidas Samba OG Black", "category": "sneakers", "retail_price": 90.0},
    {"sku_id": "ADI-SAMBA-WHT-42", "name": "Adidas Samba OG White", "category": "sneakers", "retail_price": 90.0},
    {"sku_id": "ADI-GAZL-BLU-42",  "name": "Adidas Gazelle Blue",   "category": "sneakers", "retail_price": 100.0},
    {"sku_id": "ADI-SUPER-BLK-42", "name": "Adidas Superstar Black","category": "sneakers", "retail_price": 85.0},
    {"sku_id": "ADI-TRACK-BLK-M",  "name": "Adidas Track Jacket",   "category": "apparel",  "retail_price": 70.0},
    {"sku_id": "ADI-TRACK-NVY-M",  "name": "Adidas Track Pant Navy","category": "apparel",  "retail_price": 60.0},
    {"sku_id": "ADI-TEE-WHT-M",    "name": "Adidas Trefoil Tee",    "category": "apparel",  "retail_price": 30.0},
    {"sku_id": "ADI-CAP-BLK",      "name": "Adidas Baseball Cap",   "category": "accessory","retail_price": 25.0},
]

SIM_START = datetime(2026, 1, 1)
SIM_END = datetime(2026, 6, 30)
SIM_DAYS = (SIM_END - SIM_START).days

# Flash-sale / discount windows: (start_offset_days, duration_hours, pct_off, skus)
DISCOUNT_EVENTS = [
    {"event_id": "EVT-JAN-FLASH", "start_offset_days": 10, "duration_hours": 6,  "discount_pct": 0.40,
     "skus": ["ADI-SAMBA-BLK-42", "ADI-SAMBA-WHT-42", "ADI-GAZL-BLU-42"]},
    {"event_id": "EVT-FEB-VDAY",  "start_offset_days": 40, "duration_hours": 12, "discount_pct": 0.30,
     "skus": ["ADI-TRACK-BLK-M", "ADI-TEE-WHT-M"]},
    {"event_id": "EVT-MAR-CLEAR", "start_offset_days": 70, "duration_hours": 8,  "discount_pct": 0.50,
     "skus": ["ADI-SUPER-BLK-42", "ADI-SAMBA-BLK-42"]},
    {"event_id": "EVT-APR-DROP",  "start_offset_days": 100,"duration_hours": 4,  "discount_pct": 0.45,
     "skus": ["ADI-GAZL-BLU-42", "ADI-SAMBA-WHT-42"]},
    {"event_id": "EVT-MAY-FLASH", "start_offset_days": 130,"duration_hours": 6,  "discount_pct": 0.35,
     "skus": ["ADI-TRACK-NVY-M", "ADI-CAP-BLK"]},
    {"event_id": "EVT-JUN-CLEAR", "start_offset_days": 160,"duration_hours": 10, "discount_pct": 0.40,
     "skus": ["ADI-SAMBA-BLK-42", "ADI-SUPER-BLK-42", "ADI-GAZL-BLU-42"]},
]


def _rand_ts(start: datetime, end: datetime) -> datetime:
    delta = end - start
    seconds = random.uniform(0, delta.total_seconds())
    return start + timedelta(seconds=seconds)


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:10]}"

def _new_ip():
    return f"{random.randint(1,223)}.{random.randint(0,255)}.{random.randint(0,255)}.{random.randint(1,254)}"


# ---------------------------------------------------------------------------
# Customer archetype generators
# ---------------------------------------------------------------------------

@dataclass
class Customer:
    customer_id: str
    account_created: datetime
    device_id: str
    payment_fingerprint: str
    shipping_address_id: str
    segment: str  # normal | loyal_bulk | solo_reseller | ring_reseller
    ring_id: str = ""  # non-empty only for ring_reseller members
    ring_style: str = ""  # "aggressive" or "stealth" — only for ring_reseller
    home_ip: str = ""
    uses_vpn: bool = False

def gen_normal_shoppers(n: int) -> list:
    customers = []
    for _ in range(n):
        created = _rand_ts(SIM_START - timedelta(days=400), SIM_END - timedelta(days=1))
        customers.append(Customer(
            customer_id=_new_id("CUST"),
            account_created=created,
            device_id=_new_id("DEV"),
            payment_fingerprint=_new_id("PAY"),
            shipping_address_id=_new_id("ADDR"),
            segment="normal",
            home_ip=_new_ip(),
        )) 
    return customers


def gen_loyal_bulk_buyers(n: int) -> list:
    """Legitimate bulk buyers: established accounts, buy in volume across
    sale AND non-sale periods, diverse SKUs, no shared identity signals.
    This is the population that must NOT be flagged.

    85% have long-established accounts (the typical case). 15% are
    genuinely new fans who happen to buy in bulk right away (e.g. someone
    outfitting a whole family after discovering the brand) — this overlap
    with reseller account-age ranges is intentional, so account age alone
    can't be a shortcut feature."""
    customers = []
    for _ in range(n):
        if random.random() < 0.85:
            created = _rand_ts(SIM_START - timedelta(days=700), SIM_START - timedelta(days=200))
        else:
            created = _rand_ts(SIM_START - timedelta(days=90), SIM_START + timedelta(days=100))
        customers.append(Customer(
            customer_id=_new_id("CUST"),
            account_created=created,
            device_id=_new_id("DEV"),
            payment_fingerprint=_new_id("PAY"),
            shipping_address_id=_new_id("ADDR"),
            segment="loyal_bulk",
            home_ip=_new_ip(),
        ))
    return customers


def gen_solo_resellers(n: int) -> list:
    """Single account, concentrated bulk buys timed to discount windows.

    75% use newer accounts (the typical, less-careful case). 25% are
    "patient" resellers who let an account age before striking, or reuse
    an older account specifically to look less suspicious — a realistic
    evasion tactic that also means account age alone can't fully separate
    resellers from `loyal_bulk`."""
    customers = []
    for _ in range(n):
        if random.random() < 0.75:
            created = _rand_ts(SIM_START - timedelta(days=60), SIM_START + timedelta(days=150))
        else:
            created = _rand_ts(SIM_START - timedelta(days=650), SIM_START - timedelta(days=180))
        customers.append(Customer(
            customer_id=_new_id("CUST"),
            account_created=created,
            device_id=_new_id("DEV"),
            payment_fingerprint=_new_id("PAY"),
            shipping_address_id=_new_id("ADDR"),
            segment="solo_reseller",
        ))
    return customers


def gen_shared_address_legit(n_households: int, people_per_household=(2, 4)) -> list:
    """False-positive stress test: unrelated-but-cohabiting legitimate
    customers (roommates, family, apartment complex) who share ONE
    shipping address but have separate devices, payment methods, and
    completely independent normal-shopper purchase behavior. Must NOT
    be flagged just because the address-sharing signal fires."""
    customers = []
    for _ in range(n_households):
        shared_address = _new_id("ADDR")
        k = random.randint(*people_per_household)
        for _ in range(k):
            created = _rand_ts(SIM_START - timedelta(days=500), SIM_END - timedelta(days=1))
            customers.append(Customer(
                customer_id=_new_id("CUST"),
                account_created=created,
                device_id=_new_id("DEV"),          # NOT shared
                payment_fingerprint=_new_id("PAY"), # NOT shared
                shipping_address_id=shared_address,  # shared — the confound
                segment="shared_address_legit",
            ))
    return customers


def gen_stealth_resellers(n: int) -> list:
    """False-negative stress test: resellers who deliberately spread
    purchases across non-sale days too (not just discount windows), to
    evade a model that over-relies on discount-timing as its primary
    signal. Still concentrated on few SKUs and still bulk quantities —
    the tell has to come from quantity/SKU-concentration, not timing.

    70% newer accounts, 30% aged accounts (patient-reseller variant) —
    prevents account age from being a clean shortcut."""
    customers = []
    for _ in range(n):
        if random.random() < 0.70:
            created = _rand_ts(SIM_START - timedelta(days=60), SIM_START + timedelta(days=150))
        else:
            created = _rand_ts(SIM_START - timedelta(days=650), SIM_START - timedelta(days=180))
        customers.append(Customer(
            customer_id=_new_id("CUST"),
            account_created=created,
            device_id=_new_id("DEV"),
            payment_fingerprint=_new_id("PAY"),
            shipping_address_id=_new_id("ADDR"),
            segment="stealth_reseller",
        ))
    return customers


def gen_ring_resellers(n_rings: int, accounts_per_ring=(2, 6)) -> list:
    """Coordinated rings: multiple accounts sharing device and/or payment
    and/or shipping address, created in tight bursts to evade per-account
    quantity limits.

    70% are "aggressive" rings — obvious bulk buying timed to discount
    windows, same as before (catchable via quantity/timing signals alone).

    30% are "stealth" rings — deliberately mimic normal shopper behavior
    in every purchase signal (modest quantities, spread across time, not
    discount-timed). The ONLY tell is the shared shipping address across
    accounts. This variant exists specifically to prove the network/graph
    feature earns its place — without it, stealth rings are
    indistinguishable from `normal` on every other signal."""
    customers = []
    for _ in range(n_rings):
        ring_id = _new_id("RING")
        k = random.randint(*accounts_per_ring)
        shared_device = _new_id("DEV")
        shared_payment = _new_id("PAY")
        shared_address = _new_id("ADDR")
        sharing_mode = random.choice(["device_and_address", "payment_and_address", "address_only"])
        ring_style = "aggressive" if random.random() < 0.70 else "stealth"

        if ring_style == "aggressive":
            burst_start = _rand_ts(SIM_START, SIM_START + timedelta(days=170))
            account_ts_fn = lambda: burst_start + timedelta(minutes=random.uniform(0, 240))
        else:
            # stealth rings don't create accounts in a suspicious tight
            # burst either — spread account creation out like normal
            # shoppers would
            account_ts_fn = lambda: _rand_ts(SIM_START - timedelta(days=300), SIM_END - timedelta(days=30))

        for _ in range(k):
            created = account_ts_fn()
            device = shared_device if sharing_mode in ("device_and_address",) else _new_id("DEV")
            payment = shared_payment if sharing_mode in ("payment_and_address",) else _new_id("PAY")
            customers.append(Customer(
                customer_id=_new_id("CUST"),
                account_created=created,
                device_id=device,
                payment_fingerprint=payment,
                shipping_address_id=shared_address,  # always shared — that's the tell
                segment="ring_reseller",
                ring_id=ring_id,
                ring_style=ring_style,
            ))
    return customers


# ---------------------------------------------------------------------------
# Order generation per segment
# ---------------------------------------------------------------------------

def _discount_windows():
    windows = []
    for evt in DISCOUNT_EVENTS:
        start = SIM_START + timedelta(days=evt["start_offset_days"])
        end = start + timedelta(hours=evt["duration_hours"])
        windows.append({**evt, "start": start, "end": end})
    return windows


def gen_orders_normal(customers, windows) -> list:
    orders = []
    for c in customers:
        n_orders = np.random.poisson(1.2)
        for _ in range(n_orders):
            sku = random.choice(SKU_CATALOG)
            qty = np.random.choice([1, 1, 1, 2], p=[0.7, 0.15, 0.1, 0.05]) if False else int(np.random.choice([1, 2], p=[0.85, 0.15]))
            ts = _rand_ts(max(SIM_START, c.account_created), SIM_END)
            price = sku["retail_price"]
            orders.append(_make_order(c, sku, qty, ts, price, discount_pct=0.0))
    return orders


def gen_orders_loyal_bulk(customers, windows) -> list:
    orders = []
    for c in customers:
        n_orders = np.random.poisson(4)  # frequent buyers
        # 80% diverse-SKU shoppers (the typical case). 20% are "brand
        # loyalists" who favor 1-2 styles and buy repeatedly for family/
        # team — this overlaps with reseller SKU-concentration patterns
        # on purpose, so concentration alone can't be a shortcut feature.
        is_loyalist = random.random() < 0.20
        favorite_skus = random.sample(SKU_CATALOG, k=random.choice([1, 2])) if is_loyalist else None
        for _ in range(max(1, n_orders)):
            sku = random.choice(favorite_skus) if is_loyalist else random.choice(SKU_CATALOG)
            qty = int(np.random.choice([3, 4, 5, 6, 8], p=[0.35, 0.25, 0.2, 0.12, 0.08]))
            ts = _rand_ts(max(SIM_START, c.account_created), SIM_END)
            discount_pct = 0.0
            price = sku["retail_price"]
            for w in windows:
                if w["start"] <= ts <= w["end"] and sku["sku_id"] in w["skus"]:
                    discount_pct = w["discount_pct"]
                    price = sku["retail_price"] * (1 - discount_pct)
            orders.append(_make_order(c, sku, qty, ts, price, discount_pct))
    return orders


def gen_orders_solo_reseller(customers, windows) -> list:
    orders = []
    for c in customers:
        # 75% concentrate on 1-2 SKUs (typical). 25% hedge across 3-4
        # SKUs (diversifying inventory risk) — overlaps with loyal_bulk's
        # SKU-diversity range on purpose.
        if random.random() < 0.75:
            target_skus = random.sample(SKU_CATALOG, k=random.choice([1, 2]))
        else:
            target_skus = random.sample(SKU_CATALOG, k=random.choice([3, 4]))
        n_hits = random.randint(2, 5)
        chosen_windows = random.sample(windows, k=min(n_hits, len(windows)))
        for w in chosen_windows:
            eligible = [s for s in target_skus if s["sku_id"] in w["skus"]]
            if not eligible:
                continue
            sku = random.choice(eligible)
            qty = int(np.random.choice([8, 10, 12, 15, 20], p=[0.3, 0.25, 0.2, 0.15, 0.1]))
            ts = w["start"] + timedelta(minutes=random.uniform(0, 30))  # rushes the window
            price = sku["retail_price"] * (1 - w["discount_pct"])
            orders.append(_make_order(c, sku, qty, ts, price, w["discount_pct"]))
    return orders


def gen_orders_shared_address_legit(customers, windows) -> list:
    # behaves exactly like normal shoppers — the ONLY anomaly is the
    # shared address, nothing else
    return gen_orders_normal(customers, windows)


def gen_orders_stealth_reseller(customers, windows) -> list:
    orders = []
    for c in customers:
        if random.random() < 0.75:
            target_skus = random.sample(SKU_CATALOG, k=random.choice([1, 2]))
        else:
            target_skus = random.sample(SKU_CATALOG, k=random.choice([3, 4]))
        n_orders = random.randint(6, 12)  # many smaller orders instead of few huge ones
        for _ in range(n_orders):
            sku = random.choice(target_skus)
            qty = int(np.random.choice([3, 4, 5], p=[0.5, 0.3, 0.2]))  # stays modest per order
            # deliberately spread across the ENTIRE timeline, sale or not —
            # this is what defeats a timing-only detector
            ts = _rand_ts(max(SIM_START, c.account_created), SIM_END)
            discount_pct = 0.0
            price = sku["retail_price"]
            for w in windows:
                if w["start"] <= ts <= w["end"] and sku["sku_id"] in w["skus"]:
                    discount_pct = w["discount_pct"]
                    price = sku["retail_price"] * (1 - discount_pct)
            orders.append(_make_order(c, sku, qty, ts, price, discount_pct))
    return orders


def gen_orders_ring_reseller(customers, windows) -> list:
    orders = []
    by_ring = {}
    for c in customers:
        by_ring.setdefault(c.ring_id, []).append(c)

    for ring_id, members in by_ring.items():
        style = members[0].ring_style

        if style == "aggressive":
            target_skus = random.sample(SKU_CATALOG, k=random.choice([1, 2]))
            n_hits = random.randint(2, 4)
            chosen_windows = random.sample(windows, k=min(n_hits, len(windows)))
            for w in chosen_windows:
                eligible = [s for s in target_skus if s["sku_id"] in w["skus"]]
                if not eligible:
                    continue
                sku = random.choice(eligible)
                for c in members:
                    qty = int(np.random.choice([3, 4, 5], p=[0.5, 0.3, 0.2]))
                    ts = w["start"] + timedelta(minutes=random.uniform(0, 20))
                    price = sku["retail_price"] * (1 - w["discount_pct"])
                    orders.append(_make_order(c, sku, qty, ts, price, w["discount_pct"]))
        else:
            # stealth ring: each member shops like a `normal` customer —
            # modest quantity, random SKU, no discount-window timing, no
            # coordinated burst. The shared address is the ONLY signal
            # that ties them together as a ring.
            for c in members:
                n_orders = np.random.poisson(1.3)
                for _ in range(max(1, n_orders)):
                    sku = random.choice(SKU_CATALOG)
                    qty = int(np.random.choice([1, 2], p=[0.85, 0.15]))
                    ts = _rand_ts(max(SIM_START, c.account_created), SIM_END)
                    discount_pct = 0.0
                    price = sku["retail_price"]
                    for w in windows:
                        if w["start"] <= ts <= w["end"] and sku["sku_id"] in w["skus"]:
                            discount_pct = w["discount_pct"]
                            price = sku["retail_price"] * (1 - discount_pct)
                    orders.append(_make_order(c, sku, qty, ts, price, discount_pct))
    return orders


def _make_order(customer: Customer, sku: dict, qty: int, ts: datetime, unit_price: float, discount_pct: float) -> dict:
    return {
        "order_id": _new_id("ORD"),
        "customer_id": customer.customer_id,
        "sku_id": sku["sku_id"],
        "category": sku["category"],
        "quantity": qty,
        "unit_price_paid": round(unit_price, 2),
        "order_timestamp": ts,
        "discount_pct_applied": discount_pct,
        "device_id": customer.device_id,
        "payment_fingerprint": customer.payment_fingerprint,
        "shipping_address_id": customer.shipping_address_id,
    }


# ---------------------------------------------------------------------------
# Resale listings (ground-truth-only signal, never used as a live feature)
# ---------------------------------------------------------------------------

def gen_resale_listings(orders_df: pd.DataFrame, customers_by_id: dict) -> pd.DataFrame:
    listings = []
    reseller_orders = orders_df[orders_df["customer_id"].map(
        lambda cid: customers_by_id[cid].segment in ("solo_reseller", "ring_reseller", "stealth_reseller")
    )]
    for _, row in reseller_orders.iterrows():
        # not every unit gets relisted (some kept/gifted), and relisting
        # happens with a delay after purchase
        if random.random() < 0.75:
            markup = random.uniform(1.3, 2.2)
            listed_price = round(row["unit_price_paid"] * markup, 2)
            list_delay_days = random.randint(1, 21)
            listings.append({
                "listing_id": _new_id("LST"),
                "matched_order_id": row["order_id"],
                "customer_id": row["customer_id"],
                "sku_id": row["sku_id"],
                "listed_price": listed_price,
                "listed_timestamp": row["order_timestamp"] + timedelta(days=list_delay_days),
                "quantity_listed": max(1, int(row["quantity"] * random.uniform(0.6, 1.0))),
            })
    return pd.DataFrame(listings)


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def generate_all(
    n_normal=1400,
    n_loyal_bulk=250,
    n_shared_address_legit_households=70,
    n_solo_reseller=230,
    n_stealth_reseller=90,
    n_rings=55,
):
    windows = _discount_windows()

    normal = gen_normal_shoppers(n_normal)
    loyal = gen_loyal_bulk_buyers(n_loyal_bulk)
    shared_legit = gen_shared_address_legit(n_shared_address_legit_households)
    solo = gen_solo_resellers(n_solo_reseller)
    stealth = gen_stealth_resellers(n_stealth_reseller)
    ring = gen_ring_resellers(n_rings)

    all_customers = normal + loyal + shared_legit + solo + stealth + ring
    customers_by_id = {c.customer_id: c for c in all_customers}

    orders = (
        gen_orders_normal(normal, windows)
        + gen_orders_loyal_bulk(loyal, windows)
        + gen_orders_shared_address_legit(shared_legit, windows)
        + gen_orders_solo_reseller(solo, windows)
        + gen_orders_stealth_reseller(stealth, windows)
        + gen_orders_ring_reseller(ring, windows)
    )

    orders_df = pd.DataFrame(orders).sort_values("order_timestamp").reset_index(drop=True)
    customers_df = pd.DataFrame([asdict(c) for c in all_customers])
    resale_df = gen_resale_listings(orders_df, customers_by_id)
    sku_df = pd.DataFrame(SKU_CATALOG)
    discount_df = pd.DataFrame(_discount_windows())[
        ["event_id", "start", "end", "discount_pct", "skus"]
    ]

    # ground-truth label table — used ONLY for training labels + eval,
    # never fed to the model as a feature
    labels_df = customers_df[["customer_id", "segment", "ring_id"]].copy()
    labels_df["is_reseller"] = labels_df["segment"].isin(
        ["solo_reseller", "ring_reseller", "stealth_reseller"]
    ).astype(int)
    labels_df["is_loyal_bulk"] = labels_df["segment"].isin(
        ["loyal_bulk", "shared_address_legit"]
    ).astype(int)

    return {
        "customers": customers_df,
        "orders": orders_df,
        "resale_listings": resale_df,
        "skus": sku_df,
        "discount_events": discount_df,
        "labels": labels_df,
    }


if __name__ == "__main__":
    import os
    out_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    out_dir = os.path.abspath(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    tables = generate_all()
    for name, df in tables.items():
        path = os.path.join(out_dir, f"{name}.csv")
        df.to_csv(path, index=False)
        print(f"wrote {path}  shape={df.shape}")

    print("\nSegment distribution:")
    print(tables["labels"]["segment"].value_counts())
