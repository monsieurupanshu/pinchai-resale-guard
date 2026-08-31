# Architecture

Visual reference for the full system — pipeline, data schema, and agent
tool architecture. GitHub renders the diagrams below natively.

## 1. End-to-End Pipeline

```mermaid
flowchart TD
    A[Synthetic Data Generator] -->|customers.csv, orders.csv| B[Feature Engineering]
    B -->|features.csv, 18 signals| C[Detection Model - LightGBM]
    C -->|reseller_model.txt| D[Policy Engine]
    D -->|score to action + reasons| E[Streamlit Dashboard]
    C -.compared against.-> F[Model Comparison<br/>LogReg / RandForest / LightGBM]
    C --> S[SHAP Analysis<br/>global + per-customer]
    D --> G[Investigation Agent]
    H[Case Retrieval<br/>BM25 + TF-IDF + RRF] --> G
    G --> E

    style A fill:#2a78d6,color:#fff
    style B fill:#2a78d6,color:#fff
    style C fill:#1baf7a,color:#fff
    style D fill:#eb6834,color:#fff
    style E fill:#d03b3b,color:#fff
    style G fill:#8e44ad,color:#fff
    style H fill:#8e44ad,color:#fff
    style S fill:#1baf7a,color:#fff
```

## 2. Data Schema

```mermaid
erDiagram
    CUSTOMERS ||--o{ ORDERS : places
    CUSTOMERS ||--o| LABELS : "has ground truth"
    ORDERS ||--o{ RESALE_LISTINGS : "sometimes relisted"
    CUSTOMERS ||--o| FEATURES : "aggregated into"

    CUSTOMERS {
        string customer_id PK
        datetime account_created
        string device_id
        string payment_fingerprint
        string shipping_address_id
        string home_ip
        string segment "ground truth only"
        bool uses_vpn
    }
    ORDERS {
        string order_id PK
        string customer_id FK
        string sku_id
        int quantity
        float unit_price_paid
        datetime order_timestamp
        string ip_address "rotates if uses_vpn"
    }
    RESALE_LISTINGS {
        string listing_id PK
        string customer_id FK
        float listed_price
    }
    LABELS {
        string customer_id PK
        int is_reseller
        int is_loyal_bulk
    }
    FEATURES {
        string customer_id PK
        float avg_qty_per_order
        float pct_orders_discount_window
        int shared_address_count
        int identity_cluster_size
        int n_distinct_ips
    }
```

**Critical:** `RESALE_LISTINGS` feeds only `LABELS`, never `FEATURES` — a
real checkout-time system has no access to future resale-market data.
Enforced in code: `build_features.py` never reads `resale_listings.csv`.

## 3. Investigation Agent — Tool Architecture

```mermaid
flowchart TD
    Q[Reviewer's question] --> AGENT[Investigation Agent<br/>open-source LLM via Groq<br/>advisory only]

    AGENT -->|reads| T1[get_customer_features]
    AGENT -->|reads| T2[get_network_cluster]
    AGENT -->|reads| T3[get_policy_decision]
    AGENT -->|reads| T4[get_cluster_orders]
    AGENT -->|simulates| T5[simulate_policy]
    AGENT -->|searches| T6[find_similar_cases<br/>BM25 + TF-IDF + RRF]

    T3 -->|sole source of truth for actions| POLICY[Real Policy Engine]
    AGENT --> ANSWER[Synthesized answer]

    style AGENT fill:#8e44ad,color:#fff
    style POLICY fill:#eb6834,color:#fff
    style T6 fill:#1baf7a,color:#fff
```

**Guardrail:** the agent can read, cross-reference, and simulate — it
never executes a real ALLOW/FLAG/LIMIT_QTY/BLOCK decision.
`get_policy_decision` always calls the real deterministic engine; the
agent reports what it says, never invents its own risk assessment. This
was verified in practice — see the system-prompt fix in
`investigation_agent.py` that corrected a case where the agent initially
mischaracterized an ALLOW decision as a "flag."