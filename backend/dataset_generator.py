"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  PS2 — Federated Learning for Cross-Institution Financial Risk Control     ║
║  Dataset Generator: Synthetic Heterogeneous Financial Data                 ║
║  Team JabsonX: Saksham, Stanley, Tanushri, Alwyn, Augustine               ║
╚══════════════════════════════════════════════════════════════════════════════╝

Generates 1,000 synthetic customer records across three institution types with
deliberately mismatched schemas, overlapping customer pools (linked via hashed
PAN/Aadhaar), and realistic feature distributions.

Institutions:
  1. Bank Node      — RBI-regulated, long-term structured data
  2. Lending App    — Alt-data, high-frequency, thin-file friendly
  3. Insurer Node   — IRDAI-regulated, actuarial long-horizon data

Usage:
    python dataset_generator.py [--seed 42] [--n_customers 1000] [--output_dir ./data]

Output:
    data/bank_node.csv
    data/lending_app_node.csv
    data/insurer_node.csv
    data/hashed_id_registry.csv   (ground truth for entity resolution)
"""

import os
import argparse
import hashlib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple

# ════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ════════════════════════════════════════════════════════════════════════════

# Salt for SHA-256 hashing (in production, this would be agreed-upon by all nodes
# and stored securely, never hardcoded)
HASH_SALT = "JabsonX_FedRisk_2026_DPDP_Compliant"

# Overlap ratios: fraction of the 1000 customers each institution holds
# Some customers appear in multiple institutions (realistic cross-selling)
BANK_COVERAGE = 0.75          # 750 customers
LENDING_APP_COVERAGE = 0.60   # 600 customers
INSURER_COVERAGE = 0.55       # 550 customers

# Risk profile distribution (low / medium / high risk)
RISK_PROFILES = {
    "low":    0.50,  # 50% of customers are low risk
    "medium": 0.35,  # 35% medium risk
    "high":   0.15,  # 15% high risk
}


def generate_hashed_id(raw_id: str) -> str:
    """
    Entity Resolution: SHA-256(PAN/Aadhaar + salt)
    
    This is the ONLY identifier that crosses institutional boundaries.
    Raw PAN/Aadhaar NEVER leaves the generating institution.
    
    Args:
        raw_id: Simulated PAN or Aadhaar number (e.g., 'CUST_00042')
    
    Returns:
        Hex digest of SHA-256 hash
    """
    payload = f"{raw_id}:{HASH_SALT}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def assign_risk_profile(n: int, rng: np.random.Generator) -> np.ndarray:
    """
    Assign each customer a latent risk profile that governs feature generation.
    This creates correlated features WITHIN an institution (realistic).
    
    Returns:
        Array of risk scores in [0, 1] drawn from a mixture distribution
    """
    profiles = rng.choice(
        ["low", "medium", "high"],
        size=n,
        p=[RISK_PROFILES["low"], RISK_PROFILES["medium"], RISK_PROFILES["high"]]
    )
    
    # Map profiles to base risk scores (latent variable)
    base_risk = np.zeros(n)
    for i, p in enumerate(profiles):
        if p == "low":
            base_risk[i] = rng.beta(2, 8)       # skewed toward 0
        elif p == "medium":
            base_risk[i] = rng.beta(5, 5)       # centered around 0.5
        else:  # high
            base_risk[i] = rng.beta(8, 2)       # skewed toward 1
    
    return base_risk, profiles


# ════════════════════════════════════════════════════════════════════════════
# BANK NODE DATASET (RBI-Regulated, Structured/Formal)
# ════════════════════════════════════════════════════════════════════════════

def generate_bank_data(
    customer_ids: List[str],
    hashed_ids: List[str],
    base_risk: np.ndarray,
    rng: np.random.Generator
) -> pd.DataFrame:
    """
    Bank Node Features (RBI-regulated, long-term structured data):
    
    ┌─────────────────────────────────┬────────┬──────────────────────────────┐
    │ Feature                         │ Weight │ Description                  │
    ├─────────────────────────────────┼────────┼──────────────────────────────┤
    │ repayment_on_time_ratio         │ 30%    │ Fraction of EMIs paid on time│
    │ debt_to_income_ratio            │ 20%    │ Total debt / monthly income  │
    │ balance_trend_3m                │  —     │ 3-month avg balance delta    │
    │ cheque_bounces_12m              │  —     │ Bounced cheques in 12 months │
    │ overdraft_usage_ratio           │  —     │ Overdraft used / limit       │
    │ account_vintage_months          │  —     │ How long account has existed │
    │ txn_velocity_anomaly_score      │  —     │ Deviation from normal txn    │
    │                                 │        │ frequency (z-score based)    │
    └─────────────────────────────────┴────────┴──────────────────────────────┘
    """
    n = len(customer_ids)
    
    # ── Repayment On-Time Ratio (30% weight) ──
    # Low-risk customers pay on time ~95%, high-risk ~50%
    repayment_ratio = np.clip(
        1.0 - base_risk * 0.5 + rng.normal(0, 0.05, n), 0.0, 1.0
    )
    
    # ── Debt-to-Income Ratio (20% weight) ──
    # Low-risk: ~0.2, High-risk: ~0.8
    dti_ratio = np.clip(
        base_risk * 0.7 + 0.1 + rng.normal(0, 0.08, n), 0.0, 1.5
    )
    
    # ── Balance Trend (3-month average delta in INR) ──
    # Positive = growing balance (good), negative = declining
    balance_trend = (
        (1.0 - base_risk) * 15000 - 5000 + rng.normal(0, 3000, n)
    )
    
    # ── Cheque Bounces in last 12 months ──
    # Low-risk: ~0, High-risk: up to 8
    cheque_bounces = np.clip(
        rng.poisson(lam=base_risk * 4, size=n), 0, 15
    ).astype(int)
    
    # ── Overdraft Usage Ratio ──
    # How much of overdraft limit is typically used (0-1)
    overdraft_usage = np.clip(
        base_risk * 0.6 + rng.normal(0, 0.1, n), 0.0, 1.0
    )
    
    # ── Account Vintage (months) ──
    # Older accounts tend to be lower risk (survivorship)
    account_vintage = np.clip(
        (1.0 - base_risk) * 120 + 12 + rng.normal(0, 20, n), 1, 360
    ).astype(int)
    
    # ── Transaction Velocity Anomaly Score ──
    # z-score of recent transaction frequency vs. historical
    # High-risk customers have more anomalies
    txn_anomaly = np.clip(
        base_risk * 3.0 - 0.5 + rng.normal(0, 0.5, n), -2.0, 5.0
    )
    
    # ── Monthly Income (INR) — auxiliary feature ──
    monthly_income = np.clip(
        (1.0 - base_risk) * 80000 + 20000 + rng.normal(0, 15000, n),
        8000, 500000
    ).astype(int)
    
    # ── Credit Score (CIBIL-like, 300-900) ──
    credit_score = np.clip(
        (1.0 - base_risk) * 500 + 350 + rng.normal(0, 30, n),
        300, 900
    ).astype(int)
    
    df = pd.DataFrame({
        "hashed_customer_id":       hashed_ids,
        "repayment_on_time_ratio":  np.round(repayment_ratio, 4),
        "debt_to_income_ratio":     np.round(dti_ratio, 4),
        "balance_trend_3m_inr":     np.round(balance_trend, 2),
        "cheque_bounces_12m":       cheque_bounces,
        "overdraft_usage_ratio":    np.round(overdraft_usage, 4),
        "account_vintage_months":   account_vintage,
        "txn_velocity_anomaly":     np.round(txn_anomaly, 4),
        "monthly_income_inr":       monthly_income,
        "credit_score_cibil":       credit_score,
    })
    
    return df


# ════════════════════════════════════════════════════════════════════════════
# LENDING APP NODE DATASET (Alt-Data / High-Frequency)
# ════════════════════════════════════════════════════════════════════════════

def generate_lending_app_data(
    customer_ids: List[str],
    hashed_ids: List[str],
    base_risk: np.ndarray,
    rng: np.random.Generator
) -> pd.DataFrame:
    """
    Lending App Node Features (thin-file friendly, alternative data):
    
    ┌─────────────────────────────────┬────────┬──────────────────────────────┐
    │ Feature                         │ Weight │ Description                  │
    ├─────────────────────────────────┼────────┼──────────────────────────────┤
    │ active_loans_across_apps        │ 25%    │ Concurrent micro-loans       │
    │ micro_repayment_consistency     │  —     │ On-time rate for micro-loans │
    │ app_engagement_score            │  —     │ Daily active usage (0-1)     │
    │ utility_bill_regularity         │  —     │ Bill payment consistency     │
    │ geolocation_consistency         │  —     │ Stability of home/work GPS   │
    │ days_since_last_request         │  —     │ Time since last loan request │
    └─────────────────────────────────┴────────┴──────────────────────────────┘
    """
    n = len(customer_ids)
    
    # ── Active Loans Across Apps (25% weight) ──
    # High-risk: many concurrent loans (loan stacking)
    active_loans = np.clip(
        rng.poisson(lam=base_risk * 5 + 0.5, size=n), 0, 12
    ).astype(int)
    
    # ── Micro-Loan Repayment Cycle Consistency ──
    # Fraction of micro-loan cycles completed on time (0-1)
    micro_repayment = np.clip(
        1.0 - base_risk * 0.6 + rng.normal(0, 0.08, n), 0.0, 1.0
    )
    
    # ── App Engagement Score ──
    # Higher engagement → slightly lower risk (engaged users are responsive)
    app_engagement = np.clip(
        (1.0 - base_risk) * 0.5 + 0.3 + rng.normal(0, 0.1, n), 0.0, 1.0
    )
    
    # ── Utility Bill Payment Regularity ──
    # Fraction of utility bills paid within 7 days of due date
    utility_regularity = np.clip(
        1.0 - base_risk * 0.4 + rng.normal(0, 0.1, n), 0.0, 1.0
    )
    
    # ── Geolocation Consistency ──
    # Stability score: low-risk customers have stable home/work patterns
    geo_consistency = np.clip(
        (1.0 - base_risk) * 0.7 + 0.2 + rng.normal(0, 0.08, n), 0.0, 1.0
    )
    
    # ── Days Since Last Loan Request ──
    # Frequent requests (low days) → higher risk
    days_since_request = np.clip(
        (1.0 - base_risk) * 180 + 5 + rng.exponential(20, n), 1, 365
    ).astype(int)
    
    # ── Phone Model Tier (1-5) — auxiliary alt-data ──
    phone_tier = np.clip(
        rng.poisson(lam=(1.0 - base_risk) * 3 + 1, size=n), 1, 5
    ).astype(int)
    
    # ── Social References Verified (count) ──
    social_refs = np.clip(
        rng.poisson(lam=(1.0 - base_risk) * 4 + 0.5, size=n), 0, 10
    ).astype(int)
    
    df = pd.DataFrame({
        "hashed_customer_id":        hashed_ids,
        "active_loans_across_apps":  active_loans,
        "micro_repayment_consistency": np.round(micro_repayment, 4),
        "app_engagement_score":      np.round(app_engagement, 4),
        "utility_bill_regularity":   np.round(utility_regularity, 4),
        "geolocation_consistency":   np.round(geo_consistency, 4),
        "days_since_last_request":   days_since_request,
        "phone_model_tier":          phone_tier,
        "social_refs_verified":      social_refs,
    })
    
    return df


# ════════════════════════════════════════════════════════════════════════════
# INSURER NODE DATASET (Actuarial / Long-Horizon)
# ════════════════════════════════════════════════════════════════════════════

def generate_insurer_data(
    customer_ids: List[str],
    hashed_ids: List[str],
    base_risk: np.ndarray,
    rng: np.random.Generator
) -> pd.DataFrame:
    """
    Insurer Node Features (IRDAI-regulated, actuarial long-horizon data):
    
    ┌─────────────────────────────────┬────────┬──────────────────────────────┐
    │ Feature                         │ Weight │ Description                  │
    ├─────────────────────────────────┼────────┼──────────────────────────────┤
    │ claims_frequency_24m            │ 30%    │ Claims filed in 24 months    │
    │ claim_rejection_fraud_flag      │  —     │ Binary: any rejected/flagged │
    │ premium_lapse_count             │  —     │ Number of lapsed premiums    │
    │ policy_renewal_consistency      │  —     │ Renewal rate across policies │
    │ active_policies_held            │  —     │ Currently active policies    │
    └─────────────────────────────────┴────────┴──────────────────────────────┘
    """
    n = len(customer_ids)
    
    # ── Claims Frequency in 24 Months (30% weight) ──
    # High-risk customers file more claims
    claims_freq = np.clip(
        rng.poisson(lam=base_risk * 5 + 0.3, size=n), 0, 20
    ).astype(int)
    
    # ── Claim Rejection / Fraud Flag ──
    # Binary: 1 if any claim was rejected or flagged for fraud
    fraud_flag_prob = base_risk * 0.4 + 0.02
    fraud_flag = rng.binomial(1, np.clip(fraud_flag_prob, 0, 1)).astype(int)
    
    # ── Premium Lapse Count ──
    # Number of premium payments missed/lapsed
    premium_lapses = np.clip(
        rng.poisson(lam=base_risk * 3, size=n), 0, 12
    ).astype(int)
    
    # ── Policy Renewal Consistency ──
    # Fraction of policies renewed at term end (0-1)
    renewal_consistency = np.clip(
        1.0 - base_risk * 0.5 + rng.normal(0, 0.1, n), 0.0, 1.0
    )
    
    # ── Active Policies Held ──
    # More diverse coverage = potentially more responsible (or more exposure)
    active_policies = np.clip(
        rng.poisson(lam=(1.0 - base_risk) * 3 + 1, size=n), 0, 10
    ).astype(int)
    
    # ── Sum Assured (INR) — auxiliary ──
    sum_assured = np.clip(
        (1.0 - base_risk) * 3000000 + 500000 + rng.normal(0, 500000, n),
        100000, 10000000
    ).astype(int)
    
    # ── Customer Age — auxiliary ──
    customer_age = np.clip(
        rng.normal(40, 12, n), 21, 75
    ).astype(int)
    
    df = pd.DataFrame({
        "hashed_customer_id":         hashed_ids,
        "claims_frequency_24m":       claims_freq,
        "claim_rejection_fraud_flag": fraud_flag,
        "premium_lapse_count":        premium_lapses,
        "policy_renewal_consistency": np.round(renewal_consistency, 4),
        "active_policies_held":       active_policies,
        "sum_assured_inr":            sum_assured,
        "customer_age":               customer_age,
    })
    
    return df


# ════════════════════════════════════════════════════════════════════════════
# MAIN GENERATION PIPELINE
# ════════════════════════════════════════════════════════════════════════════

def generate_all_datasets(
    n_customers: int = 1000,
    seed: int = 42,
    output_dir: str = "./data"
) -> Dict[str, pd.DataFrame]:
    """
    Master pipeline:
    
    1. Create N unique customer IDs (simulated PAN/Aadhaar)
    2. Assign each customer a latent risk profile
    3. Sample institution-specific subsets (with overlap)
    4. Generate features for each institution's subset
    5. Write CSVs + hashed ID registry
    
    Returns:
        Dictionary of DataFrames keyed by institution name
    """
    rng = np.random.default_rng(seed)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"{'═' * 70}")
    print(f"  PS2 — Synthetic Dataset Generator")
    print(f"  Generating {n_customers} customers across 3 institutions")
    print(f"  Seed: {seed}  |  Output: {output_dir}/")
    print(f"{'═' * 70}\n")
    
    # ── Step 1: Generate raw customer IDs ──
    raw_ids = [f"CUST_{i:05d}" for i in range(n_customers)]
    hashed_ids = [generate_hashed_id(rid) for rid in raw_ids]
    
    # ── Step 2: Assign latent risk profiles ──
    base_risk, risk_profiles = assign_risk_profile(n_customers, rng)
    
    print(f"  Risk Distribution:")
    print(f"    Low:    {np.sum(risk_profiles == 'low'):>4d} customers")
    print(f"    Medium: {np.sum(risk_profiles == 'medium'):>4d} customers")
    print(f"    High:   {np.sum(risk_profiles == 'high'):>4d} customers\n")
    
    # ── Step 3: Sample institution-specific subsets ──
    # Use deterministic sampling with overlap
    all_indices = np.arange(n_customers)
    
    bank_size = int(n_customers * BANK_COVERAGE)
    lending_size = int(n_customers * LENDING_APP_COVERAGE)
    insurer_size = int(n_customers * INSURER_COVERAGE)
    
    # Shuffle and take first N for each institution
    bank_indices = np.sort(rng.choice(all_indices, size=bank_size, replace=False))
    lending_indices = np.sort(rng.choice(all_indices, size=lending_size, replace=False))
    insurer_indices = np.sort(rng.choice(all_indices, size=insurer_size, replace=False))
    
    # Compute overlaps for reporting
    bank_set = set(bank_indices)
    lending_set = set(lending_indices)
    insurer_set = set(insurer_indices)
    all_three = bank_set & lending_set & insurer_set
    
    print(f"  Institution Coverage:")
    print(f"    Bank Node:        {len(bank_indices):>4d} customers")
    print(f"    Lending App Node: {len(lending_indices):>4d} customers")
    print(f"    Insurer Node:     {len(insurer_indices):>4d} customers")
    print(f"    In all 3:         {len(all_three):>4d} customers\n")
    
    # ── Step 4: Generate institution-specific datasets ──
    print("  Generating Bank Node data...", end=" ")
    bank_df = generate_bank_data(
        [raw_ids[i] for i in bank_indices],
        [hashed_ids[i] for i in bank_indices],
        base_risk[bank_indices],
        rng
    )
    print(f"✓ ({len(bank_df)} rows, {len(bank_df.columns)} columns)")
    
    print("  Generating Lending App Node data...", end=" ")
    lending_df = generate_lending_app_data(
        [raw_ids[i] for i in lending_indices],
        [hashed_ids[i] for i in lending_indices],
        base_risk[lending_indices],
        rng
    )
    print(f"✓ ({len(lending_df)} rows, {len(lending_df.columns)} columns)")
    
    print("  Generating Insurer Node data...", end=" ")
    insurer_df = generate_insurer_data(
        [raw_ids[i] for i in insurer_indices],
        [hashed_ids[i] for i in insurer_indices],
        base_risk[insurer_indices],
        rng
    )
    print(f"✓ ({len(insurer_df)} rows, {len(insurer_df.columns)} columns)\n")
    
    # ── Step 5: Write CSVs ──
    bank_path = os.path.join(output_dir, "bank_node.csv")
    lending_path = os.path.join(output_dir, "lending_app_node.csv")
    insurer_path = os.path.join(output_dir, "insurer_node.csv")
    
    bank_df.to_csv(bank_path, index=False)
    lending_df.to_csv(lending_path, index=False)
    insurer_df.to_csv(insurer_path, index=False)
    
    print(f"  Written: {bank_path}")
    print(f"  Written: {lending_path}")
    print(f"  Written: {insurer_path}")
    
    # ── Step 6: Write hashed ID registry (ground truth for entity resolution) ──
    registry = pd.DataFrame({
        "raw_customer_id":    raw_ids,
        "hashed_customer_id": hashed_ids,
        "latent_risk_score":  np.round(base_risk, 4),
        "risk_profile":       risk_profiles,
        "in_bank":            [1 if i in bank_set else 0 for i in range(n_customers)],
        "in_lending_app":     [1 if i in lending_set else 0 for i in range(n_customers)],
        "in_insurer":         [1 if i in insurer_set else 0 for i in range(n_customers)],
    })
    
    registry_path = os.path.join(output_dir, "hashed_id_registry.csv")
    registry.to_csv(registry_path, index=False)
    print(f"  Written: {registry_path}")
    
    print(f"\n{'═' * 70}")
    print(f"  Dataset generation complete!")
    print(f"{'═' * 70}\n")
    
    return {
        "bank": bank_df,
        "lending_app": lending_df,
        "insurer": insurer_df,
        "registry": registry,
    }


# ════════════════════════════════════════════════════════════════════════════
# CLI ENTRY POINT
# ════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PS2 — Generate synthetic federated financial datasets"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument(
        "--n_customers", type=int, default=1000,
        help="Total number of unique customers (default: 1000)"
    )
    parser.add_argument(
        "--output_dir", type=str, default="./data",
        help="Output directory for CSV files (default: ./data)"
    )
    
    args = parser.parse_args()
    generate_all_datasets(
        n_customers=args.n_customers,
        seed=args.seed,
        output_dir=args.output_dir
    )
