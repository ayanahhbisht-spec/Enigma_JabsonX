"""
PS2 — Full Pipeline Runner
Run this single script to execute the entire pipeline end-to-end:
  1. Generate synthetic datasets
  2. Train local models
  3. Run federated aggregation + attack simulations
  4. Export results

Usage:
    cd backend
    python run_pipeline.py
"""

import os
import sys

def main():
    print("\n" + "=" * 70)
    print("  PS2 — Federated Ensemble Scoring: Full Pipeline")
    print("  Team JabsonX: Saksham, Joel, Ayana, Neha")
    print("=" * 70 + "\n")

    # Ensure we're in the right directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)

    # ── Step 1: Generate Datasets ──
    print("[1/3] Generating synthetic datasets...")
    from dataset_generator import generate_all_datasets
    generate_all_datasets(n_customers=1000, seed=42, output_dir="./data")

    # ── Step 2 & 3: Train + Aggregate + Attack Simulation ──
    print("[2/3] Training local models + running aggregation...")
    from federated_aggregator import main as run_aggregator
    aggregator = run_aggregator(data_dir=".", simulate_attack_type=None)

    print("\n[3/3] Pipeline complete!")
    print("  Output files:")
    print("    data/bank_node.csv")
    print("    data/lending_app_node.csv")
    print("    data/insurer_node.csv")
    print("    data/hashed_id_registry.csv")
    print("    models/*.joblib")
    print("    results/aggregated_risk_scores.csv")
    print("    results/security_alerts.json")
    print("    results/round_history.json")
    print("    results/node_trust_scores.json")
    print()

if __name__ == "__main__":
    main()
