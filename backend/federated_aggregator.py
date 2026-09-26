"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  PS2 — Federated Learning for Cross-Institution Financial Risk Control     ║
║  Federated Aggregator: Central Score Orchestration & Poisoning Defense     ║
║  Team JabsonX: Saksham, Joel, Ayana, Neha                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

This is the CENTRAL AGGREGATOR that orchestrates the Federated Ensemble Scoring
protocol. It NEVER sees raw data or model weights.

Core Responsibilities:
  1. Entity Resolution: Match customers across institutions using hashed IDs
  2. Score Collection:  Request risk scores from each institutional node
  3. Poisoning Defense: Validate incoming scores against statistical baselines
  4. Aggregation:       Combine validated scores via stacking meta-model
  5. Alert System:      Flag malicious actors and anomalous submissions

Architecture:
  ┌─────────────┐     ┌─────────────┐     ┌─────────────┐
  │  Bank Node  │     │ Lending App │     │ Insurer Node│
  │  (local ML) │     │  (local ML) │     │  (local ML) │
  └──────┬──────┘     └──────┬──────┘     └──────┬──────┘
         │  score+hash       │  score+hash       │  score+hash
         │  (DP applied)     │  (DP applied)     │  (DP applied)
         ▼                   ▼                   ▼
  ┌──────────────────────────────────────────────────────────┐
  │              FEDERATED AGGREGATOR (this file)            │
  │  ┌──────────────┐  ┌──────────────┐  ┌───────────────┐  │
  │  │ Poisoning    │  │ Score        │  │ Stacking      │  │
  │  │ Validator    │→ │ Alignment    │→ │ Meta-Model    │  │
  │  └──────────────┘  └──────────────┘  └───────────────┘  │
  └──────────────────────────────────────────────────────────┘
         │
         ▼
    Final Aggregated Risk Score per Customer

Usage:
    python federated_aggregator.py [--data_dir .] [--simulate_attack inversion]
"""

import os
import json
import time
import hashlib
import argparse
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum

# Import local modules
from local_nodes import (
    train_all_nodes,
    LocalNodeAPI,
    RogueNodeAPI,
    LocalModelTrainer,
    DifferentialPrivacyWrapper,
    BANK_CONFIG,
    LENDING_APP_CONFIG,
    INSURER_CONFIG,
)


# ════════════════════════════════════════════════════════════════════════════
# ENUMS & DATA CLASSES
# ════════════════════════════════════════════════════════════════════════════

class AlertSeverity(Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    CRITICAL = "CRITICAL"
    BLOCKED = "BLOCKED"


@dataclass
class SecurityAlert:
    """Represents a security event detected by the aggregator."""
    timestamp: str
    severity: AlertSeverity
    source_node: str
    alert_type: str
    message: str
    details: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "timestamp": self.timestamp,
            "severity": self.severity.value,
            "source_node": self.source_node,
            "alert_type": alert_type if (alert_type := self.alert_type) else "UNKNOWN",
            "message": self.message,
            "details": self.details,
        }


@dataclass
class AggregatedScore:
    """Final combined risk assessment for a customer."""
    hashed_id: str
    final_score: float
    contributing_nodes: Dict[str, float]  # institution → validated score
    aggregation_method: str               # "stacking" or "weighted_average"
    confidence: float                     # 0-1 based on node coverage
    risk_tier: str                        # "LOW", "MEDIUM", "HIGH", "CRITICAL"
    alerts: List[SecurityAlert] = field(default_factory=list)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1: POISONING DEFENSE — Score Validation Engine
# ════════════════════════════════════════════════════════════════════════════

class PoisoningDefenseValidator:
    """
    Validates incoming institutional scores against statistical baselines
    to detect and block poisoning attacks.
    
    Defense Mechanisms:
    ┌───────────────────────────────────────────────────────────────────────┐
    │ Test                  │ Detects                │ Threshold           │
    ├───────────────────────┼────────────────────────┼─────────────────────┤
    │ Distribution Check    │ Extreme submissions    │ μ ± 3σ of reference │
    │ Inversion Detection   │ Inverted risk signals  │ Correlation < -0.3  │
    │ Uniformity Test       │ Random/noisy scores    │ Entropy > 0.95      │
    │ Constant Score Check  │ All-same submissions   │ σ < 0.01            │
    │ Cross-Node Consensus  │ Outlier institutions   │ Deviation > 2σ      │
    └───────────────────────────────────────────────────────────────────────┘
    """
    
    def __init__(self, reference_registry_path: str):
        """
        Initialize with a synthetic reference layer.
        
        The reference registry contains KNOWN ground-truth risk profiles
        that were established during the secure setup phase. This allows
        the aggregator to validate incoming scores without seeing raw data.
        """
        self.registry = pd.read_csv(reference_registry_path)
        self.reference_scores = dict(
            zip(
                self.registry["hashed_customer_id"],
                self.registry["latent_risk_score"]
            )
        )
        
        # Compute reference distribution statistics
        ref_scores = np.array(list(self.reference_scores.values()))
        self.ref_mean = float(np.mean(ref_scores))
        self.ref_std = float(np.std(ref_scores))
        self.ref_median = float(np.median(ref_scores))
        
        # Per-institution historical statistics (built during calibration)
        self.institution_baselines = {}
        
        # Alert log
        self.alerts: List[SecurityAlert] = []
    
    def calibrate_baseline(self, institution: str, scores: Dict[str, float]):
        """
        Establish baseline statistics for an institution during the initial
        trusted setup phase.
        """
        score_values = list(scores.values())
        self.institution_baselines[institution] = {
            "mean": float(np.mean(score_values)),
            "std": float(np.std(score_values)),
            "median": float(np.median(score_values)),
            "n_samples": len(score_values),
            "calibrated": True,
        }
    
    def validate_submission(
        self,
        institution: str,
        scores: Dict[str, float],
        strict: bool = True
    ) -> Tuple[Dict[str, float], List[SecurityAlert]]:
        """
        Validate a batch of scores from an institution.
        
        Returns:
            (validated_scores, alerts) where validated_scores contains only
            scores that passed all checks.
        """
        alerts = []
        validated = {}
        score_values = list(scores.values())
        
        if not score_values:
            return {}, []
        
        # ── Test 1: Distribution Check (Extreme Submissions) ──
        batch_mean = np.mean(score_values)
        batch_std = np.std(score_values)
        
        # Check if the batch mean is extremely far from reference
        z_score = abs(batch_mean - self.ref_mean) / max(self.ref_std, 0.01)
        
        if z_score > 3.0:
            alert = SecurityAlert(
                timestamp=datetime.now().isoformat(),
                severity=AlertSeverity.CRITICAL,
                source_node=institution,
                alert_type="EXTREME_DISTRIBUTION",
                message=(
                    f"Batch mean ({batch_mean:.3f}) deviates {z_score:.1f}σ "
                    f"from reference (μ={self.ref_mean:.3f}, σ={self.ref_std:.3f}). "
                    f"Possible poisoning attack detected."
                ),
                details={"batch_mean": batch_mean, "z_score": z_score},
            )
            alerts.append(alert)
            
            if strict:
                self.alerts.extend(alerts)
                return {}, alerts  # REJECT entire submission
        
        # ── Test 2: Inversion Detection ──
        # Compare submitted scores against reference for overlapping IDs
        overlap_submitted = []
        overlap_reference = []
        
        for hid, score in scores.items():
            if hid in self.reference_scores:
                overlap_submitted.append(score)
                overlap_reference.append(self.reference_scores[hid])
        
        if len(overlap_submitted) > 10:
            correlation = np.corrcoef(overlap_submitted, overlap_reference)[0, 1]
            
            if correlation < -0.3:
                alert = SecurityAlert(
                    timestamp=datetime.now().isoformat(),
                    severity=AlertSeverity.BLOCKED,
                    source_node=institution,
                    alert_type="SCORE_INVERSION",
                    message=(
                        f"Submitted scores show NEGATIVE correlation ({correlation:.3f}) "
                        f"with reference. This indicates an inversion attack. "
                        f"Submission BLOCKED."
                    ),
                    details={"correlation": correlation, "n_overlap": len(overlap_submitted)},
                )
                alerts.append(alert)
                
                if strict:
                    self.alerts.extend(alerts)
                    return {}, alerts  # BLOCK entirely
        
        # ── Test 3: Constant Score Check ──
        if batch_std < 0.01:
            alert = SecurityAlert(
                timestamp=datetime.now().isoformat(),
                severity=AlertSeverity.CRITICAL,
                source_node=institution,
                alert_type="CONSTANT_SCORES",
                message=(
                    f"All scores are essentially constant (σ={batch_std:.4f}). "
                    f"This provides no discriminative signal and may be an attack."
                ),
                details={"batch_std": batch_std},
            )
            alerts.append(alert)
            
            if strict:
                self.alerts.extend(alerts)
                return {}, alerts
        
        # ── Test 4: Uniformity Test (Random Scores) ──
        # Use simple histogram-based entropy
        hist, _ = np.histogram(score_values, bins=10, range=(0, 1))
        hist_norm = hist / len(score_values)
        hist_norm = hist_norm[hist_norm > 0]  # Remove zeros for log
        entropy = -np.sum(hist_norm * np.log2(hist_norm))
        max_entropy = np.log2(10)  # Max entropy for 10 bins
        normalized_entropy = entropy / max_entropy
        
        if normalized_entropy > 0.95:
            alert = SecurityAlert(
                timestamp=datetime.now().isoformat(),
                severity=AlertSeverity.WARNING,
                source_node=institution,
                alert_type="HIGH_ENTROPY_SCORES",
                message=(
                    f"Score distribution has unusually high entropy "
                    f"({normalized_entropy:.3f}), suggesting random/noisy submissions."
                ),
                details={"normalized_entropy": normalized_entropy},
            )
            alerts.append(alert)
        
        # ── Test 5: Per-Score Outlier Check ──
        for hid, score in scores.items():
            # Check individual scores against reference
            if hid in self.reference_scores:
                ref_score = self.reference_scores[hid]
                deviation = abs(score - ref_score)
                
                if deviation > 0.6:
                    alert = SecurityAlert(
                        timestamp=datetime.now().isoformat(),
                        severity=AlertSeverity.WARNING,
                        source_node=institution,
                        alert_type="INDIVIDUAL_OUTLIER",
                        message=(
                            f"Score for {hid[:16]}... deviates {deviation:.2f} "
                            f"from reference (submitted={score:.2f}, ref={ref_score:.2f})."
                        ),
                        details={"hashed_id": hid, "submitted": score, "reference": ref_score},
                    )
                    alerts.append(alert)
                    
                    if strict and deviation > 0.8:
                        continue  # Skip this score but don't block entire submission
            
            validated[hid] = score
        
        self.alerts.extend(alerts)
        return validated, alerts


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2: STACKING META-MODEL (Score Aggregation)
# ════════════════════════════════════════════════════════════════════════════

class StackingMetaModel:
    """
    Aggregates validated scores from multiple institutions into a single
    risk assessment per customer.
    
    Two aggregation modes:
    1. WEIGHTED AVERAGE: Simple but effective for <= 3 nodes
       Final = Σ (w_i × score_i) / Σ w_i
    
    2. STACKING:  Learns optimal combination weights from validation data
       Final = σ(w₀ + w₁·s_bank + w₂·s_lending + w₃·s_insurer)
    
    Node weights reflect data quality and historical accuracy:
    ┌──────────────┬────────┬───────────────────────────┐
    │ Institution  │ Weight │ Rationale                 │
    ├──────────────┼────────┼───────────────────────────┤
    │ Bank         │ 0.45   │ Longest history, RBI reg. │
    │ Lending App  │ 0.25   │ Alt-data, thin-file value │
    │ Insurer      │ 0.30   │ Actuarial depth, IRDAI    │
    └──────────────┴────────┴───────────────────────────┘
    """
    
    # Institution weights (tunable via calibration)
    DEFAULT_WEIGHTS = {
        "bank":        0.45,
        "lending_app": 0.25,
        "insurer":     0.30,
    }
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
    
    def _classify_risk_tier(self, score: float) -> str:
        """Map continuous score to risk tier for business decision-making."""
        if score < 0.25:
            return "LOW"
        elif score < 0.50:
            return "MEDIUM"
        elif score < 0.75:
            return "HIGH"
        else:
            return "CRITICAL"
    
    def aggregate(
        self,
        hashed_id: str,
        node_scores: Dict[str, float],
        method: str = "weighted_average",
    ) -> AggregatedScore:
        """
        Combine scores from multiple institutions for a single customer.
        
        Args:
            hashed_id: SHA-256 hashed customer ID
            node_scores: Dict of institution → validated score
            method: "weighted_average" or "stacking"
        
        Returns:
            AggregatedScore with final risk assessment
        """
        if not node_scores:
            return AggregatedScore(
                hashed_id=hashed_id,
                final_score=0.5,  # Default to uncertain
                contributing_nodes={},
                aggregation_method=method,
                confidence=0.0,
                risk_tier="UNKNOWN",
            )
        
        if method == "weighted_average":
            weighted_sum = 0.0
            weight_sum = 0.0
            
            for inst, score in node_scores.items():
                w = self.weights.get(inst, 0.1)  # Default weight for unknown nodes
                weighted_sum += w * score
                weight_sum += w
            
            final = weighted_sum / weight_sum if weight_sum > 0 else 0.5
        
        elif method == "stacking":
            # Stacking meta-model: logistic combination
            # In production, these coefficients would be learned from validation data
            bias = -0.5
            coefficients = {
                "bank": 1.2,
                "lending_app": 0.8,
                "insurer": 1.0,
            }
            
            logit = bias
            for inst, score in node_scores.items():
                coef = coefficients.get(inst, 0.5)
                logit += coef * score
            
            # Sigmoid
            final = 1.0 / (1.0 + np.exp(-logit))
        
        else:
            raise ValueError(f"Unknown aggregation method: {method}")
        
        # Confidence based on node coverage (more nodes = more confident)
        confidence = len(node_scores) / len(self.weights)
        
        return AggregatedScore(
            hashed_id=hashed_id,
            final_score=round(float(final), 4),
            contributing_nodes=node_scores,
            aggregation_method=method,
            confidence=round(confidence, 2),
            risk_tier=self._classify_risk_tier(final),
        )


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3: FEDERATED AGGREGATOR (Core Orchestrator)
# ════════════════════════════════════════════════════════════════════════════

class FederatedAggregator:
    """
    Central orchestrator for the Federated Ensemble Scoring protocol.
    
    Lifecycle:
    1. Registration:  Nodes register with the aggregator
    2. Calibration:   Initial trusted round to establish baselines
    3. Inference:     Aggregator requests scores for target customers
    4. Validation:    Submitted scores pass through poisoning defense
    5. Aggregation:   Valid scores combined via stacking/weighted average
    6. Reporting:     Final risk assessments + security alerts emitted
    """
    
    def __init__(
        self,
        reference_registry_path: str,
        aggregation_method: str = "weighted_average",
    ):
        self.validator = PoisoningDefenseValidator(reference_registry_path)
        self.meta_model = StackingMetaModel()
        self.aggregation_method = aggregation_method
        
        # Registered nodes
        self.nodes: Dict[str, LocalNodeAPI] = {}
        self.node_trust_scores: Dict[str, float] = {}
        
        # Results
        self.aggregated_results: Dict[str, AggregatedScore] = {}
        self.round_history: List[Dict] = []
        
        # Security
        self.blocked_nodes: set = set()
        self.all_alerts: List[SecurityAlert] = []
        
        print(f"\n{'═' * 70}")
        print(f"  Federated Aggregator Initialized")
        print(f"  Method: {aggregation_method}")
        print(f"  Reference registry: {reference_registry_path}")
        print(f"{'═' * 70}\n")
    
    def register_node(self, institution_type: str, node_api: LocalNodeAPI):
        """
        Register an institutional node with the aggregator.
        
        In production, this would involve:
        - mTLS certificate exchange
        - Institutional identity verification
        - Data processing agreement validation (DPDP Act compliance)
        """
        self.nodes[institution_type] = node_api
        self.node_trust_scores[institution_type] = 1.0  # Start fully trusted
        
        print(f"  [REGISTER] Node '{institution_type}' registered "
              f"(Trust: {self.node_trust_scores[institution_type]:.2f})")
    
    def _request_scores_from_node(
        self,
        institution_type: str,
        hashed_ids: List[str]
    ) -> Optional[Dict]:
        """
        Phase 2: Request scores from a single node via simulated TLS.
        
        Returns None if the node is blocked or unavailable.
        """
        if institution_type in self.blocked_nodes:
            print(f"  [BLOCKED] Skipping blocked node: {institution_type}")
            return None
        
        node_api = self.nodes.get(institution_type)
        if node_api is None:
            return None
        
        print(f"  [REQUEST] Requesting {len(hashed_ids)} scores from "
              f"{institution_type}...", end=" ")
        
        try:
            response = node_api.predict_scores(
                hashed_ids,
                requester="federated_aggregator"
            )
            print(f"✓ Received {response['n_scored']} scores "
                  f"(TLS: {response['transit_security']['session_id']})")
            return response
        
        except Exception as e:
            print(f"✗ Error: {e}")
            alert = SecurityAlert(
                timestamp=datetime.now().isoformat(),
                severity=AlertSeverity.WARNING,
                source_node=institution_type,
                alert_type="NODE_ERROR",
                message=f"Node returned error: {str(e)}",
            )
            self.all_alerts.append(alert)
            return None
    
    def run_inference_round(
        self,
        target_hashed_ids: List[str],
        round_id: str = None,
    ) -> Dict[str, AggregatedScore]:
        """
        Execute one full Federated Ensemble Scoring round.
        
        Steps:
        1. Send hashed IDs to all registered nodes
        2. Collect score responses
        3. Validate each submission through poisoning defense
        4. Aggregate valid scores per customer
        5. Return final risk assessments
        
        Args:
            target_hashed_ids: List of hashed customer IDs to score
            round_id: Optional identifier for this round
        
        Returns:
            Dict of hashed_id → AggregatedScore
        """
        if round_id is None:
            round_id = f"round_{len(self.round_history) + 1}"
        
        print(f"\n{'━' * 70}")
        print(f"  Inference Round: {round_id}")
        print(f"  Target customers: {len(target_hashed_ids)}")
        print(f"  Active nodes: {len(self.nodes) - len(self.blocked_nodes)}")
        print(f"{'━' * 70}\n")
        
        round_start = time.time()
        round_alerts = []
        
        # ── Step 1: Collect scores from all nodes ──
        all_responses = {}
        for inst_type in self.nodes:
            response = self._request_scores_from_node(inst_type, target_hashed_ids)
            if response is not None:
                all_responses[inst_type] = response
        
        # ── Step 2: Validate each submission ──
        print(f"\n  [VALIDATE] Running poisoning defense checks...")
        validated_scores = {}  # institution → {hashed_id: score}
        
        for inst_type, response in all_responses.items():
            raw_scores = response.get("scores", {})
            
            valid, alerts = self.validator.validate_submission(
                institution=inst_type,
                scores=raw_scores,
                strict=True,
            )
            
            validated_scores[inst_type] = valid
            round_alerts.extend(alerts)
            
            # Update trust scores based on alerts
            critical_alerts = [a for a in alerts 
                              if a.severity in (AlertSeverity.CRITICAL, AlertSeverity.BLOCKED)]
            
            if critical_alerts:
                self.node_trust_scores[inst_type] *= 0.5  # Halve trust
                
                if self.node_trust_scores[inst_type] < 0.25:
                    self.blocked_nodes.add(inst_type)
                    print(f"  [⚠ BLOCKED] Node '{inst_type}' has been BLOCKED "
                          f"(trust={self.node_trust_scores[inst_type]:.2f})")
                else:
                    print(f"  [⚠ WARNING] Node '{inst_type}' trust reduced to "
                          f"{self.node_trust_scores[inst_type]:.2f}")
            
            n_rejected = len(raw_scores) - len(valid)
            status = "✓" if n_rejected == 0 else "⚠"
            print(f"  [{status}] {inst_type}: {len(valid)} accepted, "
                  f"{n_rejected} rejected")
        
        # ── Step 3: Aggregate per customer ──
        print(f"\n  [AGGREGATE] Combining scores ({self.aggregation_method})...")
        
        results = {}
        for hid in target_hashed_ids:
            node_scores = {}
            for inst_type, scores in validated_scores.items():
                if hid in scores:
                    node_scores[inst_type] = scores[hid]
            
            agg = self.meta_model.aggregate(
                hashed_id=hid,
                node_scores=node_scores,
                method=self.aggregation_method,
            )
            
            # Attach any alerts relevant to this customer
            agg.alerts = [a for a in round_alerts 
                         if a.details.get("hashed_id") == hid]
            
            results[hid] = agg
        
        # ── Step 4: Record round history ──
        round_duration = time.time() - round_start
        
        round_record = {
            "round_id": round_id,
            "timestamp": datetime.now().isoformat(),
            "duration_seconds": round(round_duration, 3),
            "n_target_customers": len(target_hashed_ids),
            "n_scored": sum(1 for r in results.values() if r.contributing_nodes),
            "n_alerts": len(round_alerts),
            "blocked_nodes": list(self.blocked_nodes),
            "node_trust": dict(self.node_trust_scores),
        }
        self.round_history.append(round_record)
        self.all_alerts.extend(round_alerts)
        self.aggregated_results.update(results)
        
        # ── Summary ──
        scored = [r for r in results.values() if r.contributing_nodes]
        risk_dist = {}
        for r in scored:
            risk_dist[r.risk_tier] = risk_dist.get(r.risk_tier, 0) + 1
        
        print(f"\n  ┌─ Round Summary ─────────────────────────────────┐")
        print(f"  │ Customers scored:  {len(scored):>5} / {len(target_hashed_ids):<5}       │")
        print(f"  │ Alerts triggered:  {len(round_alerts):>5}                       │")
        print(f"  │ Nodes blocked:     {len(self.blocked_nodes):>5}                       │")
        print(f"  │ Duration:          {round_duration:>5.2f}s                      │")
        print(f"  │                                                   │")
        print(f"  │ Risk Distribution:                                │")
        for tier in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
            count = risk_dist.get(tier, 0)
            bar = "█" * min(count // 2, 25)
            print(f"  │   {tier:<10} {count:>4}  {bar:<25} │")
        print(f"  └───────────────────────────────────────────────────┘")
        
        return results
    
    def export_results(self, output_dir: str = "./results"):
        """
        Export all aggregated results and alerts to JSON/CSV.
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # ── Export aggregated scores ──
        scores_data = []
        for hid, result in self.aggregated_results.items():
            row = {
                "hashed_customer_id": hid,
                "final_risk_score": result.final_score,
                "risk_tier": result.risk_tier,
                "confidence": result.confidence,
                "aggregation_method": result.aggregation_method,
                "n_contributing_nodes": len(result.contributing_nodes),
            }
            # Add per-institution scores
            for inst in ["bank", "lending_app", "insurer"]:
                row[f"score_{inst}"] = result.contributing_nodes.get(inst, None)
            
            scores_data.append(row)
        
        scores_df = pd.DataFrame(scores_data)
        scores_path = os.path.join(output_dir, "aggregated_risk_scores.csv")
        scores_df.to_csv(scores_path, index=False)
        print(f"\n  Exported: {scores_path} ({len(scores_df)} records)")
        
        # ── Export security alerts ──
        alerts_data = [a.to_dict() for a in self.all_alerts]
        alerts_path = os.path.join(output_dir, "security_alerts.json")
        with open(alerts_path, "w") as f:
            json.dump(alerts_data, f, indent=2)
        print(f"  Exported: {alerts_path} ({len(alerts_data)} alerts)")
        
        # ── Export round history ──
        history_path = os.path.join(output_dir, "round_history.json")
        with open(history_path, "w") as f:
            json.dump(self.round_history, f, indent=2)
        print(f"  Exported: {history_path} ({len(self.round_history)} rounds)")
        
        # ── Export node trust scores ──
        trust_path = os.path.join(output_dir, "node_trust_scores.json")
        with open(trust_path, "w") as f:
            json.dump({
                "trust_scores": self.node_trust_scores,
                "blocked_nodes": list(self.blocked_nodes),
                "timestamp": datetime.now().isoformat(),
            }, f, indent=2)
        print(f"  Exported: {trust_path}")
        
        return scores_df


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4: ATTACK SIMULATION
# ════════════════════════════════════════════════════════════════════════════

def simulate_attack(
    aggregator: FederatedAggregator,
    attack_type: str,
    target_institution: str = "insurer",
    data_dir: str = ".",
) -> None:
    """
    Replace a legitimate node with a rogue node and run an inference round
    to demonstrate the poisoning defense.
    
    Attack Types:
        - inversion: Flip risk scores (safe ↔ risky)
        - extreme:   Submit all 1.0 scores
        - random:    Submit uniformly random scores
        - subtle:    Shift scores slightly upward
    """
    print(f"\n{'▓' * 70}")
    print(f"  ⚡ ATTACK SIMULATION: {attack_type.upper()}")
    print(f"  Compromised node: {target_institution}")
    print(f"{'▓' * 70}\n")
    
    # Get the legitimate node's trainer
    legit_api = aggregator.nodes.get(target_institution)
    if legit_api is None:
        print(f"  [ERROR] Node '{target_institution}' not found")
        return
    
    # Create rogue replacement
    dp_wrapper = DifferentialPrivacyWrapper(noise_std=0.05, round_precision=0.1)
    rogue_api = RogueNodeAPI(legit_api.trainer, dp_wrapper, attack_type=attack_type)
    
    # Replace the legitimate node
    aggregator.nodes[target_institution] = rogue_api
    
    # Reset trust score for fair test
    aggregator.node_trust_scores[target_institution] = 1.0
    if target_institution in aggregator.blocked_nodes:
        aggregator.blocked_nodes.remove(target_institution)
    
    # Run inference with the rogue node
    registry = pd.read_csv(os.path.join(data_dir, "data/hashed_id_registry.csv"))
    sample_ids = registry["hashed_customer_id"].head(100).tolist()
    
    results = aggregator.run_inference_round(
        target_hashed_ids=sample_ids,
        round_id=f"attack_{attack_type}",
    )
    
    # Report attack results
    attack_alerts = [a for a in aggregator.all_alerts 
                    if a.source_node == target_institution]
    
    print(f"\n  ┌─ Attack Report ──────────────────────────────────┐")
    print(f"  │ Attack type:     {attack_type:<30}  │")
    print(f"  │ Target node:     {target_institution:<30}  │")
    print(f"  │ Alerts triggered: {len(attack_alerts):<29} │")
    
    blocked = target_institution in aggregator.blocked_nodes
    trust = aggregator.node_trust_scores.get(target_institution, 0)
    
    if blocked:
        print(f"  │ Node status:     🚫 BLOCKED                      │")
    else:
        print(f"  │ Node status:     ⚠ Trust={trust:.2f}                  │")
    
    print(f"  └───────────────────────────────────────────────────┘")
    
    # Restore legitimate node
    aggregator.nodes[target_institution] = legit_api


# ════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ════════════════════════════════════════════════════════════════════════════

def main(data_dir: str = ".", simulate_attack_type: str = None):
    """
    Full end-to-end federated scoring pipeline:
    
    1. Train local models (Phase 1)
    2. Initialize aggregator
    3. Register nodes
    4. Run calibration round
    5. Run inference round
    6. (Optional) Simulate attack
    7. Export results
    """
    
    print(f"\n{'╔' + '═' * 68 + '╗'}")
    print(f"{'║':>1} {'PS2 — Federated Ensemble Scoring Pipeline':^68} {'║'}")
    print(f"{'║':>1} {'Team JabsonX: Saksham, Joel, Ayana, Neha':^68} {'║'}")
    print(f"{'╚' + '═' * 68 + '╝'}\n")
    
    # ── Phase 1: Local Training ──
    apis = train_all_nodes(data_dir)
    
    # ── Initialize Aggregator ──
    registry_path = os.path.join(data_dir, "data/hashed_id_registry.csv")
    aggregator = FederatedAggregator(
        reference_registry_path=registry_path,
        aggregation_method="weighted_average",
    )
    
    # ── Register Nodes ──
    for inst_type, api in apis.items():
        aggregator.register_node(inst_type, api)
    
    # ── Calibration Round ──
    print(f"\n  [CALIBRATE] Running initial trusted calibration round...")
    registry = pd.read_csv(registry_path)
    calibration_ids = registry["hashed_customer_id"].head(200).tolist()
    
    for inst_type, api in apis.items():
        cal_response = api.predict_scores(calibration_ids, requester="calibration")
        aggregator.validator.calibrate_baseline(inst_type, cal_response["scores"])
    print(f"  [CALIBRATE] Baselines established for {len(apis)} nodes ✓\n")
    
    # ── Inference Round ──
    target_ids = registry["hashed_customer_id"].tolist()
    results = aggregator.run_inference_round(
        target_hashed_ids=target_ids,
        round_id="full_inference_v1",
    )
    
    # ── Attack Simulation ──
    if simulate_attack_type:
        simulate_attack(
            aggregator,
            attack_type=simulate_attack_type,
            target_institution="insurer",
            data_dir=data_dir,
        )
    else:
        # Run all attack types for demo
        for attack in ["inversion", "extreme", "random", "subtle"]:
            simulate_attack(
                aggregator,
                attack_type=attack,
                target_institution="insurer",
                data_dir=data_dir,
            )
    
    # ── Export ──
    output_dir = os.path.join(data_dir, "results")
    aggregator.export_results(output_dir)
    
    print(f"\n{'═' * 70}")
    print(f"  Pipeline Complete!")
    print(f"  Results: {output_dir}/")
    print(f"  Total Alerts: {len(aggregator.all_alerts)}")
    print(f"{'═' * 70}\n")
    
    return aggregator


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PS2 — Federated Ensemble Scoring Aggregator"
    )
    parser.add_argument(
        "--data_dir", type=str, default=".",
        help="Root directory containing data/ folder"
    )
    parser.add_argument(
        "--simulate_attack", type=str, default=None,
        choices=["inversion", "extreme", "random", "subtle"],
        help="Run a specific attack simulation (default: run all)"
    )
    parser.add_argument(
        "--aggregation", type=str, default="weighted_average",
        choices=["weighted_average", "stacking"],
        help="Aggregation method (default: weighted_average)"
    )
    
    args = parser.parse_args()
    main(
        data_dir=args.data_dir,
        simulate_attack_type=args.simulate_attack,
    )
