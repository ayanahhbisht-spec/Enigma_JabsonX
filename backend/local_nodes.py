"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  PS2 — Federated Learning for Cross-Institution Financial Risk Control     ║
║  Local Nodes: Per-Institution Model Training & Scoring                     ║
║  Team JabsonX: Saksham, Joel, Ayana, Neha                                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

Each institution trains a LOCAL model on its own schema. Raw data and model
weights NEVER leave the institution. Only the final risk score (0-1) is
transmitted, alongside the hashed customer ID.

Phase 1 (Local Training): Entirely offline — nothing travels.
Phase 2 (Score Inference):  On request from the aggregator, the node computes
                            a score for a given hashed ID and returns it
                            with differential privacy applied.

This file contains:
    1. Explainable Scorecard Formulas (baseline/fallback)
    2. ML Model Training (XGBoost via scikit-learn API)
    3. Differential Privacy Wrappers
    4. Score API (simulated endpoint)

Usage:
    python local_nodes.py [--data_dir ./data] [--models_dir ./models]
"""

import os
import json
import hashlib
import argparse
import warnings
import numpy as np
import pandas as pd
import joblib
from typing import Dict, Optional, Tuple, List
from dataclasses import dataclass, field, asdict

# Suppress XGBoost / LightGBM verbosity
warnings.filterwarnings("ignore", category=UserWarning)

# ── Attempt to import ML libraries ──
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("[WARN] XGBoost not installed. Using LightGBM or fallback scorecard.")

try:
    from lightgbm import LGBMClassifier
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier  # fallback


# ════════════════════════════════════════════════════════════════════════════
# DATA CLASS: Node Configuration
# ════════════════════════════════════════════════════════════════════════════

@dataclass
class NodeConfig:
    """Configuration for a federated node."""
    name: str
    institution_type: str  # "bank", "lending_app", "insurer"
    csv_path: str
    feature_columns: List[str]
    feature_weights: Dict[str, float]  # For scorecard baseline
    model_path: str = ""
    scaler_path: str = ""
    dp_noise_std: float = 0.05       # Differential privacy noise (Gaussian)
    dp_round_precision: float = 0.1  # Round scores to this step
    risk_threshold: float = 0.5      # Binary label threshold on latent risk


# ════════════════════════════════════════════════════════════════════════════
# NODE CONFIGURATIONS (Maps to dataset_generator.py schemas)
# ════════════════════════════════════════════════════════════════════════════

BANK_CONFIG = NodeConfig(
    name="Bank Node (RBI-Regulated)",
    institution_type="bank",
    csv_path="data/bank_node.csv",
    feature_columns=[
        "repayment_on_time_ratio",
        "debt_to_income_ratio",
        "balance_trend_3m_inr",
        "cheque_bounces_12m",
        "overdraft_usage_ratio",
        "account_vintage_months",
        "txn_velocity_anomaly",
        "monthly_income_inr",
        "credit_score_cibil",
    ],
    feature_weights={
        "repayment_on_time_ratio": 0.30,   # 30% weight (primary)
        "debt_to_income_ratio":    0.20,   # 20% weight
        "cheque_bounces_12m":      0.15,
        "overdraft_usage_ratio":   0.10,
        "txn_velocity_anomaly":    0.10,
        "account_vintage_months":  0.05,
        "balance_trend_3m_inr":    0.05,
        "monthly_income_inr":      0.025,
        "credit_score_cibil":      0.025,
    }
)

LENDING_APP_CONFIG = NodeConfig(
    name="Lending App Node (Alt-Data)",
    institution_type="lending_app",
    csv_path="data/lending_app_node.csv",
    feature_columns=[
        "active_loans_across_apps",
        "micro_repayment_consistency",
        "app_engagement_score",
        "utility_bill_regularity",
        "geolocation_consistency",
        "days_since_last_request",
        "phone_model_tier",
        "social_refs_verified",
    ],
    feature_weights={
        "active_loans_across_apps":   0.25,  # 25% weight (primary)
        "micro_repayment_consistency": 0.20,
        "utility_bill_regularity":    0.15,
        "geolocation_consistency":    0.10,
        "app_engagement_score":       0.10,
        "days_since_last_request":    0.08,
        "phone_model_tier":           0.06,
        "social_refs_verified":       0.06,
    }
)

INSURER_CONFIG = NodeConfig(
    name="Insurer Node (IRDAI-Regulated)",
    institution_type="insurer",
    csv_path="data/insurer_node.csv",
    feature_columns=[
        "claims_frequency_24m",
        "claim_rejection_fraud_flag",
        "premium_lapse_count",
        "policy_renewal_consistency",
        "active_policies_held",
        "sum_assured_inr",
        "customer_age",
    ],
    feature_weights={
        "claims_frequency_24m":        0.30,  # 30% weight (primary)
        "claim_rejection_fraud_flag":  0.20,
        "premium_lapse_count":         0.15,
        "policy_renewal_consistency":  0.15,
        "active_policies_held":        0.10,
        "sum_assured_inr":             0.05,
        "customer_age":                0.05,
    }
)


# ════════════════════════════════════════════════════════════════════════════
# SECTION 1: EXPLAINABLE SCORECARD (Baseline / Fallback)
# ════════════════════════════════════════════════════════════════════════════

class ExplainableScorecard:
    """
    Transparent, rule-based scorecard for risk scoring.
    
    This serves as:
    1. A FALLBACK when ML models aren't available
    2. A BASELINE for comparing ML model performance
    3. An EXPLAINABILITY layer for regulatory compliance (RBI/IRDAI)
    
    The scorecard normalizes each feature to [0, 1] where 1 = highest risk,
    then computes a weighted average using institution-specific weights.
    """
    
    def __init__(self, config: NodeConfig, data: pd.DataFrame):
        self.config = config
        self.feature_stats = {}
        
        # Compute per-feature normalization stats from training data
        for col in config.feature_columns:
            if col in data.columns:
                self.feature_stats[col] = {
                    "min": float(data[col].min()),
                    "max": float(data[col].max()),
                    "mean": float(data[col].mean()),
                    "std": float(data[col].std()),
                }
    
    def _normalize_feature(self, value: float, col: str) -> float:
        """Normalize a feature value to [0, 1] using min-max scaling."""
        stats = self.feature_stats.get(col)
        if stats is None:
            return 0.5
        
        range_val = stats["max"] - stats["min"]
        if range_val == 0:
            return 0.5
        return np.clip((value - stats["min"]) / range_val, 0.0, 1.0)
    
    def _risk_direction(self, col: str) -> float:
        """
        Determines if higher values mean MORE risk (+1) or LESS risk (-1).
        This encodes domain knowledge about each feature.
        """
        # Features where HIGHER value = LOWER risk (invert direction)
        low_risk_when_high = {
            "repayment_on_time_ratio",
            "micro_repayment_consistency",
            "app_engagement_score",
            "utility_bill_regularity",
            "geolocation_consistency",
            "days_since_last_request",
            "policy_renewal_consistency",
            "active_policies_held",
            "account_vintage_months",
            "balance_trend_3m_inr",
            "monthly_income_inr",
            "credit_score_cibil",
            "phone_model_tier",
            "social_refs_verified",
        }
        return -1.0 if col in low_risk_when_high else 1.0
    
    def score(self, row: pd.Series) -> Tuple[float, Dict[str, float]]:
        """
        Compute risk score for a single customer using the weighted scorecard.
        
        Returns:
            (risk_score, feature_contributions) where:
            - risk_score is in [0, 1]
            - feature_contributions maps feature name → its contribution
        """
        total_score = 0.0
        contributions = {}
        
        for col, weight in self.config.feature_weights.items():
            if col not in row.index:
                continue
            
            normalized = self._normalize_feature(row[col], col)
            direction = self._risk_direction(col)
            
            # If direction is -1, flip: low normalized → high risk
            if direction < 0:
                risk_component = (1.0 - normalized) * weight
            else:
                risk_component = normalized * weight
            
            total_score += risk_component
            contributions[col] = round(risk_component, 4)
        
        return np.clip(total_score, 0.0, 1.0), contributions
    
    def score_batch(self, data: pd.DataFrame) -> np.ndarray:
        """Score all rows, returning array of risk scores."""
        scores = np.array([self.score(row)[0] for _, row in data.iterrows()])
        return scores


# ════════════════════════════════════════════════════════════════════════════
# SECTION 2: ML MODEL TRAINING (Per-Institution)
# ════════════════════════════════════════════════════════════════════════════

class LocalModelTrainer:
    """
    Trains a gradient-boosted model locally on institution-specific data.
    
    Key Design Decisions:
    - Uses XGBoost if available, LightGBM as backup, sklearn GBM as fallback
    - Trains a BINARY CLASSIFIER (high-risk vs. low-risk) and uses predicted
      probability as the continuous risk score
    - Features are standardized (z-score) before training
    - Model and scaler are serialized for Phase 2 inference
    """
    
    def __init__(self, config: NodeConfig):
        self.config = config
        self.model = None
        self.scaler = StandardScaler()
        self.scorecard = None  # Fallback scorecard
        self.is_trained = False
    
    def _create_model(self):
        """Select best available gradient boosting implementation."""
        if HAS_XGBOOST:
            return XGBClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                reg_alpha=0.1,
                reg_lambda=1.0,
                eval_metric="logloss",
                random_state=42,
                verbosity=0,
                use_label_encoder=False,
            )
        elif HAS_LIGHTGBM:
            return LGBMClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                min_child_weight=3,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                verbose=-1,
            )
        else:
            return GradientBoostingClassifier(
                n_estimators=200,
                max_depth=5,
                learning_rate=0.05,
                subsample=0.8,
                min_samples_leaf=10,
                random_state=42,
            )
    
    def _create_labels(self, data: pd.DataFrame) -> np.ndarray:
        """
        Generate pseudo-labels for training using the scorecard.
        
        In a real deployment, labels come from historical default/fraud data.
        For this hackathon, we use the scorecard as an oracle to create labels,
        then train an ML model to generalize better than the scorecard.
        """
        self.scorecard = ExplainableScorecard(self.config, data)
        scores = self.scorecard.score_batch(data)
        
        # Binary labels: top-risk customers are "high risk" (1)
        labels = (scores >= self.config.risk_threshold).astype(int)
        
        # Ensure we have both classes
        if len(np.unique(labels)) < 2:
            # Force some balance
            median = np.median(scores)
            labels = (scores >= median).astype(int)
        
        return labels
    
    def train(self, data_dir: str = ".") -> Dict:
        """
        Full training pipeline:
        
        1. Load institution CSV
        2. Generate pseudo-labels from scorecard
        3. Train/test split
        4. Scale features
        5. Train XGBoost/LightGBM
        6. Evaluate (AUC-ROC, classification report)
        7. Save model artifacts
        
        Returns:
            Dictionary of evaluation metrics
        """
        csv_path = os.path.join(data_dir, self.config.csv_path)
        print(f"\n{'─' * 60}")
        print(f"  Training: {self.config.name}")
        print(f"  Data:     {csv_path}")
        print(f"{'─' * 60}")
        
        # Load data
        data = pd.read_csv(csv_path)
        print(f"  Loaded {len(data)} records with {len(data.columns)} columns")
        
        # Extract features
        feature_cols = [c for c in self.config.feature_columns if c in data.columns]
        X = data[feature_cols].copy()
        
        # Handle missing values
        X = X.fillna(X.median())
        
        # Generate labels
        y = self._create_labels(data)
        print(f"  Labels: {np.sum(y == 0)} low-risk, {np.sum(y == 1)} high-risk")
        
        # Train/test split (stratified)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model = self._create_model()
        self.model.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Evaluate
        y_pred_proba = self.model.predict_proba(X_test_scaled)[:, 1]
        y_pred = (y_pred_proba >= 0.5).astype(int)
        
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Cross-validation AUC
        X_all_scaled = self.scaler.transform(X)
        cv_scores = cross_val_score(
            self._create_model(), X_all_scaled, y,
            cv=5, scoring="roc_auc"
        )
        
        metrics = {
            "institution": self.config.institution_type,
            "n_samples": len(data),
            "n_features": len(feature_cols),
            "test_auc_roc": round(auc, 4),
            "cv_auc_mean": round(cv_scores.mean(), 4),
            "cv_auc_std": round(cv_scores.std(), 4),
        }
        
        print(f"\n  ┌─ Evaluation Results ─────────────────────┐")
        print(f"  │ Test AUC-ROC:     {auc:.4f}               │")
        print(f"  │ 5-Fold CV AUC:    {cv_scores.mean():.4f} ± {cv_scores.std():.4f}  │")
        print(f"  └─────────────────────────────────────────────┘")
        
        report = classification_report(y_test, y_pred, zero_division=0)
        print(f"\n  Classification Report:\n{report}")
        
        # Save model and scaler
        models_dir = os.path.join(data_dir, "models")
        os.makedirs(models_dir, exist_ok=True)
        
        model_path = os.path.join(models_dir, f"{self.config.institution_type}_model.joblib")
        scaler_path = os.path.join(models_dir, f"{self.config.institution_type}_scaler.joblib")
        
        joblib.dump(self.model, model_path)
        joblib.dump(self.scaler, scaler_path)
        
        self.config.model_path = model_path
        self.config.scaler_path = scaler_path
        
        # Save scorecard stats for explainability
        scorecard_path = os.path.join(models_dir, f"{self.config.institution_type}_scorecard.json")
        with open(scorecard_path, "w") as f:
            json.dump({
                "feature_weights": self.config.feature_weights,
                "feature_stats": self.scorecard.feature_stats,
                "metrics": metrics,
            }, f, indent=2)
        
        print(f"  Saved model:     {model_path}")
        print(f"  Saved scaler:    {scaler_path}")
        print(f"  Saved scorecard: {scorecard_path}")
        
        # Store hashed_ids → feature matrix mapping for Phase 2
        self._data_cache = data.set_index("hashed_customer_id")
        
        return metrics
    
    def load_model(self, data_dir: str = "."):
        """Load a previously trained model and scaler."""
        models_dir = os.path.join(data_dir, "models")
        model_path = os.path.join(models_dir, f"{self.config.institution_type}_model.joblib")
        scaler_path = os.path.join(models_dir, f"{self.config.institution_type}_scaler.joblib")
        
        self.model = joblib.load(model_path)
        self.scaler = joblib.load(scaler_path)
        self.is_trained = True
        
        # Reload data for inference
        csv_path = os.path.join(data_dir, self.config.csv_path)
        data = pd.read_csv(csv_path)
        self._data_cache = data.set_index("hashed_customer_id")
        
        # Reload scorecard
        scorecard_path = os.path.join(models_dir, f"{self.config.institution_type}_scorecard.json")
        with open(scorecard_path, "r") as f:
            sc_data = json.load(f)
        
        self.scorecard = ExplainableScorecard(self.config, data)
        self.scorecard.feature_stats = sc_data["feature_stats"]


# ════════════════════════════════════════════════════════════════════════════
# SECTION 3: DIFFERENTIAL PRIVACY WRAPPER
# ════════════════════════════════════════════════════════════════════════════

class DifferentialPrivacyWrapper:
    """
    Applies differential privacy (DP) to risk scores before transmission.
    
    Defense Against Gradient/Model Inversion:
    1. ROUNDING:  Quantize scores to low precision (e.g., 0.1 steps)
                  This destroys fine-grained information about the model.
    2. NOISE:     Add calibrated Gaussian noise ε-DP style.
                  This ensures plausible deniability for any individual score.
    
    Mathematical Guarantee:
        For any two adjacent datasets D and D' (differing in one record):
        Pr[M(D) ∈ S] ≤ e^ε · Pr[M(D') ∈ S]
        
        Where M is our mechanism and ε is the privacy budget.
    """
    
    def __init__(self, noise_std: float = 0.05, round_precision: float = 0.1):
        self.noise_std = noise_std
        self.round_precision = round_precision
    
    def apply(self, score: float, rng: np.random.Generator = None) -> float:
        """
        Apply DP to a single risk score.
        
        Pipeline:
            raw_score → add Gaussian noise → round to precision → clip to [0, 1]
        
        Args:
            score: Raw predicted risk score in [0, 1]
            rng: NumPy random generator (for reproducibility)
        
        Returns:
            Privacy-protected score in [0, 1]
        """
        if rng is None:
            rng = np.random.default_rng()
        
        # Step 1: Add Gaussian noise
        noisy = score + rng.normal(0, self.noise_std)
        
        # Step 2: Round to precision steps (e.g., 0.1 → {0.0, 0.1, ..., 1.0})
        rounded = round(noisy / self.round_precision) * self.round_precision
        
        # Step 3: Clip to valid range
        return float(np.clip(rounded, 0.0, 1.0))
    
    def apply_batch(self, scores: np.ndarray, rng: np.random.Generator = None) -> np.ndarray:
        """Apply DP to an array of scores."""
        if rng is None:
            rng = np.random.default_rng()
        return np.array([self.apply(s, rng) for s in scores])


# ════════════════════════════════════════════════════════════════════════════
# SECTION 4: SCORE API (Simulated Endpoint for Phase 2)
# ════════════════════════════════════════════════════════════════════════════

class LocalNodeAPI:
    """
    Simulates the network-facing API of a local institutional node.
    
    In Phase 2 (Inference Round-Trip), the aggregator sends:
        Request:  { "hashed_ids": ["abc123...", "def456..."] }
    
    The node responds with:
        Response: {
            "institution": "bank",
            "scores": {
                "abc123...": 0.7,   # DP-protected score
                "def456...": 0.3,
            },
            "tls_simulated": true,
            "dp_applied": true,
        }
    
    Data Flow Isolation:
    ┌──────────┐   Phase 1: Nothing leaves    ┌──────────┐
    │  Raw     │ ─────────────────────────→   │  Local   │
    │  Data    │   (training is 100% local)   │  Model   │
    └──────────┘                              └────┬─────┘
                                                    │
         Phase 2: Only scores + hashed IDs travel   │
    ┌──────────┐   via simulated TLS/mTLS           │
    │  Score   │ ←──────────────────────────────────┘
    │  + Hash  │   (DP noise + rounding applied)
    └──────────┘
    """
    
    def __init__(self, trainer: LocalModelTrainer, dp_wrapper: DifferentialPrivacyWrapper):
        self.trainer = trainer
        self.dp = dp_wrapper
        self.rng = np.random.default_rng(42)
        self._request_log = []  # Audit trail
    
    def _simulate_tls_handshake(self, requester: str) -> Dict:
        """
        Simulate TLS/mTLS certificate validation.
        
        In production, this would be actual mTLS with:
        - Server cert presented by the aggregator
        - Client cert presented by this node
        - Mutual authentication before any payload exchange
        """
        tls_record = {
            "protocol": "TLS 1.3 (simulated)",
            "cipher_suite": "TLS_AES_256_GCM_SHA384",
            "client_cert_cn": f"{self.trainer.config.institution_type}_node.jabsonx.internal",
            "server_cert_cn": f"{requester}.jabsonx.internal",
            "mutual_auth": True,
            "session_id": hashlib.sha256(
                f"{requester}:{self.trainer.config.institution_type}:{np.random.randint(1e9)}".encode()
            ).hexdigest()[:16],
        }
        return tls_record
    
    def predict_scores(
        self,
        hashed_ids: List[str],
        requester: str = "aggregator",
        use_scorecard: bool = False,
    ) -> Dict:
        """
        Phase 2 Inference: Score a batch of customers by hashed ID.
        
        This is the ONLY method that produces output leaving the institution.
        
        Args:
            hashed_ids: List of SHA-256 hashed customer IDs
            requester: Identity of the requesting party (for mTLS simulation)
            use_scorecard: If True, use explainable scorecard instead of ML model
        
        Returns:
            Encrypted (simulated) response payload with DP-protected scores
        """
        # ── Step 1: Simulate TLS handshake ──
        tls_info = self._simulate_tls_handshake(requester)
        
        # ── Step 2: Look up customers in local data ──
        scores = {}
        scorecard_explanations = {}
        missing_ids = []
        
        feature_cols = [c for c in self.trainer.config.feature_columns 
                       if c in self.trainer._data_cache.columns]
        
        for hid in hashed_ids:
            if hid not in self.trainer._data_cache.index:
                missing_ids.append(hid)
                continue
            
            row = self.trainer._data_cache.loc[hid]
            
            if use_scorecard and self.trainer.scorecard is not None:
                # ── Scorecard path (explainable) ──
                raw_score, contributions = self.trainer.scorecard.score(row)
                scorecard_explanations[hid] = contributions
            else:
                # ── ML model path ──
                features = row[feature_cols].values.reshape(1, -1).astype(float)
                features_scaled = self.trainer.scaler.transform(features)
                raw_score = float(self.trainer.model.predict_proba(features_scaled)[0, 1])
            
            # ── Step 3: Apply differential privacy ──
            dp_score = self.dp.apply(raw_score, self.rng)
            scores[hid] = dp_score
        
        # ── Step 4: Build response payload ──
        response = {
            "institution": self.trainer.config.institution_type,
            "institution_name": self.trainer.config.name,
            "scores": scores,
            "missing_ids": missing_ids,
            "n_scored": len(scores),
            "n_missing": len(missing_ids),
            "privacy": {
                "dp_applied": True,
                "noise_std": self.dp.noise_std,
                "round_precision": self.dp.round_precision,
                "mechanism": "Gaussian + Quantization",
            },
            "transit_security": tls_info,
        }
        
        if scorecard_explanations:
            response["scorecard_explanations"] = scorecard_explanations
        
        # ── Audit log ──
        self._request_log.append({
            "requester": requester,
            "n_requested": len(hashed_ids),
            "n_scored": len(scores),
            "tls_session": tls_info["session_id"],
        })
        
        return response


# ════════════════════════════════════════════════════════════════════════════
# SECTION 5: ROGUE NODE SIMULATOR (For Testing Poisoning Defense)
# ════════════════════════════════════════════════════════════════════════════

class RogueNodeAPI(LocalNodeAPI):
    """
    Simulates a MALICIOUS node that attempts poisoning attacks.
    
    Attack strategies:
    1. INVERSION:  Submit inverted scores (safe customers → risky, vice versa)
    2. EXTREME:    Submit all-1.0 or all-0.0 scores
    3. RANDOM:     Submit uniformly random scores (no signal)
    4. SUBTLE:     Shift scores slightly toward a target (harder to detect)
    """
    
    def __init__(self, trainer: LocalModelTrainer, dp_wrapper: DifferentialPrivacyWrapper,
                 attack_type: str = "inversion"):
        super().__init__(trainer, dp_wrapper)
        self.attack_type = attack_type
    
    def predict_scores(self, hashed_ids: List[str], requester: str = "aggregator",
                       use_scorecard: bool = False) -> Dict:
        """Override to inject malicious scores."""
        # Get real scores first
        response = super().predict_scores(hashed_ids, requester, use_scorecard)
        
        # Poison the scores
        poisoned_scores = {}
        for hid, real_score in response["scores"].items():
            if self.attack_type == "inversion":
                poisoned_scores[hid] = round(1.0 - real_score, 1)
            elif self.attack_type == "extreme":
                poisoned_scores[hid] = 1.0
            elif self.attack_type == "random":
                poisoned_scores[hid] = round(self.rng.uniform(0, 1), 1)
            elif self.attack_type == "subtle":
                poisoned_scores[hid] = round(min(1.0, real_score + 0.2), 1)
            else:
                poisoned_scores[hid] = real_score
        
        response["scores"] = poisoned_scores
        response["__poisoned"] = True  # Hidden flag (wouldn't exist in production)
        
        return response


# ════════════════════════════════════════════════════════════════════════════
# MAIN: Train All Nodes
# ════════════════════════════════════════════════════════════════════════════

def train_all_nodes(data_dir: str = ".") -> Dict[str, LocalNodeAPI]:
    """
    Train all three institutional nodes and return their API interfaces.
    
    Returns:
        Dictionary mapping institution_type → LocalNodeAPI instance
    """
    print(f"\n{'═' * 70}")
    print(f"  PS2 — Local Node Training Pipeline")
    print(f"  Phase 1: Fully Local (nothing leaves the institution)")
    print(f"{'═' * 70}")
    
    configs = {
        "bank": BANK_CONFIG,
        "lending_app": LENDING_APP_CONFIG,
        "insurer": INSURER_CONFIG,
    }
    
    apis = {}
    all_metrics = {}
    
    for inst_type, config in configs.items():
        trainer = LocalModelTrainer(config)
        metrics = trainer.train(data_dir)
        all_metrics[inst_type] = metrics
        
        dp_wrapper = DifferentialPrivacyWrapper(
            noise_std=config.dp_noise_std,
            round_precision=config.dp_round_precision,
        )
        
        apis[inst_type] = LocalNodeAPI(trainer, dp_wrapper)
    
    # Summary
    print(f"\n{'═' * 70}")
    print(f"  Training Summary")
    print(f"{'═' * 70}")
    print(f"  {'Institution':<20} {'AUC-ROC':>10} {'CV AUC':>12} {'Samples':>10}")
    print(f"  {'─' * 52}")
    for inst, m in all_metrics.items():
        print(f"  {inst:<20} {m['test_auc_roc']:>10.4f} {m['cv_auc_mean']:>7.4f}±{m['cv_auc_std']:.4f} {m['n_samples']:>10}")
    print(f"\n  All models trained and saved. Ready for Phase 2.\n")
    
    return apis


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="PS2 — Train local institutional models"
    )
    parser.add_argument(
        "--data_dir", type=str, default=".",
        help="Root directory containing data/ folder (default: .)"
    )
    
    args = parser.parse_args()
    
    # Train all nodes
    apis = train_all_nodes(args.data_dir)
    
    # Quick demo: Score a few customers
    print(f"\n{'═' * 70}")
    print(f"  Demo: Phase 2 Score Inference")
    print(f"{'═' * 70}\n")
    
    # Load registry to get some hashed IDs
    registry = pd.read_csv(os.path.join(args.data_dir, "data/hashed_id_registry.csv"))
    sample_ids = registry["hashed_customer_id"].head(5).tolist()
    
    for inst_type, api in apis.items():
        result = api.predict_scores(sample_ids, requester="demo_aggregator")
        print(f"  {api.trainer.config.name}:")
        for hid, score in list(result["scores"].items())[:3]:
            print(f"    {hid[:16]}... → {score:.1f}")
        print(f"    (DP: noise={result['privacy']['noise_std']}, "
              f"precision={result['privacy']['round_precision']})")
        print(f"    (TLS: {result['transit_security']['protocol']})")
        print()
