"""
trainer.py — train a breakout classifier from labeled (zone, breakout) pairs.

Strategy
────────
• Positive examples  : each (triangle_zone + triangle_breakout) pair
• Negative examples  : synthetic — take each zone, slide a "fake breakout" candle
                       at a random position *inside* the zone (not a real breakout)
• Model              : XGBoostClassifier (fast, handles small datasets well)
• Evaluation         : Leave-One-Out CV on the small dataset
• Fallback           : if < MIN_EXAMPLES_FOR_ML pairs found, use RuleBasedScorer

The trained model + scaler are persisted to disk so the scanner can load them
without re-training on every run.  Re-training is triggered explicitly via
  python main.py --train
"""

import json
import logging
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.model_selection import LeaveOneOut
from sklearn.preprocessing import RobustScaler
from xgboost import XGBClassifier

from config import (
    FEATURE_IMPORTANCE_PATH,
    MIN_EXAMPLES_FOR_ML,
    MODEL_PATH,
    MIN_VOLUME_RATIO,
)
from feature_extractor import (
    FEATURE_NAMES,
    extract_features,
    features_to_array,
)
from questdb_reader import fetch_paired_examples

logger = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class TrainingResult:
    mode: str                   # "ml" or "rule_based"
    n_positive: int
    n_negative: int
    loocv_accuracy: Optional[float]
    loocv_precision: Optional[float]
    loocv_recall: Optional[float]
    feature_importances: Optional[dict]
    model_path: Optional[str]


# ── Negative example generator ────────────────────────────────────────────────

def _generate_negatives(pairs: list[dict], n_per_positive: int = 2) -> list[tuple[np.ndarray, int]]:
    """
    Generate synthetic negative examples from the zone candles.

    Strategy: for each zone, pick 1–2 random candles *inside* the zone
    and treat them as "fake breakout" candles.  The features will look
    like an in-zone candle, not a real breakout.
    """
    negatives = []

    for pair in pairs:
        zone = pair["zone_candles"]
        if len(zone) < 6:
            continue

        # Pick random candles from the middle of the zone (avoid first/last)
        indices = list(range(2, len(zone) - 2))
        chosen = random.sample(indices, min(n_per_positive, len(indices)))

        for idx in chosen:
            fake_breakout = zone.iloc[idx]
            feats = extract_features(
                zone_candles=zone.iloc[:idx],   # zone up to this point
                breakout_candle=fake_breakout,
                zone_to_ts=zone.iloc[idx - 1]["ts"] if idx > 0 else None,
            )
            if feats is not None:
                negatives.append((features_to_array(feats), 0))

    logger.info("Generated %d synthetic negative examples", len(negatives))
    return negatives


# ── Build dataset ─────────────────────────────────────────────────────────────

def build_dataset(pairs: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """
    Build X (feature matrix) and y (labels) from paired examples + synthetics.
    """
    positives = []
    for pair in pairs:
        feats = extract_features(
            zone_candles=pair["zone_candles"],
            breakout_candle=pair["breakout_candle"],
            zone_to_ts=pair["zone_to"],
        )
        if feats is not None:
            positives.append((features_to_array(feats), 1))
        else:
            logger.warning("Feature extraction failed for pair ticker=%s", pair["ticker"])

    if not positives:
        raise ValueError("No valid positive examples after feature extraction")

    negatives = _generate_negatives(pairs)

    all_samples = positives + negatives
    random.shuffle(all_samples)

    X = np.array([s[0] for s in all_samples])
    y = np.array([s[1] for s in all_samples])

    logger.info("Dataset: %d positives, %d negatives, %d total",
                len(positives), len(negatives), len(all_samples))

    return X, y


# ── LOOCV evaluation ──────────────────────────────────────────────────────────

def _loocv_evaluate(X: np.ndarray, y: np.ndarray) -> dict:
    """
    Run Leave-One-Out CV.  Returns accuracy, precision, recall.
    With ~20 examples this is the right eval strategy (no held-out test set waste).
    """
    loo = LeaveOneOut()
    y_true, y_pred = [], []

    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train         = y[train_idx]

        scaler = RobustScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s  = scaler.transform(X_test)

        clf = _make_classifier()
        clf.fit(X_train_s, y_train)

        pred = clf.predict(X_test_s)[0]
        y_true.append(y[test_idx][0])
        y_pred.append(pred)

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    accuracy  = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall    = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    logger.info("LOOCV → acc=%.2f  prec=%.2f  recall=%.2f  (TP=%d FP=%d FN=%d TN=%d)",
                accuracy, precision, recall, tp, fp, fn, tn)

    return {"accuracy": accuracy, "precision": precision, "recall": recall,
            "tp": tp, "fp": fp, "fn": fn, "tn": tn}


def _make_classifier() -> XGBClassifier:
    return XGBClassifier(
        n_estimators=50,
        max_depth=3,            # shallow — prevents overfitting on small data
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
        verbosity=0,
    )


# ── Main training entry point ─────────────────────────────────────────────────

def train() -> TrainingResult:
    """
    Full training pipeline:
      1. Fetch paired examples from QuestDB
      2. Build feature matrix
      3. LOOCV evaluation
      4. Retrain on all data
      5. Persist model + scaler to disk
    """
    logger.info("═" * 60)
    logger.info("Starting training pipeline")

    pairs = fetch_paired_examples()

    if len(pairs) < MIN_EXAMPLES_FOR_ML:
        logger.warning(
            "Only %d paired examples found (need %d for ML). "
            "System will use rule-based fallback.",
            len(pairs), MIN_EXAMPLES_FOR_ML
        )
        return TrainingResult(
            mode="rule_based",
            n_positive=len(pairs),
            n_negative=0,
            loocv_accuracy=None,
            loocv_precision=None,
            loocv_recall=None,
            feature_importances=None,
            model_path=None,
        )

    # Build dataset
    X, y = build_dataset(pairs)
    n_pos = int(y.sum())
    n_neg = int((y == 0).sum())

    # LOOCV evaluation
    logger.info("Running Leave-One-Out CV on %d samples...", len(y))
    cv_results = _loocv_evaluate(X, y)

    # Final model: train on ALL data
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)

    clf = _make_classifier()
    clf.fit(X_scaled, y)

    # Feature importances
    importances = dict(zip(FEATURE_NAMES, clf.feature_importances_.tolist()))
    importances_sorted = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

    logger.info("Top features:")
    for feat, score in list(importances_sorted.items())[:5]:
        logger.info("  %-35s %.4f", feat, score)

    # Persist model bundle
    model_path = Path(MODEL_PATH)
    model_path.parent.mkdir(parents=True, exist_ok=True)

    bundle = {
        "classifier": clf,
        "scaler":     scaler,
        "feature_names": FEATURE_NAMES,
        "n_training_examples": len(pairs),
        "cv_results": cv_results,
    }

    with open(model_path, "wb") as f:
        pickle.dump(bundle, f)

    logger.info("Model saved → %s", model_path)

    # Persist feature importances as JSON (useful for inspection / dashboards)
    imp_path = Path(FEATURE_IMPORTANCE_PATH)
    imp_path.parent.mkdir(parents=True, exist_ok=True)
    with open(imp_path, "w") as f:
        json.dump(importances_sorted, f, indent=2)

    logger.info("Feature importances saved → %s", imp_path)
    logger.info("Training complete.")
    logger.info("═" * 60)

    return TrainingResult(
        mode="ml",
        n_positive=n_pos,
        n_negative=n_neg,
        loocv_accuracy=cv_results["accuracy"],
        loocv_precision=cv_results["precision"],
        loocv_recall=cv_results["recall"],
        feature_importances=importances_sorted,
        model_path=str(model_path),
    )


# ── Model loader ──────────────────────────────────────────────────────────────

def load_model() -> Optional[dict]:
    """Load persisted model bundle.  Returns None if no model exists yet."""
    path = Path(MODEL_PATH)
    if not path.exists():
        logger.warning("No trained model found at %s — run with --train first", path)
        return None

    with open(path, "rb") as f:
        bundle = pickle.load(f)

    logger.info("Model loaded from %s  (trained on %d examples)",
                path, bundle.get("n_training_examples", "?"))
    return bundle
