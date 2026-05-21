"""
train_model.py  (landmark model)
─────────────────────────────────
Train a hand-sign classifier on landmark data collected by collect_data.py.

Usage:
    python train_model.py --team 1
    python train_model.py --team 2 --model mlp

Outputs (saved to Teams/TeamN/models/):
    hand_sign_classifier.pkl
    label_encoder.pkl
"""

from __future__ import annotations

import argparse
import csv
import os
import pathlib
import sys

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder

SCRIPT_DIR     = pathlib.Path(__file__).parent
TEAMS_ROOT_DIR = SCRIPT_DIR.parent.parent.parent / "Teams"


def load_data(path: pathlib.Path) -> tuple[np.ndarray, np.ndarray]:
    """Read a hand_sign_data.csv file; return (features, labels) numpy arrays."""
    labels, features = [], []
    with open(path, newline="") as f:
        for row in csv.reader(f):
            if not row:
                continue
            labels.append(row[0])
            features.append([float(x) for x in row[1:]])
    return np.array(features), np.array(labels)


def build_model(name: str):
    """Construct an sklearn classifier by name: 'rf' (RandomForest) or 'mlp'."""
    if name == "rf":
        return RandomForestClassifier(
            n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
    if name == "mlp":
        return MLPClassifier(
            hidden_layer_sizes=(128, 64), activation="relu",
            max_iter=500, early_stopping=True, random_state=42)
    raise ValueError(f"Unknown model: {name}")


def main() -> None:
    """Train and evaluate a landmark classifier; save model + label encoder."""
    parser = argparse.ArgumentParser(description="Train landmark hand-sign classifier")
    parser.add_argument("--team",  type=int, required=True, choices=range(1, 7))
    parser.add_argument("--model", choices=["rf", "mlp"], default="rf",
                        help="rf=RandomForest (default), mlp=MLP neural net")
    parser.add_argument("--test-size", type=float, default=0.2)
    args = parser.parse_args()

    data_dir     = SCRIPT_DIR / "teams" / f"Team{args.team}"
    data_file    = data_dir / "hand_sign_data.csv"

    team_models  = TEAMS_ROOT_DIR / f"Team{args.team}" / "models"
    team_models.mkdir(parents=True, exist_ok=True)
    model_file   = team_models / "hand_sign_classifier.pkl"
    encoder_file = team_models / "label_encoder.pkl"

    print("=" * 50)
    print(f"  LANDMARK CLASSIFIER — Team {args.team}")
    print("=" * 50)

    if not data_file.exists():
        print(f"\n[ERROR] Data file not found: {data_file}")
        print("  Run collect_data.py --team", args.team, "first.")
        sys.exit(1)

    print(f"\n[1/6] Loading data from {data_file} ...")
    X, y_raw = load_data(data_file)
    print(f"  {len(X)} samples, {len(set(y_raw))} classes, {X.shape[1]} features")

    print("\n[2/6] Class distribution ...")
    unique, counts = np.unique(y_raw, return_counts=True)
    for cls, cnt in zip(unique, counts):
        print(f"  {cls:>10s}: {cnt:>5d}  {'#' * (cnt // 10)}")
    if counts.min() < 50:
        print(f"  WARNING: '{unique[counts.argmin()]}' has only {counts.min()} samples.")

    print("\n[3/6] Encoding labels ...")
    le = LabelEncoder()
    y  = le.fit_transform(y_raw)
    print(f"  Classes: {list(le.classes_)}")

    print(f"\n[4/6] Splitting (test_size={args.test_size}) ...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, random_state=42, stratify=y)
    print(f"  Train: {len(X_train)}   Test: {len(X_test)}")

    model = build_model(args.model)
    print(f"\n[5/6] Training {type(model).__name__} ...")
    model.fit(X_train, y_train)

    print("\n[6/6] Evaluating ...")
    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    print(f"  Accuracy: {acc:.1%}\n")
    print(classification_report(y_test, y_pred, target_names=le.classes_))

    joblib.dump(model, model_file)
    joblib.dump(le,    encoder_file)
    print(f"  Model   → {model_file}")
    print(f"  Encoder → {encoder_file}")
    print("\nDone. Run inference.py --team", args.team, "to use the model.")


if __name__ == "__main__":
    main()
