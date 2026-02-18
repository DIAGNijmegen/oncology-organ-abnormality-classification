# Copyright Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
# Licensed under Apache-2.0

import os
import sys

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, roc_auc_score

from util.util import fix_random_seeds
from .evaluation_utils import (
    get_base_args_parser,
    load_features_and_labels,
    validate_evaluation_inputs,
    load_and_validate_annotations,
    validate_features_and_labels,
    save_metrics,
)


def main(args):
    fix_random_seeds(getattr(args, "seed", 0))
    
    # Validate inputs early
    validate_evaluation_inputs(
        args.feature_dir_training,
        args.feature_dir_validation,
        args.feature_dir_test,
        args.annotations_train_csv,
        args.annotations_test_csv,
        args.organ_name,
        args.output_metrics,
    )
    
    # Load annotations
    train_annotations, test_annotations = load_and_validate_annotations(
        args.annotations_train_csv,
        args.annotations_test_csv,
    )
    
    # Combine annotations for validation (validation uses train annotations)
    val_annotations = train_annotations
    
    # Load features and labels for each split
    try:
        X_train, y_train, train_scan_ids = load_features_and_labels(
            args.feature_dir_training, train_annotations, args.organ_name, return_scan_ids=True
        )
        X_val, y_val, val_scan_ids = load_features_and_labels(
            args.feature_dir_validation, val_annotations, args.organ_name, return_scan_ids=True
        )
        X_test, y_test, test_scan_ids = load_features_and_labels(
            args.feature_dir_test, test_annotations, args.organ_name, return_scan_ids=True
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load features: {e}") from e
    
    # Validate features and labels
    validate_features_and_labels(
        X_train, y_train, X_val, y_val, X_test, y_test, args.organ_name
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val) if len(X_val) > 0 else X_val
    X_test_scaled = scaler.transform(X_test) if len(X_test) > 0 else X_test
    
    k_values = [10, 20, 100, 200]
    results = []
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)
        
        # Evaluate on training set
        y_pred_train = knn.predict(X_train_scaled)
        y_prob_train = knn.predict_proba(X_train_scaled)[:, 1] if len(set(y_train)) == 2 else None
        acc_train = accuracy_score(y_train, y_pred_train)
        auc_train = roc_auc_score(y_train, y_prob_train) if y_prob_train is not None else None
        
        # Evaluate on validation set
        acc_val = None
        auc_val = None
        if len(X_val) > 0:
            y_pred_val = knn.predict(X_val_scaled)
            y_prob_val = knn.predict_proba(X_val_scaled)[:, 1] if len(set(y_train)) == 2 else None
            acc_val = accuracy_score(y_val, y_pred_val)
            auc_val = roc_auc_score(y_val, y_prob_val) if y_prob_val is not None else None
        
        # Evaluate on test set
        acc_test = None
        auc_test = None
        if len(X_test) > 0:
            y_pred_test = knn.predict(X_test_scaled)
            y_prob_test = knn.predict_proba(X_test_scaled)[:, 1] if len(set(y_train)) == 2 else None
            acc_test = accuracy_score(y_test, y_pred_test)
            auc_test = roc_auc_score(y_test, y_prob_test) if y_prob_test is not None else None
        
        results.append({
            "k": k,
            "train": {"accuracy": float(acc_train), "auc": float(auc_train) if auc_train is not None else None},
            "validation": {"accuracy": float(acc_val) if acc_val is not None else None, "auc": float(auc_val) if auc_val is not None else None},
            "test": {"accuracy": float(acc_test) if acc_test is not None else None, "auc": float(auc_test) if auc_test is not None else None},
        })
    
    # Save results
    save_metrics(args.output_metrics, results)
    
    return 0


if __name__ == "__main__":
    parser = get_base_args_parser(description="kNN Evaluation")
    args = parser.parse_args()

    sys.exit(main(args))
