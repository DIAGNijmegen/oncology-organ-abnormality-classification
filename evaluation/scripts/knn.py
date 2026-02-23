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
    get_feature_dir,
    get_metrics_output_path,
    load_features_and_labels,
    validate_evaluation_inputs,
    load_and_validate_annotations,
    load_subgroup_annotations,
    validate_features_and_labels,
    save_metrics,
    filter_by_subgroup,
    get_available_subgroups,
)


def main(args):
    fix_random_seeds(getattr(args, "seed", 0))

    feature_dir_training = get_feature_dir(
        args.output_root, args.model_name, args.organ_name, "training", args.aggregation_method
    )
    feature_dir_validation = get_feature_dir(
        args.output_root, args.model_name, args.organ_name, "validation", args.aggregation_method
    )
    feature_dir_test = get_feature_dir(
        args.output_root, args.model_name, args.organ_name, "test", args.aggregation_method
    )
    output_metrics = get_metrics_output_path(
        args.output_root, args.model_name, args.organ_name, args.aggregation_method, "knn"
    )
    
    # Validate inputs early
    validate_evaluation_inputs(
        feature_dir_training,
        feature_dir_validation,
        feature_dir_test,
        args.annotations_train_csv,
        args.annotations_test_csv,
        args.organ_name,
        output_metrics,
    )
    
    # Load annotations
    train_annotations, test_annotations = load_and_validate_annotations(
        args.annotations_train_csv,
        args.annotations_test_csv,
    )
    
    # Combine annotations for validation (validation uses train annotations)
    val_annotations = train_annotations
    
    # Load subgroup annotations
    train_subgroups, test_subgroups = load_subgroup_annotations(
        args.annotations_train_csv,
        args.annotations_test_csv,
    )
    val_subgroups = train_subgroups  # Validation uses train subgroups
    
    # Load features and labels for each split
    try:
        X_train, y_train, train_scan_ids = load_features_and_labels(
            feature_dir_training, train_annotations, args.organ_name, return_scan_ids=True
        )
        X_val, y_val, val_scan_ids = load_features_and_labels(
            feature_dir_validation, val_annotations, args.organ_name, return_scan_ids=True
        )
        X_test, y_test, test_scan_ids = load_features_and_labels(
            feature_dir_test, test_annotations, args.organ_name, return_scan_ids=True
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
    
    # Get available subgroups for train and test
    train_available_subgroups = get_available_subgroups(train_subgroups, args.organ_name)
    test_available_subgroups = get_available_subgroups(test_subgroups, args.organ_name)
    
    # Combine and deduplicate subgroups (evaluate on all subgroups present in either split)
    all_subgroups = sorted(set(train_available_subgroups + test_available_subgroups))
    
    k_values = [1, 3, 5, 10, 20, 30]
    results = []
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)
        
        # Evaluate on training set (overall)
        y_pred_train = knn.predict(X_train_scaled)
        y_prob_train = knn.predict_proba(X_train_scaled)[:, 1] if len(set(y_train)) == 2 else None
        acc_train = accuracy_score(y_train, y_pred_train)
        auc_train = roc_auc_score(y_train, y_prob_train) if y_prob_train is not None else None
        
        # Evaluate on validation set (overall)
        acc_val = None
        auc_val = None
        if len(X_val) > 0:
            y_pred_val = knn.predict(X_val_scaled)
            y_prob_val = knn.predict_proba(X_val_scaled)[:, 1] if len(set(y_train)) == 2 else None
            acc_val = accuracy_score(y_val, y_pred_val)
            auc_val = roc_auc_score(y_val, y_prob_val) if y_prob_val is not None else None
        
        # Evaluate on test set (overall)
        acc_test = None
        auc_test = None
        if len(X_test) > 0:
            y_pred_test = knn.predict(X_test_scaled)
            y_prob_test = knn.predict_proba(X_test_scaled)[:, 1] if len(set(y_train)) == 2 else None
            acc_test = accuracy_score(y_test, y_pred_test)
            auc_test = roc_auc_score(y_test, y_prob_test) if y_prob_test is not None else None
        
        # Initialize result dict for this k
        result = {
            "k": k,
            "overall": {
                "train": {"accuracy": float(acc_train), "auc": float(auc_train) if auc_train is not None else None},
                "validation": {"accuracy": float(acc_val) if acc_val is not None else None, "auc": float(auc_val) if auc_val is not None else None},
                "test": {"accuracy": float(acc_test) if acc_test is not None else None, "auc": float(auc_test) if auc_test is not None else None},
            },
            "subgroups": {},
        }
        
        # Evaluate on each subgroup
        for subgroup_name in all_subgroups:
            # Filter train split
            X_train_sub, y_train_sub = filter_by_subgroup(
                X_train, y_train, train_scan_ids, train_subgroups,
                args.organ_name, subgroup_name, subgroup_value=1
            )
            
            # Filter validation split
            X_val_sub, y_val_sub = filter_by_subgroup(
                X_val, y_val, val_scan_ids, val_subgroups,
                args.organ_name, subgroup_name, subgroup_value=1
            )
            
            # Filter test split
            X_test_sub, y_test_sub = filter_by_subgroup(
                X_test, y_test, test_scan_ids, test_subgroups,
                args.organ_name, subgroup_name, subgroup_value=1
            )
            
            # Only evaluate if we have samples
            if len(X_train_sub) > 0 or len(X_val_sub) > 0 or len(X_test_sub) > 0:
                # Scale filtered features
                X_train_sub_scaled = scaler.transform(X_train_sub) if len(X_train_sub) > 0 else X_train_sub
                X_val_sub_scaled = scaler.transform(X_val_sub) if len(X_val_sub) > 0 else X_val_sub
                X_test_sub_scaled = scaler.transform(X_test_sub) if len(X_test_sub) > 0 else X_test_sub
                
                # Evaluate on training set
                acc_train_sub = None
                auc_train_sub = None
                if len(X_train_sub) > 0:
                    y_pred_train_sub = knn.predict(X_train_sub_scaled)
                    y_prob_train_sub = knn.predict_proba(X_train_sub_scaled)[:, 1] if len(set(y_train)) == 2 else None
                    acc_train_sub = accuracy_score(y_train_sub, y_pred_train_sub)
                    auc_train_sub = roc_auc_score(y_train_sub, y_prob_train_sub) if y_prob_train_sub is not None else None
                
                # Evaluate on validation set
                acc_val_sub = None
                auc_val_sub = None
                if len(X_val_sub) > 0:
                    y_pred_val_sub = knn.predict(X_val_sub_scaled)
                    y_prob_val_sub = knn.predict_proba(X_val_sub_scaled)[:, 1] if len(set(y_train)) == 2 else None
                    acc_val_sub = accuracy_score(y_val_sub, y_pred_val_sub)
                    auc_val_sub = roc_auc_score(y_val_sub, y_prob_val_sub) if y_prob_val_sub is not None else None
                
                # Evaluate on test set
                acc_test_sub = None
                auc_test_sub = None
                if len(X_test_sub) > 0:
                    y_pred_test_sub = knn.predict(X_test_sub_scaled)
                    y_prob_test_sub = knn.predict_proba(X_test_sub_scaled)[:, 1] if len(set(y_train)) == 2 else None
                    acc_test_sub = accuracy_score(y_test_sub, y_pred_test_sub)
                    auc_test_sub = roc_auc_score(y_test_sub, y_prob_test_sub) if y_prob_test_sub is not None else None
                
                subgroup_metrics = {
                    "train": {
                        "accuracy": float(acc_train_sub) if acc_train_sub is not None else None,
                        "auc": float(auc_train_sub) if auc_train_sub is not None else None,
                        "n_samples": int(len(X_train_sub)),
                    },
                    "validation": {
                        "accuracy": float(acc_val_sub) if acc_val_sub is not None else None,
                        "auc": float(auc_val_sub) if auc_val_sub is not None else None,
                        "n_samples": int(len(X_val_sub)),
                    },
                    "test": {
                        "accuracy": float(acc_test_sub) if acc_test_sub is not None else None,
                        "auc": float(auc_test_sub) if auc_test_sub is not None else None,
                        "n_samples": int(len(X_test_sub)),
                    },
                }
            else:
                # No samples for this subgroup
                subgroup_metrics = {
                    "train": {"accuracy": None, "auc": None, "n_samples": 0},
                    "validation": {"accuracy": None, "auc": None, "n_samples": 0},
                    "test": {"accuracy": None, "auc": None, "n_samples": 0},
                }
            
            result["subgroups"][subgroup_name] = subgroup_metrics
        
        results.append(result)
    
    # Save results
    save_metrics(output_metrics, results)
    
    return 0


if __name__ == "__main__":
    parser = get_base_args_parser(description="kNN Evaluation")
    args = parser.parse_args()

    sys.exit(main(args))
