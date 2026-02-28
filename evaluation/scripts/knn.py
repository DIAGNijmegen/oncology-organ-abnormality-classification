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
    filter_normal_and_subgroup_abnormal,
    load_amos22_scan_ids,
    get_dataset_root_from_annotations_path,
    filter_by_scan_ids,
)


def run_knn_evaluation(
    X_train, y_train, train_scan_ids,
    X_val, y_val, val_scan_ids,
    X_test, y_test, test_scan_ids,
    train_subgroups, val_subgroups, test_subgroups,
    organ_name,
    k_values=[1, 3, 5, 10, 20, 30]
):
    """
    Run kNN evaluation on the provided data.
    
    Returns:
        List of results, one per k value
    """
    # Validate features and labels
    validate_features_and_labels(
        X_train, y_train, X_val, y_val, X_test, y_test, organ_name
    )
    
    # Standardize features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val) if len(X_val) > 0 else X_val
    X_test_scaled = scaler.transform(X_test) if len(X_test) > 0 else X_test
    
    results = []
    
    for k in k_values:
        knn = KNeighborsClassifier(n_neighbors=k)
        knn.fit(X_train_scaled, y_train)
        
        result = {"k": k, "evaluation_groups": {}}
        
        # Define evaluation groups: all, normal+diffuse, normal+focal
        evaluation_groups = [
            ("all", None),  # All samples
            ("normal_and_diffuse", "diffuse"),  # Normal + diffuse abnormal
            ("normal_and_focal", "focal"),  # Normal + focal abnormal
        ]
        
        for group_name, subgroup_name in evaluation_groups:
            group_metrics = {}
            
            # Filter data for this group
            if subgroup_name is None:
                # All samples - no filtering
                X_train_group, y_train_group = X_train, y_train
                X_val_group, y_val_group = X_val, y_val
                X_test_group, y_test_group = X_test, y_test
            else:
                # Normal + specific subgroup abnormal
                X_train_group, y_train_group = filter_normal_and_subgroup_abnormal(
                    X_train, y_train, train_scan_ids, train_subgroups,
                    organ_name, subgroup_name
                )
                X_val_group, y_val_group = filter_normal_and_subgroup_abnormal(
                    X_val, y_val, val_scan_ids, val_subgroups,
                    organ_name, subgroup_name
                )
                X_test_group, y_test_group = filter_normal_and_subgroup_abnormal(
                    X_test, y_test, test_scan_ids, test_subgroups,
                    organ_name, subgroup_name
                )
            
            # Evaluate on each split
            splits = [
                ("train", X_train_group, y_train_group, X_train_scaled, X_train),
                ("validation", X_val_group, y_val_group, X_val_scaled, X_val),
                ("test", X_test_group, y_test_group, X_test_scaled, X_test),
            ]
            
            for split_name, X_split, y_split, X_split_scaled_full, X_split_full in splits:
                if len(X_split) == 0:
                    group_metrics[split_name] = {
                        "accuracy": None,
                        "auc": None,
                        "n_normal": 0,
                        "n_abnormal": 0,
                    }
                    continue
                
                # Scale features: use pre-scaled if all samples, otherwise scale filtered
                if subgroup_name is None:
                    # All samples - use pre-scaled arrays
                    X_split_scaled = X_split_scaled_full
                else:
                    # Filtered samples - scale the filtered features
                    X_split_scaled = scaler.transform(X_split)
                
                # Predictions
                y_pred = knn.predict(X_split_scaled)
                y_prob = knn.predict_proba(X_split_scaled)[:, 1] if len(set(y_train)) == 2 else None
                
                # Metrics
                acc = accuracy_score(y_split, y_pred)
                auc = roc_auc_score(y_split, y_prob) if y_prob is not None else None
                
                # Count normal and abnormal samples
                n_normal = int(np.sum(y_split == 0))
                n_abnormal = int(np.sum(y_split == 1))
                
                group_metrics[split_name] = {
                    "accuracy": float(acc),
                    "auc": float(auc) if auc is not None else None,
                    "n_normal": n_normal,
                    "n_abnormal": n_abnormal,
                }
            
            result["evaluation_groups"][group_name] = group_metrics
        
        results.append(result)
    
    return results


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
    
    # Get dataset root and load AMOS22 scan IDs
    dataset_root = get_dataset_root_from_annotations_path(args.annotations_train_csv)
    amos22_scan_ids = load_amos22_scan_ids(dataset_root)
    
    # Run evaluation on all data
    all_results = run_knn_evaluation(
        X_train, y_train, train_scan_ids,
        X_val, y_val, val_scan_ids,
        X_test, y_test, test_scan_ids,
        train_subgroups, val_subgroups, test_subgroups,
        args.organ_name,
    )
    
    # Filter out AMOS22 scans and run evaluation again
    X_train_filtered, y_train_filtered, train_scan_ids_filtered = filter_by_scan_ids(
        X_train, y_train, train_scan_ids, amos22_scan_ids
    )
    X_val_filtered, y_val_filtered, val_scan_ids_filtered = filter_by_scan_ids(
        X_val, y_val, val_scan_ids, amos22_scan_ids
    )
    X_test_filtered, y_test_filtered, test_scan_ids_filtered = filter_by_scan_ids(
        X_test, y_test, test_scan_ids, amos22_scan_ids
    )
    
    exclude_amos22_results = run_knn_evaluation(
        X_train_filtered, y_train_filtered, train_scan_ids_filtered,
        X_val_filtered, y_val_filtered, val_scan_ids_filtered,
        X_test_filtered, y_test_filtered, test_scan_ids_filtered,
        train_subgroups, val_subgroups, test_subgroups,
        args.organ_name,
    )
    
    # Structure output with two top-level objects
    metrics = {
        "all_data": all_results,
        "exclude_amos22": exclude_amos22_results,
    }
    
    # Save results
    save_metrics(output_metrics, metrics)
    
    return 0


if __name__ == "__main__":
    parser = get_base_args_parser(description="kNN Evaluation")
    args = parser.parse_args()

    sys.exit(main(args))
