# Copyright Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
# Licensed under Apache-2.0

import os
import glob
from typing import Tuple, Optional
import numpy as np
from util.leavs_utils import parse_train_annotations, parse_test_annotations


def get_base_args_parser(description: Optional[str] = None, add_help: bool = True):
    """Get base argument parser with common arguments for evaluation scripts."""
    import argparse
    parser = argparse.ArgumentParser(description=description, add_help=add_help)
    parser.add_argument(
        "--feature-dir-training",
        type=str,
        required=True,
        help="Directory containing training feature files",
    )
    parser.add_argument(
        "--feature-dir-validation",
        type=str,
        required=True,
        help="Directory containing validation feature files",
    )
    parser.add_argument(
        "--feature-dir-test",
        type=str,
        required=True,
        help="Directory containing test feature files",
    )
    parser.add_argument(
        "--annotations-train-csv",
        type=str,
        required=True,
        help="Path to training annotations CSV",
    )
    parser.add_argument(
        "--annotations-test-csv",
        type=str,
        required=True,
        help="Path to test annotations CSV",
    )
    parser.add_argument(
        "--organ-name",
        type=str,
        required=True,
        help="Organ name",
    )
    parser.add_argument(
        "--output-metrics",
        type=str,
        required=True,
        help="Output file to write metrics",
    )
    return parser


def load_features_and_labels(
    feature_dir: str, 
    annotations: dict, 
    organ_name: str,
    return_scan_ids: bool = False
) -> Tuple[np.ndarray, np.ndarray, Optional[list]]:
    """
    Load features and labels from feature directory.
    Features are aggregated but may have any shape - they will be flattened for evaluation.
    
    Args:
        feature_dir: Directory containing .npz feature files
        annotations: Dictionary mapping scan_id to organ labels
        organ_name: Name of the organ
        return_scan_ids: If True, also return list of scan IDs
    
    Returns:
        (features, labels) or (features, labels, scan_ids) if return_scan_ids=True
        Features are flattened to 1D arrays for each sample
    """
    feature_files = glob.glob(os.path.join(feature_dir, "*.npz"))
    
    features_list = []
    labels_list = []
    scan_ids_list = []
    
    for feature_file in sorted(feature_files):
        scan_id = os.path.basename(feature_file).replace(".npz", "")
        
        # Get label from annotations
        if scan_id in annotations and organ_name in annotations[scan_id]:
            label = annotations[scan_id][organ_name]
            if label in [0, 1]:  # Only include valid labels
                data = np.load(feature_file)
                
                # Skip placeholder files
                is_placeholder = data.get("is_placeholder", False)
                if is_placeholder:
                    continue
                
                feature = data["features"]
                
                # Skip empty features
                if feature.size == 0:
                    continue
                
                # Flatten aggregated features to 1D for evaluation
                # Aggregated features can have any shape (e.g., from feature maps)
                feature_flat = feature.flatten()
                
                features_list.append(feature_flat)
                labels_list.append(label)
                if return_scan_ids:
                    scan_ids_list.append(scan_id)
    
    if len(features_list) == 0:
        if return_scan_ids:
            return np.array([]), np.array([]), []
        return np.array([]), np.array([])
    
    if return_scan_ids:
        return np.array(features_list), np.array(labels_list), scan_ids_list
    return np.array(features_list), np.array(labels_list)


def validate_evaluation_inputs(
    feature_dir_training: str,
    feature_dir_validation: str,
    feature_dir_test: str,
    annotations_train_csv: str,
    annotations_test_csv: str,
    organ_name: str,
    output_metrics: str,
    output_checkpoint: Optional[str] = None,
):
    """
    Validate all inputs for evaluation scripts.
    Raises exceptions if validation fails.
    """
    # Validate feature directories
    if not os.path.isdir(feature_dir_training):
        raise NotADirectoryError(f"Training feature directory does not exist: {feature_dir_training}")
    if not os.path.isdir(feature_dir_validation):
        raise NotADirectoryError(f"Validation feature directory does not exist: {feature_dir_validation}")
    if not os.path.isdir(feature_dir_test):
        raise NotADirectoryError(f"Test feature directory does not exist: {feature_dir_test}")
    
    # Validate annotation CSV files
    if not os.path.exists(annotations_train_csv):
        raise FileNotFoundError(f"Training annotations CSV not found: {annotations_train_csv}")
    if not os.path.isfile(annotations_train_csv):
        raise ValueError(f"Training annotations path is not a file: {annotations_train_csv}")
    if not os.access(annotations_train_csv, os.R_OK):
        raise PermissionError(f"Cannot read training annotations CSV: {annotations_train_csv}")
    
    if not os.path.exists(annotations_test_csv):
        raise FileNotFoundError(f"Test annotations CSV not found: {annotations_test_csv}")
    if not os.path.isfile(annotations_test_csv):
        raise ValueError(f"Test annotations path is not a file: {annotations_test_csv}")
    if not os.access(annotations_test_csv, os.R_OK):
        raise PermissionError(f"Cannot read test annotations CSV: {annotations_test_csv}")
    
    # Validate organ name
    if not organ_name or not organ_name.strip():
        raise ValueError("Organ name cannot be empty")
    
    # Ensure output directory can be created
    output_dir = os.path.dirname(output_metrics)
    if output_dir:
        try:
            os.makedirs(output_dir, exist_ok=True)
        except OSError as e:
            raise OSError(f"Cannot create output directory {output_dir}: {e}")
    
    # Check write permissions for output
    if os.path.exists(output_metrics) and not os.access(output_metrics, os.W_OK):
        raise PermissionError(f"Cannot write to output file: {output_metrics}")
    
    # Validate checkpoint directory if provided
    if output_checkpoint is not None:
        try:
            os.makedirs(output_checkpoint, exist_ok=True)
        except OSError as e:
            raise OSError(f"Cannot create checkpoint directory {output_checkpoint}: {e}")
        if not os.access(output_checkpoint, os.W_OK):
            raise PermissionError(f"Cannot write to checkpoint directory: {output_checkpoint}")


def load_and_validate_annotations(
    annotations_train_csv: str,
    annotations_test_csv: str,
) -> Tuple[dict, dict]:
    """
    Load and validate annotations from CSV files.
    Returns: (train_annotations, test_annotations)
    """
    try:
        train_annotations = parse_train_annotations(annotations_train_csv)
        test_annotations = parse_test_annotations(annotations_test_csv)
    except Exception as e:
        raise RuntimeError(f"Failed to parse annotations: {e}") from e
    
    return train_annotations, test_annotations


def validate_features_and_labels(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    organ_name: str,
):
    """
    Validate feature consistency and labels across all splits.
    Raises exceptions if validation fails.
    """
    if len(X_train) == 0:
        raise ValueError(f"No training samples found for organ {organ_name}. Cannot perform evaluation.")
    
    # Validate feature consistency
    if len(X_train.shape) != 2:
        raise ValueError(f"Training features must be 2D array (n_samples, n_features), got shape {X_train.shape}")
    if len(y_train) != len(X_train):
        raise ValueError(f"Mismatch between training features ({len(X_train)}) and labels ({len(y_train)})")
    
    if len(X_val) > 0:
        if X_val.shape[1] != X_train.shape[1]:
            raise ValueError(f"Feature dimension mismatch: training has {X_train.shape[1]} features, validation has {X_val.shape[1]}")
        if len(y_val) != len(X_val):
            raise ValueError(f"Mismatch between validation features ({len(X_val)}) and labels ({len(y_val)})")
    
    if len(X_test) > 0:
        if X_test.shape[1] != X_train.shape[1]:
            raise ValueError(f"Feature dimension mismatch: training has {X_train.shape[1]} features, test has {X_test.shape[1]}")
        if len(y_test) != len(X_test):
            raise ValueError(f"Mismatch between test features ({len(X_test)}) and labels ({len(y_test)})")
    
    # Validate labels
    unique_labels = set(y_train)
    if len(unique_labels) < 2:
        raise ValueError(f"Need at least 2 classes for evaluation, found {len(unique_labels)}: {unique_labels}")
    if not unique_labels.issubset({0, 1}):
        raise ValueError(f"Labels must be 0 or 1, found: {unique_labels}")


def save_metrics(output_path: str, metrics: dict):
    """
    Save metrics to JSON file with validation.
    """
    import json
    try:
        with open(output_path, "w") as f:
            json.dump(metrics, f, indent=2)
    except Exception as e:
        raise RuntimeError(f"Failed to write output metrics to {output_path}: {e}") from e
    
    # Verify output was created
    if not os.path.exists(output_path):
        raise RuntimeError(f"Output metrics file was not created: {output_path}")
