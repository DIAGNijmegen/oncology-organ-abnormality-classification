# Copyright Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
# Licensed under Apache-2.0

import os
import glob
from typing import Tuple, Optional, Dict, List
import numpy as np
from util.leavs_utils import (
    parse_train_subgroup_annotations,
    parse_test_subgroup_annotations,
    infer_labels_from_subgroups,
)
from util.snakemake_helpers import VALID_ORGANS


def get_base_args_parser(description: Optional[str] = None, add_help: bool = True):
    """Get base argument parser with common arguments for evaluation scripts."""
    import argparse
    parser = argparse.ArgumentParser(description=description, add_help=add_help)
    parser.add_argument("--output-root", type=str, required=True, help="Workflow output root directory")
    parser.add_argument("--model-name", type=str, required=True, help="Feature model name")
    parser.add_argument("--aggregation-method", type=str, required=True, choices=["mean", "max", "std", "median", "meanstd", "vitg"], help="Aggregation method (use 'vitg' for spectrevitg model)")
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
    return parser


def get_feature_dir(output_root: str, model_name: str, organ_name: str, split: str, aggregation_method: str) -> str:
    """
    Get feature directory path. For spectrevitg, returns raw feature directory
    since features are already aggregated by the model.
    """
    # spectrevitg writes already-aggregated features to raw path
    if model_name == "spectrevitg":
        return get_raw_feature_dir(output_root, model_name, organ_name, split)
    
    return os.path.join(
        output_root,
        model_name,
        organ_name,
        split,
        "features",
        "aggregated",
        aggregation_method,
    )


def get_raw_feature_dir(output_root: str, model_name: str, organ_name: str, split: str) -> str:
    """Get directory path for raw (non-aggregated) features."""
    return os.path.join(
        output_root,
        model_name,
        organ_name,
        split,
        "features",
        "raw",
    )


def get_metrics_output_path(output_root: str, model_name: str, organ_name: str, aggregation_method: str, evaluation_mode: str) -> str:
    return os.path.join(
        output_root,
        model_name,
        organ_name,
        "metrics",
        "aggregated",
        aggregation_method,
        f"{evaluation_mode}.json",
    )


def get_attention_metrics_output_path(output_root: str, model_name: str, organ_name: str) -> str:
    """Get metrics output path for attention evaluation (uses raw features, no aggregation)."""
    return os.path.join(
        output_root,
        model_name,
        organ_name,
        "metrics",
        "attention",
        "attention.json",
    )


def get_attention_checkpoint_output_dir(output_root: str, model_name: str, organ_name: str) -> str:
    """Get checkpoint output directory for attention evaluation."""
    return os.path.join(
        output_root,
        model_name,
        organ_name,
        "checkpoints",
        "attention",
    )


def get_checkpoint_output_dir(output_root: str, model_name: str, organ_name: str, aggregation_method: str) -> str:
    return os.path.join(
        output_root,
        model_name,
        organ_name,
        "checkpoints",
        "aggregated",
        aggregation_method,
    )


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


def load_raw_features_and_labels(
    feature_dir: str, 
    annotations: dict, 
    organ_name: str,
    return_scan_ids: bool = False
) -> Tuple[List[np.ndarray], np.ndarray, Optional[list]]:
    """
    Load raw patch features and labels from feature directory.
    Returns patch features as a list of arrays (one per sample) since each sample
    can have a different number of patches.
    
    Args:
        feature_dir: Directory containing .npz feature files with raw patch features
        annotations: Dictionary mapping scan_id to organ labels
        organ_name: Name of the organ
        return_scan_ids: If True, also return list of scan IDs
    
    Returns:
        (features_list, labels, scan_ids) where features_list is a list of arrays
        Each array has shape (n_patches, flattened_dim) where each patch is flattened to 1D
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
                
                patch_features = data["features"]  # Shape: (n_patches, ...) - patches can have any shape
                
                # Skip empty features
                if patch_features.size == 0 or len(patch_features) == 0:
                    continue
                
                # Flatten each patch to 1D vector
                # Handle different input shapes:
                if len(patch_features.shape) == 1:
                    # Single patch case - flatten and reshape to (1, flattened_dim)
                    patch_features_flat = patch_features.flatten()
                    patch_features = patch_features_flat.reshape(1, -1)
                else:
                    # Multi-dimensional: (n_patches, ...) - flatten each patch individually
                    n_patches = patch_features.shape[0]
                    flattened_patches = []
                    for i in range(n_patches):
                        # Flatten each patch to 1D vector
                        flattened_patch = patch_features[i].flatten()
                        flattened_patches.append(flattened_patch)
                    patch_features = np.array(flattened_patches)  # Shape: (n_patches, flattened_dim)
                
                features_list.append(patch_features)
                labels_list.append(label)
                if return_scan_ids:
                    scan_ids_list.append(scan_id)
    
    if len(features_list) == 0:
        if return_scan_ids:
            return [], np.array([]), []
        return [], np.array([])
    
    if return_scan_ids:
        return features_list, np.array(labels_list), scan_ids_list
    return features_list, np.array(labels_list)


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
    
    Annotations are inferred from subgroup annotations:
    - If any subgroup has value 1, then label = 1 (abnormal)
    - Otherwise, label = 0 (normal)
    """
    try:
        train_subgroups = parse_train_subgroup_annotations(annotations_train_csv)
        test_subgroups = parse_test_subgroup_annotations(annotations_test_csv)
        train_annotations = infer_labels_from_subgroups(train_subgroups)
        test_annotations = infer_labels_from_subgroups(test_subgroups)
    except Exception as e:
        raise RuntimeError(f"Failed to parse annotations: {e}") from e
    
    return train_annotations, test_annotations


def load_subgroup_annotations(
    annotations_train_csv: str,
    annotations_test_csv: str,
) -> Tuple[dict, dict]:
    """
    Load subgroup annotations from CSV files.
    Returns: (train_subgroup_annotations, test_subgroup_annotations)
    Format: {scan_id: {organ: {subgroup_name: value}}}
    """
    try:
        train_subgroups = parse_train_subgroup_annotations(annotations_train_csv)
        test_subgroups = parse_test_subgroup_annotations(annotations_test_csv)
    except Exception as e:
        raise RuntimeError(f"Failed to parse subgroup annotations: {e}") from e
    
    return train_subgroups, test_subgroups


def filter_normal_and_subgroup_abnormal(
    X: np.ndarray,
    y: np.ndarray,
    scan_ids: List[str],
    subgroup_annotations: Dict[str, Dict[str, Dict[str, int]]],
    organ_name: str,
    subgroup_name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Filter features and labels to include:
    - All normal samples (label=0)
    - All abnormal samples (label=1) with the specified subgroup value=1
    
    Args:
        X: Feature array
        y: Label array
        scan_ids: List of scan IDs corresponding to X and y
        subgroup_annotations: Subgroup annotations dict
        organ_name: Name of the organ
        subgroup_name: Name of the subgroup (e.g., 'diffuse', 'focal')
    
    Returns:
        Filtered (X_filtered, y_filtered)
    """
    if len(X) == 0:
        return X, y
    
    if len(scan_ids) != len(X):
        raise ValueError(f"Mismatch: {len(scan_ids)} scan_ids but {len(X)} samples")
    
    filtered_indices = []
    for idx, scan_id in enumerate(scan_ids):
        # Include all normal samples (label=0)
        if y[idx] == 0:
            filtered_indices.append(idx)
        # Include abnormal samples (label=1) with the specified subgroup
        elif y[idx] == 1:
            if scan_id in subgroup_annotations:
                organ_subgroups = subgroup_annotations[scan_id].get(organ_name, {})
                # Check if this abnormal sample has the specified subgroup value=1
                if subgroup_name in organ_subgroups and organ_subgroups[subgroup_name] == 1:
                    filtered_indices.append(idx)
    
    if len(filtered_indices) == 0:
        return np.array([]), np.array([])
    
    filtered_indices = np.array(filtered_indices)
    return X[filtered_indices], y[filtered_indices]


def get_available_subgroups(
    subgroup_annotations: Dict[str, Dict[str, Dict[str, int]]],
    organ_name: str,
) -> List[str]:
    """
    Get list of available subgroups for a given organ.
    
    Args:
        subgroup_annotations: Subgroup annotations dict
        organ_name: Name of the organ
    
    Returns:
        List of subgroup names that have at least one sample with value=1
    """
    available_subgroups = set()
    
    for scan_id, organs in subgroup_annotations.items():
        if organ_name in organs:
            for subgroup_name, value in organs[organ_name].items():
                if value == 1:
                    available_subgroups.add(subgroup_name)
    
    return sorted(list(available_subgroups))


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


def load_amos22_scan_ids(dataset_root: str) -> set:
    """
    Load AMOS22 scan IDs from the amos22-list.txt file.
    
    Args:
        dataset_root: Root directory of the dataset
    
    Returns:
        Set of scan IDs (e.g., {'amos_0001', 'amos_0002', ...})
    """
    amos22_file = os.path.join(dataset_root, "LEAVS", "amos22-list.txt")
    
    if not os.path.exists(amos22_file):
        raise FileNotFoundError(f"AMOS22 list file not found: {amos22_file}")
    
    scan_ids = set()
    with open(amos22_file, "r") as f:
        for line in f:
            scan_id = line.strip()
            if scan_id:
                scan_ids.add(scan_id)
    
    return scan_ids


def get_dataset_root_from_annotations_path(annotations_path: str) -> str:
    """
    Extract dataset root from annotations CSV path.
    Assumes path format: {dataset_root}/LEAVS/amos_*_annotations.csv
    
    Args:
        annotations_path: Path to annotations CSV file
    
    Returns:
        Dataset root directory
    """
    # Remove /LEAVS/amos_*_annotations.csv to get dataset root
    path = os.path.dirname(annotations_path)  # Remove filename
    if path.endswith("/LEAVS"):
        dataset_root = os.path.dirname(path)
    else:
        # Fallback: try to find LEAVS in path
        parts = path.split("/LEAVS")
        if len(parts) > 1:
            dataset_root = parts[0]
        else:
            raise ValueError(f"Cannot extract dataset root from path: {annotations_path}")
    return dataset_root


def filter_by_scan_ids(
    X: np.ndarray,
    y: np.ndarray,
    scan_ids: List[str],
    exclude_scan_ids: set,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Filter features, labels, and scan IDs by excluding specified scan IDs.
    
    Args:
        X: Feature array
        y: Label array
        scan_ids: List of scan IDs corresponding to X and y
        exclude_scan_ids: Set of scan IDs to exclude
    
    Returns:
        Filtered (X_filtered, y_filtered, scan_ids_filtered)
    """
    if len(X) == 0:
        return X, y, scan_ids
    
    if len(scan_ids) != len(X):
        raise ValueError(f"Mismatch: {len(scan_ids)} scan_ids but {len(X)} samples")
    
    filtered_indices = []
    for idx, scan_id in enumerate(scan_ids):
        if scan_id not in exclude_scan_ids:
            filtered_indices.append(idx)
    
    if len(filtered_indices) == 0:
        return np.array([]), np.array([]), []
    
    filtered_indices = np.array(filtered_indices)
    return X[filtered_indices], y[filtered_indices], [scan_ids[i] for i in filtered_indices]


def filter_patch_features_by_scan_ids(
    patch_features_list: List[np.ndarray],
    y: np.ndarray,
    scan_ids: List[str],
    exclude_scan_ids: set,
) -> Tuple[List[np.ndarray], np.ndarray, List[str]]:
    """
    Filter patch features, labels, and scan IDs by excluding specified scan IDs.
    
    Args:
        patch_features_list: List of patch feature arrays
        y: Label array
        scan_ids: List of scan IDs corresponding to patch_features_list and y
        exclude_scan_ids: Set of scan IDs to exclude
    
    Returns:
        Filtered (patch_features_filtered, y_filtered, scan_ids_filtered)
    """
    if len(patch_features_list) == 0:
        return patch_features_list, y, scan_ids
    
    if len(scan_ids) != len(patch_features_list):
        raise ValueError(f"Mismatch: {len(scan_ids)} scan_ids but {len(patch_features_list)} samples")
    
    filtered_features = []
    filtered_labels = []
    filtered_scan_ids = []
    
    for idx, scan_id in enumerate(scan_ids):
        if scan_id not in exclude_scan_ids:
            filtered_features.append(patch_features_list[idx])
            filtered_labels.append(y[idx])
            filtered_scan_ids.append(scan_id)
    
    return filtered_features, np.array(filtered_labels), filtered_scan_ids


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


def get_predictions_output_path(output_path: str) -> str:
    """
    Get predictions output path from metrics output path.
    Example: linear.json -> linear_predictions.json
    """
    base_path = output_path.replace(".json", "")
    return f"{base_path}_predictions.json"


def get_subgroup_info(
    scan_id: str,
    subgroup_annotations: Dict[str, Dict[str, Dict[str, int]]],
    organ_name: str,
) -> Tuple[Optional[bool], Optional[bool]]:
    """
    Get focal and diffuse information for a scan ID and organ.
    
    Args:
        scan_id: Scan ID
        subgroup_annotations: Subgroup annotations dict
        organ_name: Name of the organ
    
    Returns:
        (is_focal, is_diffuse) - booleans or None if not available
    """
    if scan_id not in subgroup_annotations:
        return None, None
    
    organ_subgroups = subgroup_annotations[scan_id].get(organ_name, {})
    is_focal = bool(organ_subgroups.get("focal", 0)) if "focal" in organ_subgroups else None
    is_diffuse = bool(organ_subgroups.get("diffuse", 0)) if "diffuse" in organ_subgroups else None
    
    return is_focal, is_diffuse


def save_predictions(output_path: str, predictions: dict):
    """
    Save predictions to JSON file with validation.
    
    Args:
        output_path: Path to save predictions JSON file
        predictions: Dictionary with predictions structure:
            {
                "all_data": {
                    "evaluation_groups": {
                        "all": {
                            "train": [{"scan_id": ..., "ground_truth": ..., "is_focal": ..., "is_diffuse": ..., "probability": ...}, ...],
                            "validation": [...],
                            "test": [...]
                        },
                        ...
                    }
                },
                "exclude_amos22": {...}
            }
    """
    import json
    try:
        with open(output_path, "w") as f:
            json.dump(predictions, f, indent=2)
    except Exception as e:
        raise RuntimeError(f"Failed to write predictions to {output_path}: {e}") from e
    
    # Verify output was created
    if not os.path.exists(output_path):
        raise RuntimeError(f"Predictions file was not created: {output_path}")


def load_features_and_labels_all_organs(
    output_root: str,
    model_name: str,
    split: str,
    aggregation_method: str,
    annotations: dict,
    return_scan_ids: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[list]]:
    """
    Load features and labels from all organs, treating each organ-scan combination as a sample.
    
    Args:
        output_root: Root directory for outputs
        model_name: Name of the feature model
        split: Split name (training, validation, test)
        aggregation_method: Aggregation method name
        annotations: Dictionary mapping scan_id to organ labels {scan_id: {organ_name: label}}
        return_scan_ids: If True, also return list of (scan_id, organ_name) tuples
    
    Returns:
        (features, labels) or (features, labels, scan_organ_ids) if return_scan_ids=True
        scan_organ_ids is a list of tuples: [(scan_id, organ_name), ...]
    """
    features_list = []
    labels_list = []
    scan_organ_ids_list = []
    
    for organ_name in VALID_ORGANS:
        feature_dir = get_feature_dir(output_root, model_name, organ_name, split, aggregation_method)
        
        if not os.path.isdir(feature_dir):
            # Skip if feature directory doesn't exist for this organ
            continue
        
        feature_files = glob.glob(os.path.join(feature_dir, "*.npz"))
        
        for feature_file in sorted(feature_files):
            scan_id = os.path.basename(feature_file).replace(".npz", "")
            
            # Get label from annotations for this scan_id and organ
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
                    feature_flat = feature.flatten()
                    
                    features_list.append(feature_flat)
                    labels_list.append(label)
                    if return_scan_ids:
                        scan_organ_ids_list.append((scan_id, organ_name))
    
    if len(features_list) == 0:
        if return_scan_ids:
            return np.array([]), np.array([]), []
        return np.array([]), np.array([])
    
    if return_scan_ids:
        return np.array(features_list), np.array(labels_list), scan_organ_ids_list
    return np.array(features_list), np.array(labels_list)


def get_all_organs_metrics_output_path(output_root: str, model_name: str, aggregation_method: str, evaluation_mode: str) -> str:
    """Get metrics output path for 'all' organs evaluation."""
    return os.path.join(
        output_root,
        model_name,
        "all",
        "metrics",
        "aggregated",
        aggregation_method,
        f"{evaluation_mode}.json",
    )


def get_all_organs_checkpoint_output_dir(output_root: str, model_name: str, aggregation_method: str) -> str:
    """Get checkpoint output directory for 'all' organs evaluation."""
    return os.path.join(
        output_root,
        model_name,
        "all",
        "checkpoints",
        "aggregated",
        aggregation_method,
    )


def get_all_organs_attention_metrics_output_path(output_root: str, model_name: str) -> str:
    """Get metrics output path for 'all' organs attention evaluation."""
    return os.path.join(
        output_root,
        model_name,
        "all",
        "metrics",
        "attention",
        "attention.json",
    )


def get_all_organs_attention_checkpoint_output_dir(output_root: str, model_name: str) -> str:
    """Get checkpoint output directory for 'all' organs attention evaluation."""
    return os.path.join(
        output_root,
        model_name,
        "all",
        "checkpoints",
        "attention",
    )


def filter_all_organs_with_scan_ids(
    X: np.ndarray,
    y: np.ndarray,
    scan_organ_ids: list,
    subgroup_annotations: dict,
    subgroup_name: str,
) -> tuple:
    """
    Filter features, labels, and scan_organ_ids for 'all' organs mode.
    Only includes organs that belong to the specified subgroup (or normal organs).
    
    Args:
        X: Feature array
        y: Label array
        scan_organ_ids: List of (scan_id, organ_name) tuples
        subgroup_annotations: Subgroup annotations dict {scan_id: {organ_name: {subgroup_name: value}}}
        subgroup_name: Name of the subgroup (e.g., 'diffuse', 'focal')
    
    Returns:
        (X_filtered, y_filtered, scan_organ_ids_filtered)
    """
    if len(X) == 0:
        return X, y, scan_organ_ids
    
    if len(scan_organ_ids) != len(X):
        raise ValueError(f"Mismatch: {len(scan_organ_ids)} scan_organ_ids but {len(X)} samples")
    
    filtered_indices = []
    for idx, (scan_id, organ_name) in enumerate(scan_organ_ids):
        # Include all normal samples (label=0)
        if y[idx] == 0:
            filtered_indices.append(idx)
        # Include abnormal samples (label=1) with the specified subgroup
        elif y[idx] == 1:
            if scan_id in subgroup_annotations:
                organ_subgroups = subgroup_annotations[scan_id].get(organ_name, {})
                # Check if this abnormal organ has the specified subgroup value=1
                if subgroup_name in organ_subgroups and organ_subgroups[subgroup_name] == 1:
                    filtered_indices.append(idx)
    
    if len(filtered_indices) == 0:
        return np.array([]), np.array([]), []
    
    filtered_indices = np.array(filtered_indices)
    return (
        X[filtered_indices],
        y[filtered_indices],
        [scan_organ_ids[i] for i in filtered_indices]
    )


def get_subgroup_info_all_organs(
    scan_id: str,
    organ_name: str,
    subgroup_annotations: Dict[str, Dict[str, Dict[str, int]]],
) -> Tuple[Optional[bool], Optional[bool]]:
    """
    Get focal and diffuse information for a scan ID and organ in 'all' organs mode.
    
    Args:
        scan_id: Scan ID
        organ_name: Name of the organ
        subgroup_annotations: Subgroup annotations dict
    
    Returns:
        (is_focal, is_diffuse) - booleans or None if not available
    """
    if scan_id not in subgroup_annotations:
        return None, None
    
    organ_subgroups = subgroup_annotations[scan_id].get(organ_name, {})
    is_focal = bool(organ_subgroups.get("focal", 0)) if "focal" in organ_subgroups else None
    is_diffuse = bool(organ_subgroups.get("diffuse", 0)) if "diffuse" in organ_subgroups else None
    
    return is_focal, is_diffuse


def load_raw_features_and_labels_all_organs(
    output_root: str,
    model_name: str,
    split: str,
    annotations: dict,
    return_scan_ids: bool = False,
) -> Tuple[List[np.ndarray], np.ndarray, Optional[list]]:
    """
    Load raw patch features and labels from all organs, treating each organ-scan combination as a sample.
    
    Args:
        output_root: Root directory for outputs
        model_name: Name of the feature model
        split: Split name (training, validation, test)
        annotations: Dictionary mapping scan_id to organ labels {scan_id: {organ_name: label}}
        return_scan_ids: If True, also return list of (scan_id, organ_name) tuples
    
    Returns:
        (patch_features_list, labels, scan_organ_ids) where:
        - patch_features_list is a list of arrays, each with shape (n_patches, flattened_dim)
        - labels is an array of labels
        - scan_organ_ids is a list of tuples: [(scan_id, organ_name), ...] if return_scan_ids=True
    """
    patch_features_list = []
    labels_list = []
    scan_organ_ids_list = []
    
    for organ_name in VALID_ORGANS:
        feature_dir = get_raw_feature_dir(output_root, model_name, organ_name, split)
        
        if not os.path.isdir(feature_dir):
            # Skip if feature directory doesn't exist for this organ
            continue
        
        feature_files = glob.glob(os.path.join(feature_dir, "*.npz"))
        
        for feature_file in sorted(feature_files):
            scan_id = os.path.basename(feature_file).replace(".npz", "")
            
            # Get label from annotations for this scan_id and organ
            if scan_id in annotations and organ_name in annotations[scan_id]:
                label = annotations[scan_id][organ_name]
                if label in [0, 1]:  # Only include valid labels
                    data = np.load(feature_file)
                    
                    # Skip placeholder files
                    is_placeholder = data.get("is_placeholder", False)
                    if is_placeholder:
                        continue
                    
                    patch_features = data["features"]  # Shape: (n_patches, ...)
                    
                    # Skip empty features
                    if patch_features.size == 0 or len(patch_features) == 0:
                        continue
                    
                    # Flatten each patch to 1D vector
                    if len(patch_features.shape) == 1:
                        # Single patch case - flatten and reshape to (1, flattened_dim)
                        patch_features_flat = patch_features.flatten()
                        patch_features = patch_features_flat.reshape(1, -1)
                    else:
                        # Multi-dimensional: (n_patches, ...) - flatten each patch individually
                        n_patches = patch_features.shape[0]
                        flattened_patches = []
                        for i in range(n_patches):
                            # Flatten each patch to 1D vector
                            flattened_patch = patch_features[i].flatten()
                            flattened_patches.append(flattened_patch)
                        patch_features = np.array(flattened_patches)  # Shape: (n_patches, flattened_dim)
                    
                    patch_features_list.append(patch_features)
                    labels_list.append(label)
                    if return_scan_ids:
                        scan_organ_ids_list.append((scan_id, organ_name))
    
    if len(patch_features_list) == 0:
        if return_scan_ids:
            return [], np.array([]), []
        return [], np.array([])
    
    if return_scan_ids:
        return patch_features_list, np.array(labels_list), scan_organ_ids_list
    return patch_features_list, np.array(labels_list)


def filter_patch_features_all_organs_with_scan_ids(
    patch_features_list: List[np.ndarray],
    y: np.ndarray,
    scan_organ_ids: list,
    subgroup_annotations: dict,
    subgroup_name: str,
) -> tuple:
    """
    Filter patch features, labels, and scan_organ_ids for 'all' organs mode.
    Only includes organs that belong to the specified subgroup (or normal organs).
    
    Args:
        patch_features_list: List of patch feature arrays
        y: Label array
        scan_organ_ids: List of (scan_id, organ_name) tuples
        subgroup_annotations: Subgroup annotations dict {scan_id: {organ_name: {subgroup_name: value}}}
        subgroup_name: Name of the subgroup (e.g., 'diffuse', 'focal')
    
    Returns:
        (patch_features_filtered, y_filtered, scan_organ_ids_filtered)
    """
    if len(patch_features_list) == 0:
        return patch_features_list, y, scan_organ_ids
    
    if len(scan_organ_ids) != len(patch_features_list):
        raise ValueError(f"Mismatch: {len(scan_organ_ids)} scan_organ_ids but {len(patch_features_list)} samples")
    
    filtered_features = []
    filtered_labels = []
    filtered_scan_organ_ids = []
    
    for idx, (scan_id, organ_name) in enumerate(scan_organ_ids):
        # Include all normal samples (label=0)
        if y[idx] == 0:
            filtered_features.append(patch_features_list[idx])
            filtered_labels.append(y[idx])
            filtered_scan_organ_ids.append((scan_id, organ_name))
        # Include abnormal samples (label=1) with the specified subgroup
        elif y[idx] == 1:
            if scan_id in subgroup_annotations:
                organ_subgroups = subgroup_annotations[scan_id].get(organ_name, {})
                # Check if this abnormal organ has the specified subgroup value=1
                if subgroup_name in organ_subgroups and organ_subgroups[subgroup_name] == 1:
                    filtered_features.append(patch_features_list[idx])
                    filtered_labels.append(y[idx])
                    filtered_scan_organ_ids.append((scan_id, organ_name))
    
    return filtered_features, np.array(filtered_labels), filtered_scan_organ_ids
