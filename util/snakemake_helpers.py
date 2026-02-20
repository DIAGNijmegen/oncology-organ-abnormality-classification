# Copyright Diagnostic Image Analysis Group, Radboudumc, Nijmegen, The Netherlands
# Licensed under Apache-2.0

"""
Helper functions for Snakemake pipeline setup.
Provides common initialization code for LEAVS dataset processing.
"""

import os
from typing import Dict, List, Tuple
from util.leavs_utils import (
    get_leavs_scans,
    parse_train_annotations,
    parse_test_annotations,
    create_train_val_split,
)

# Hard-coded list of valid organs
VALID_ORGANS = ['gallbladder', 'kidney_left', 'kidney_right', 'large_bowel', 'liver', 'pancreas', 'small_bowel', 'spleen', 'stomach']


def setup_leavs_dataset(
    dataset_root: str,
    val_ratio: float = 0.2,
    seed: int = 42,
    filter_valid_labels: bool = True
) -> Dict:
    """
    Set up LEAVS dataset information for Snakemake pipelines.
    
    Args:
        dataset_root: Root directory containing the LEAVS dataset
        val_ratio: Ratio for validation split (default: 0.2)
        seed: Random seed for train/val split (default: 42)
        filter_valid_labels: If True, only include organs with valid labels (0 or 1)
    
    Returns:
        Dictionary containing:
            - train_annotations: Dict of training annotations
            - test_annotations: Dict of test annotations
            - train_scan_ids: List of training scan IDs
            - test_scan_ids: List of test scan IDs
            - train_scan_ids_split: List of training scan IDs (after split)
            - val_scan_ids_split: List of validation scan IDs (after split)
            - get_scans_for_split_and_organ: Helper function
    """
    # Set up paths
    leavs_root = os.path.join(dataset_root, "LEAVS")
    train_csv = os.path.join(leavs_root, "amos_train_annotations.csv")
    test_csv = os.path.join(leavs_root, "amos_test_annotations.csv")
    
    # Parse annotations
    train_annotations = parse_train_annotations(train_csv)
    test_annotations = parse_test_annotations(test_csv)
    
    # Get scan IDs
    train_scan_ids, test_scan_ids = get_leavs_scans(dataset_root)
    
    # Create train/val split (80/20)
    train_scan_ids_split, val_scan_ids_split = create_train_val_split(
        train_scan_ids, val_ratio=val_ratio, seed=seed
    )
    
    # Define helper function
    def get_scans_for_split_and_organ(split: str, organ_name: str) -> List[str]:
        """Get scan IDs for a given split and organ."""
        if split == "training":
            scans = train_scan_ids_split
            annotations = train_annotations
        elif split == "validation":
            scans = val_scan_ids_split
            annotations = train_annotations
        else:  # test
            scans = test_scan_ids
            annotations = test_annotations
        
        valid_scans = []
        for scan_id in scans:
            if scan_id in annotations and organ_name in annotations[scan_id]:
                label = annotations[scan_id][organ_name]
                if not filter_valid_labels or label in [0, 1]:
                    valid_scans.append(scan_id)

        # Enforce deterministic ordering independent of directory/CSV traversal order.
        return sorted(valid_scans)
    
    return {
        "train_annotations": train_annotations,
        "test_annotations": test_annotations,
        "train_scan_ids": train_scan_ids,
        "test_scan_ids": test_scan_ids,
        "train_scan_ids_split": train_scan_ids_split,
        "val_scan_ids_split": val_scan_ids_split,
        "get_scans_for_split_and_organ": get_scans_for_split_and_organ,
    }
